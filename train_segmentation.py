from dataset.SegmentationDataset import DICOMSegmentationDataset
import argparse
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import random
from torch.utils.data import DataLoader
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
from models.USFM.models import build_seg_model
import yaml
import torch.optim as Optim
from tqdm import tqdm
from utils.segmentation import compute_confusion_stats, dice_iou_from_stats, semantic_map_like_ap

import json
import cv2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--log_dir", type=str, default="logs/")
    return parser.parse_args()


class MaskResizeOnly:
    def __init__(self, size=(224, 224)):
        self.size = size

    def __call__(self, mask_pil):
        mask_pil = F.resize(mask_pil, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return torch.from_numpy(np.array(mask_pil)).long()


def get_image_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    raise ValueError(f"Model {args.model} not supported currently.")


def get_mask_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([MaskResizeOnly(size=(224, 224))])
    raise ValueError(f"Model {args.model} not supported currently.")


def get_model(args, num_classes) -> torch.nn.Module:
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/SegVit.yaml"
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)

        config["model"]["model_cfg"]["decode_head"]["num_classes"] = num_classes
        config["model"]["model_cfg"]["decode_head"]["loss_decode"]["num_classes"] = num_classes
        config["model"]["model_cfg"]["decode_head"]["img_size"] = 224
        config["model"]["model_cfg"]["backbone"]["img_size"] = 224

        model = build_seg_model(config["model"]["model_cfg"])
        return model

    raise ValueError(f"Model {args.model} not supported currently.")


def get_train_test_jsons(args):
    all_json_fnames = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]
    random.shuffle(all_json_fnames)

    num_train = int(len(all_json_fnames) * 0.70)
    train = all_json_fnames[:num_train]
    test = all_json_fnames[num_train:]
    return train, test


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int, ignore_bg: bool = True):
    model.eval()

    total_tp = torch.zeros(num_classes, device=device, dtype=torch.float32)
    total_fp = torch.zeros(num_classes, device=device, dtype=torch.float32)
    total_fn = torch.zeros(num_classes, device=device, dtype=torch.float32)

    all_ap_preds = []
    all_ap_gts = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        imgs = batch["image"].to(device=device)
        gt = batch["mask"].to(device=device).long()  # [B,H,W]

        feats = model.backbone(model.data_preprocessor(imgs))
        out = model.decode_head(feats)
        semseg = out["pred"]            # [B,C,H,W]
        pred = semseg.argmax(dim=1)     # [B,H,W]

        tp, fp, fn = compute_confusion_stats(pred, gt, num_classes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ap_preds.append(pred.detach().cpu())
        all_ap_gts.append(gt.detach().cpu())

    dice, iou = dice_iou_from_stats(total_tp, total_fp, total_fn)

    if ignore_bg:
        dice_mean = dice[1:].mean().item() if num_classes > 1 else dice.mean().item()
        iou_mean = iou[1:].mean().item() if num_classes > 1 else iou.mean().item()
    else:
        dice_mean = dice.mean().item()
        iou_mean = iou.mean().item()

    pred_cat = torch.cat(all_ap_preds, dim=0)
    gt_cat = torch.cat(all_ap_gts, dim=0)
    map_like, prec_thr, rec_thr = semantic_map_like_ap(
        pred_cat, gt_cat, num_classes=num_classes, ignore_bg=ignore_bg
    )

    return {
        "dice_per_class": dice.detach().cpu().tolist(),
        "iou_per_class": iou.detach().cpu().tolist(),
        "dice_mean": dice_mean,
        "iou_mean": iou_mean,
        "mAP_like_50_95": map_like,
        "prec_by_thr": prec_thr,
        "rec_by_thr": rec_thr,
        "tp": total_tp.detach().cpu().tolist(),
        "fp": total_fp.detach().cpu().tolist(),
        "fn": total_fn.detach().cpu().tolist(),
    }


# -------------------- NEW: per-frame eval + save top overlays --------------------

def _denormalize_img(img_chw: torch.Tensor) -> torch.Tensor:
    """
    img_chw: float tensor [3,H,W] normalized with IMAGENET_DEFAULT_MEAN/STD.
    returns float tensor [3,H,W] in [0,1] (clamped).
    """
    mean = torch.tensor(IMAGENET_DEFAULT_MEAN, device=img_chw.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_DEFAULT_STD, device=img_chw.device).view(3, 1, 1)
    x = img_chw * std + mean
    return x.clamp(0.0, 1.0)


def _to_uint8_hwc(img_chw_01: torch.Tensor) -> np.ndarray:
    """
    img_chw_01: [3,H,W] in [0,1]
    returns uint8 HWC RGB
    """
    x = (img_chw_01 * 255.0).round().clamp(0, 255).to(torch.uint8)
    return x.permute(1, 2, 0).cpu().numpy()  # HWC RGB uint8


def _palette(num_classes: int) -> np.ndarray:
    # deterministic simple palette (RGB)
    base = np.array([
        [0, 0, 0],        # bg
        [230, 25, 75],
        [60, 180, 75],
        [255, 225, 25],
        [0, 130, 200],
        [245, 130, 48],
        [145, 30, 180],
        [70, 240, 240],
        [240, 50, 230],
        [210, 245, 60],
    ], dtype=np.uint8)
    if num_classes <= base.shape[0]:
        return base[:num_classes]
    # extend if needed
    out = [base[i % base.shape[0]] for i in range(num_classes)]
    return np.stack(out, axis=0)


def _colorize_mask(mask_hw: np.ndarray, pal_rgb: np.ndarray) -> np.ndarray:
    # mask_hw: int HxW, values in [0..C-1]
    return pal_rgb[mask_hw]  # HWC RGB


@torch.no_grad()
def evaluate_per_frame_and_save_overlays(
    model,
    loader,
    device,
    num_classes: int,
    log_obj: dict,
    log_dir: str,
    ignore_bg: bool = True,
    topk_save: int = 5,
):
    model.eval()
    log_obj.setdefault("per_dicom_stats", {})

    # Keep best frames by dice_mean (foreground mean by default)
    best_frames = []  # list of (dice_mean, dicom, frame_idx, rgb_uint8, pred_hw, gt_hw)

    pal = _palette(num_classes)

    for batch in tqdm(loader, desc="Best-model per-frame eval", leave=True):
        imgs = batch["image"].to(device=device)     # [B,3,224,224]
        gt = batch["mask"].to(device=device).long() # [B,224,224]
        dicoms = batch["dicom"]                     # list[str] (usually)
        frame_idxs = batch["frame_idx"]             # tensor/list
        orig_dims = batch["orig_dims"]              # e.g. [B,2] or list[tuple]

        feats = model.backbone(model.data_preprocessor(imgs))
        out = model.decode_head(feats)
        semseg = out["pred"]                        # [B,C,224,224] (semseg probs/logits-like)
        pred = semseg.argmax(dim=1)                 # [B,224,224]

        # per-sample stats
        B = pred.shape[0]
        for b in range(B):
            dicom_name = dicoms[b]
            # frame_idx could be tensor scalar
            fi = int(frame_idxs[b].item()) if torch.is_tensor(frame_idxs) else int(frame_idxs[b])

            # orig_dims might be tensor [B,2] or list/tuple
            if torch.is_tensor(orig_dims):
                H0 = int(orig_dims[b, 0].item())
                W0 = int(orig_dims[b, 1].item())
            else:
                # could be tuple/list like (H,W)
                H0 = int(orig_dims[b][0])
                W0 = int(orig_dims[b][1])

            pred_b = pred[b]  # [224,224]
            gt_b = gt[b]      # [224,224]

            # confusion stats per sample
            tp, fp, fn = compute_confusion_stats(pred_b[None, ...], gt_b[None, ...], num_classes)
            dice, iou = dice_iou_from_stats(tp, fp, fn)

            if ignore_bg and num_classes > 1:
                dice_mean = float(dice[1:].mean().item())
                iou_mean = float(iou[1:].mean().item())
            else:
                dice_mean = float(dice.mean().item())
                iou_mean = float(iou.mean().item())

            # store per-frame
            per_d = log_obj["per_dicom_stats"].setdefault(dicom_name, {})
            per_d[str(fi)] = {
                "dice_mean": dice_mean,
                "iou_mean": iou_mean,
                "dice_per_class": dice.detach().cpu().tolist(),
                "iou_per_class": iou.detach().cpu().tolist(),
                "tp": tp.detach().cpu().tolist(),
                "fp": fp.detach().cpu().tolist(),
                "fn": fn.detach().cpu().tolist(),
            }

            # candidate for top-k saves
            # Build resized overlays to original dims
            # - image: denorm -> uint8 RGB -> resize to (W0,H0)
            img_b = _denormalize_img(imgs[b])
            img_rgb = _to_uint8_hwc(img_b)  # RGB uint8 [224,224,3]
            img_rgb = cv2.resize(img_rgb, (W0, H0), interpolation=cv2.INTER_LINEAR)

            # - masks: resize via nearest (keep labels)
            pred_np = pred_b.detach().cpu().numpy().astype(np.int32)
            gt_np = gt_b.detach().cpu().numpy().astype(np.int32)

            pred_np = cv2.resize(pred_np, (W0, H0), interpolation=cv2.INTER_NEAREST)
            gt_np = cv2.resize(gt_np, (W0, H0), interpolation=cv2.INTER_NEAREST)

            # keep top-k
            best_frames.append((dice_mean, dicom_name, fi, img_rgb, pred_np, gt_np))

    # Save top-k overlays
    best_frames.sort(key=lambda x: x[0], reverse=True)
    best_frames = best_frames[:topk_save]

    out_root = os.path.join(log_dir, "overlaid")
    os.makedirs(out_root, exist_ok=True)

    for dice_mean, dicom_name, fi, img_rgb, pred_np, gt_np in best_frames:
        dicom_dir = os.path.join(out_root, os.path.splitext(dicom_name)[0])
        os.makedirs(dicom_dir, exist_ok=True)

        pred_col = _colorize_mask(pred_np, pal)  # RGB
        gt_col = _colorize_mask(gt_np, pal)      # RGB

        # overlay pred strongly, gt lightly (so you can see mismatches)
        overlay = img_rgb.copy()
        overlay = cv2.addWeighted(overlay, 1.0, pred_col, 0.45, 0)
        overlay = cv2.addWeighted(overlay, 1.0, gt_col, 0.20, 0)

        # add label text
        txt = f"dice={dice_mean:.4f}  {dicom_name}  frame={fi}"
        cv2.putText(
            overlay, txt, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (255, 255, 255), 2, cv2.LINE_AA
        )
        cv2.putText(
            overlay, txt, (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 1, cv2.LINE_AA
        )

        out_path = os.path.join(dicom_dir, f"{fi:03d}.png")
        # OpenCV expects BGR for imwrite
        cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def main():
    args = get_args()

    metadata_file = os.path.join(args.json_dir, ".v7", "metadata.json")
    train_jsons, test_jsons = get_train_test_jsons(args)

    img_transforms = get_image_transforms(args)
    mask_transforms = get_mask_transforms(args)

    train_ds = DICOMSegmentationDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        metadata_file=metadata_file,
        jsons_to_include=train_jsons,
        transform=img_transforms,
        mask_transform=mask_transforms
    )
    test_ds = DICOMSegmentationDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        metadata_file=metadata_file,
        jsons_to_include=test_jsons,
        transform=img_transforms,
        mask_transform=mask_transforms
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = get_model(args, num_classes=train_ds.get_num_classes()).to(device=device)

    print(train_ds.get_num_classes(), train_ds.get_class_name_to_id(), len(train_ds))
    print(test_ds.get_num_classes(), test_ds.get_class_name_to_id(), len(test_ds))

    os.makedirs(args.log_dir, exist_ok=True)

    log_obj = {}
    log_obj["test_jsons"] = test_jsons
    log_obj["train_jsons"] = train_jsons
    log_obj["class_name_to_id"] = train_ds.get_class_name_to_id()
    log_obj["num_classes"] = train_ds.get_num_classes()

    best_model_path = os.path.join(args.log_dir, "best.pt")
    best_dice = 0.0
    best_epoch = 0

    optim = Optim.Adam(params=model.parameters(), lr=1e-4)

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=True)
        for step, batch in enumerate(pbar):
            imgs = batch["image"].to(device=device)
            masks = batch["mask"].to(device=device).long()

            optim.zero_grad(set_to_none=True)

            extra_features = model.backbone(model.data_preprocessor(imgs))
            loss, outputs, labels = model.decode_head.forward_with_loss(extra_features, masks)

            loss.backward()
            optim.step()

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            num_classes=train_ds.get_num_classes(),
            ignore_bg=True,
        )

        if metrics["dice_mean"] > best_dice:
            best_dice = metrics["dice_mean"]
            best_epoch = epoch + 1
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"val_dice={metrics['dice_mean']:.4f} | "
            f"val_iou={metrics['iou_mean']:.4f} | "
            f"mAP(0.50:0.95)~={metrics['mAP_like_50_95']:.4f}"
        )
        print(f"Best Dice: {best_dice:.4f} at epoch: {best_epoch}")

    # -------------------- NEW: load best + per-frame eval + overlays + save log --------------------
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"\nLoaded best model from: {best_model_path} (epoch={best_epoch}, dice={best_dice:.4f})\n")
    else:
        print(f"\n[WARN] best model not found at {best_model_path}; using last epoch weights.\n")

    log_obj["best_epoch"] = best_epoch
    log_obj["best_dice"] = best_dice

    # Per-frame stats + save top-5 overlays
    evaluate_per_frame_and_save_overlays(
        model=model,
        loader=test_loader,
        device=device,
        num_classes=train_ds.get_num_classes(),
        log_obj=log_obj,
        log_dir=args.log_dir,
        ignore_bg=True,
        topk_save=5,
    )

    # Save log json
    out_json = os.path.join(args.log_dir, "log_obj.json")
    with open(out_json, "w") as f:
        json.dump(log_obj, f, indent=2)
    print(f"\nSaved log_obj with per-frame stats to: {out_json}")
    print(f"Saved top-5 overlays to: {os.path.join(args.log_dir, 'overlaid')}\n")


if __name__ == "__main__":
    main()