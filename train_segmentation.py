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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)  
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)

    return parser.parse_args()

class MaskResizeOnly:
    def __init__(self, size=(224,224)):
        self.size = size

    def __call__(self, mask_pil):
        
        mask_pil = F.resize(mask_pil, self.size, interpolation=transforms.InterpolationMode.NEAREST)
        return torch.from_numpy(np.array(mask_pil)).long()

def get_image_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    
def get_mask_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            MaskResizeOnly(size=(224, 224))
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    
def get_model(args, num_classes):
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/SegVit.yaml"

        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)

        config["model"]["model_cfg"]["decode_head"]["num_classes"] = num_classes
        config["model"]["model_cfg"]["decode_head"]["loss_decode"]["num_classes"] = num_classes

        config["model"]["model_cfg"]["decode_head"]["img_size"] = 224
        config["model"]["model_cfg"]["backbone"]["img_size"] = 224

        model = build_seg_model(config["model"]["model_cfg"])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")

    return model

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

    # for mAP-like
    all_ap_preds = []
    all_ap_gts = []

    for batch in tqdm(loader, desc="Evaluating", leave=False):
        imgs = batch["image"].to(device=device)
        gt   = batch["mask"].to(device=device).long()  # [B,H,W]

        # forward
        feats = model.backbone(model.data_preprocessor(imgs))
        out = model.decode_head(feats)  # dict with "pred"
        semseg = out["pred"]            # [B,C,H,W] (prob-like)
        pred = semseg.argmax(dim=1)     # [B,H,W]

        tp, fp, fn = compute_confusion_stats(pred, gt, num_classes)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        all_ap_preds.append(pred.detach().cpu())
        all_ap_gts.append(gt.detach().cpu())

    dice, iou = dice_iou_from_stats(total_tp, total_fp, total_fn)

    # mean over foreground classes by default
    if ignore_bg:
        dice_mean = dice[1:].mean().item() if num_classes > 1 else dice.mean().item()
        iou_mean  = iou[1:].mean().item()  if num_classes > 1 else iou.mean().item()
    else:
        dice_mean = dice.mean().item()
        iou_mean  = iou.mean().item()

    # mAP-like on CPU tensors
    pred_cat = torch.cat(all_ap_preds, dim=0)
    gt_cat   = torch.cat(all_ap_gts, dim=0)
    map_like, prec_thr, rec_thr = semantic_map_like_ap(
        pred_cat, gt_cat, num_classes=num_classes, ignore_bg=ignore_bg
    )

    # return everything you might want to print/log
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

    model = get_model(args, num_classes=train_ds.get_num_classes())
    model = model.to(device=device)

    print(train_ds.get_num_classes(), train_ds.get_class_name_to_id(), len(train_ds))
    print(test_ds.get_num_classes(), test_ds.get_class_name_to_id(), len(test_ds))
    
    n = 0

    optim = Optim.Adam(params=model.parameters(), lr=1e-4)

    for batch in test_ds:
        for mask in batch["mask"]:
            print("Classes available:", mask.unique())

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
            num_classes=train_ds.get_num_classes(),  # includes background
            ignore_bg=True,
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"val_dice={metrics['dice_mean']:.4f} | "
            f"val_iou={metrics['iou_mean']:.4f} | "
            f"mAP(0.50:0.95)~={metrics['mAP_like_50_95']:.4f}"
        )

if __name__ == "__main__":
    main()
