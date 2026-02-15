from dataset.DICOMDataset import DICOMDataset
from models.USFM.USFMClip import USFMClip
import argparse
import torch
from torch.utils.data import DataLoader
import os
import random
from tqdm import tqdm
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.optim as Optim
import torch.nn as nn

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=5)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()

def get_model(args):
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/model.yaml"
        model = USFMClip(model_config_path=model_config_path, clip_size=args.context_window * 2 + 1, finetune_encoder=False)
    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    return model

def get_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(224,224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")

def get_train_test_jsons(args):
    all_json_fnames = [f for f in os.listdir(args.json_dir) if f.endswith(".json")]
    random.shuffle(all_json_fnames)

    num_train = int(len(all_json_fnames) * 0.70)
    train = all_json_fnames[:num_train]
    test = all_json_fnames[num_train:]
    return train, test

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    total_frames = 0
    correct_frames = 0

    total_clips = 0
    correct_clips = 0

    tp = fp = tn = fn = 0

    for batch in loader:
        clip = batch["clip"].to(device=device)
        y = batch["keyframe_mask"].to(device=device, dtype=torch.float32)  # [B,P]

        logits = model(clip)  # [B,P]
        probs = torch.sigmoid(logits)
        yhat = (probs >= threshold).to(torch.int64)   # [B,P]
        yint = y.to(torch.int64)

        # frame-wise accuracy
        matches = (yhat == yint)
        correct_frames += matches.sum().item()
        total_frames += matches.numel()

        # clip-wise accuracy: all frames must match
        clip_correct = matches.all(dim=1)  # [B]
        correct_clips += clip_correct.sum().item()
        total_clips += clip_correct.numel()

        # confusion counts (frame-wise)
        tp += ((yhat == 1) & (yint == 1)).sum().item()
        fp += ((yhat == 1) & (yint == 0)).sum().item()
        tn += ((yhat == 0) & (yint == 0)).sum().item()
        fn += ((yhat == 0) & (yint == 1)).sum().item()

    frame_acc = correct_frames / max(1, total_frames)
    clip_acc = correct_clips / max(1, total_clips)

    precision = tp / max(1, tp + fp)
    recall    = tp / max(1, tp + fn)
    f1 = (2 * precision * recall) / max(1e-12, (precision + recall))

    return {
        "frame_acc": frame_acc,
        "clip_acc": clip_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn
    }

def main():
    args = get_args()
    train_jsons, test_jsons = get_train_test_jsons(args)

    img_transforms = get_transforms(args)

    train_ds = DICOMDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        context_window=args.context_window,
        pad_mode="zero",
        jsons_to_include=train_jsons,
        transform=img_transforms,
        zero_prob=0.0
    )
    test_ds = DICOMDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,
        context_window=args.context_window,
        pad_mode="zero",
        jsons_to_include=test_jsons,
        transform=img_transforms,
        zero_prob=0.0
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args).to(device)

    optim = Optim.Adam(params=model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([35], device=device))

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)

        for step, batch in enumerate(pbar):
            clip = batch["clip"].to(device=device, non_blocking=True)
            y = batch["keyframe_mask"].to(device=device, dtype=torch.float32, non_blocking=True)  # [B,P]

            optim.zero_grad(set_to_none=True)
            logits = model(clip)  # [B,P]

            loss = criterion(logits, y)  # <- correct order: (pred, target)
            loss.backward()
            optim.step()

            running_loss += loss.item()
            avg_loss = running_loss / (step + 1)
            pbar.set_postfix(loss=f"{avg_loss:.4f}")

        # Validation
        metrics = evaluate(model, test_loader, device, threshold=args.threshold)

        # One-line epoch summary
        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"val_frame_acc={metrics['frame_acc']:.3f} | "
            f"val_clip_acc={metrics['clip_acc']:.3f} | "
            f"prec={metrics['precision']:.3f} | "
            f"rec={metrics['recall']:.3f} | "
            f"f1={metrics['f1']:.3f} | "
            f"(tp={metrics['tp']}, fp={metrics['fp']}, tn={metrics['tn']}, fn={metrics['fn']})"
        )

if __name__ == "__main__":
    main()
