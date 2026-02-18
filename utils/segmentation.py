import torch
import numpy as np
from dataset.DICOMDataset import get_bmode_region

def crop_ultrasound_mask_np(ds, mask: np.ndarray, side_margin: int = 125) -> np.ndarray:
    """
    Crop segmentation mask using Ultrasound Region metadata.
    Supports:
        [H, W]
        [T, H, W]
        [C, H, W]  (e.g. one-hot masks)

    side_margin: pixels removed horizontally from both sides
                 after region cropping.

    Returns cropped mask with same ndim.
    """
    region = get_bmode_region(ds)
    if region is None:
        return mask

    x0, y0, x1, y1 = region

    # Determine spatial dims
    if mask.ndim == 2:
        H, W = mask.shape
        spatial_slice = lambda m: m[y0:y1, :]
    elif mask.ndim == 3:
        # could be [T,H,W] OR [C,H,W]
        H, W = mask.shape[-2:]
        spatial_slice = lambda m: m[..., y0:y1, :]
    else:
        raise ValueError(f"Unexpected mask shape: {mask.shape}")

    # Clamp bounds
    x0 = max(0, min(x0, W))
    x1 = max(0, min(x1, W))
    y0 = max(0, min(y0, H))
    y1 = max(0, min(y1, H))

    # Horizontal margin safety
    left  = x0 + side_margin
    right = x1 - side_margin

    # If margin destroys region, fallback to region without margin
    if left >= right:
        left, right = x0, x1

    # Perform crop
    if mask.ndim == 2:
        return mask[y0:y1, left:right]
    else:
        # works for [T,H,W] or [C,H,W]
        return mask[..., y0:y1, left:right]

@torch.no_grad()
def compute_confusion_stats(pred: torch.Tensor, gt: torch.Tensor, num_classes: int):
    """
    pred, gt: [B,H,W] long
    returns per-class tp, fp, fn as float tensors [C]
    """
    tp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fp = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)
    fn = torch.zeros(num_classes, device=pred.device, dtype=torch.float32)

    for c in range(num_classes):
        pred_c = (pred == c)
        gt_c   = (gt == c)
        tp[c] = (pred_c & gt_c).sum().float()
        fp[c] = (pred_c & (~gt_c)).sum().float()
        fn[c] = ((~pred_c) & gt_c).sum().float()

    return tp, fp, fn


def dice_iou_from_stats(tp, fp, fn, eps=1e-7):
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou  = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou


@torch.no_grad()
def semantic_map_like_ap(
    pred: torch.Tensor,
    gt: torch.Tensor,
    num_classes: int,
    thresholds=None,
    ignore_bg=True,
    eps=1e-7,
):
    """
    mAP-like metric for semantic segmentation:
    For each image and each class, treat the class mask as a single prediction.
    A prediction is TP if IoU(class) >= thr and GT(class) exists.
    Aggregates precision/recall across dataset and averages precision across thresholds.

    pred, gt: [B,H,W] long
    """
    if thresholds is None:
        thresholds = [x / 100 for x in range(50, 100, 5)]  # 0.50..0.95

    cls_range = range(1, num_classes) if ignore_bg else range(num_classes)

    # counts per threshold
    tp_t = torch.zeros(len(thresholds), dtype=torch.float64)
    fp_t = torch.zeros(len(thresholds), dtype=torch.float64)
    fn_t = torch.zeros(len(thresholds), dtype=torch.float64)

    B = pred.shape[0]
    for b in range(B):
        for c in cls_range:
            pred_c = (pred[b] == c)
            gt_c   = (gt[b] == c)

            pred_any = bool(pred_c.any().item())
            gt_any   = bool(gt_c.any().item())

            if not pred_any and not gt_any:
                continue

            if pred_any and gt_any:
                inter = (pred_c & gt_c).sum().float()
                union = (pred_c | gt_c).sum().float()
                iou = (inter + eps) / (union + eps)
            else:
                iou = torch.tensor(0.0)

            for i, thr in enumerate(thresholds):
                if pred_any and gt_any:
                    if iou >= thr:
                        tp_t[i] += 1
                    else:
                        # predicted this class but doesn't match GT well enough
                        fp_t[i] += 1
                        fn_t[i] += 1
                elif pred_any and not gt_any:
                    fp_t[i] += 1
                elif (not pred_any) and gt_any:
                    fn_t[i] += 1

    prec = tp_t / (tp_t + fp_t + eps)
    rec  = tp_t / (tp_t + fn_t + eps)
    ap = prec.mean().item()  # average precision across IoU thresholds (simple)
    return ap, prec.tolist(), rec.tolist()
