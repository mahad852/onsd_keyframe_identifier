import argparse
import numpy as np
from dataset.DICOMKeyFrameDataset import DICOMKeyframeDataset
from models.USFM.models import build_vit
import yaml
from torchvision import transforms
from torch.utils.data import Dataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
from tqdm import tqdm
import torch
from typing import Dict, Any, List, Tuple
import json
import os
from models.USFM.models import build_seg_model
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)    
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    return parser.parse_args()

def get_image_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    raise ValueError(f"Model {args.model} not supported currently.")

def get_segmentation_logs(log_dir):
    with open(os.path.join(log_dir, "log_obj.json"), "r") as f:
        logs = json.load(f)
    return logs

def get_test_file_ids(logs):
    test_jsons = logs["test_jsons"]
    return list(map(lambda s: s.split(".")[0], test_jsons))

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

@torch.no_grad
def get_frame_score(model: nn.Module, img: torch.Tensor, class_name_to_id: dict) -> float:
    """
    img: [C,H,W] or [1,C,H,W] (already normalized)
    returns: float score in [0,1]
    """

    # forward (same as your training/eval)
    feats = model.backbone(model.data_preprocessor(img))
    out = model.decode_head(feats)
    semseg = out["pred"]  # [1,C,H,W] "prob-like" (from ATMHead semantic_inference)

    # enforce range-ish (ATMHead returns semseg = einsum(softmax * sigmoid), can exceed 1)
    # for scoring we just clamp to [0,1]
    semseg = semseg.clamp(0.0, 1.0)

    # predicted label map
    pred = semseg.argmax(dim=1)[0]   # [H,W]
    H, W = pred.shape
    HW = float(H * W)

    # ---- 1) confidence (non-background pixels only) ----
    nonbg = pred != 0
    if nonbg.any():
        # take the winning class probability at each pixel (like confidence of prediction)
        maxprob = semseg[0].amax(dim=0)  # [H,W]
        conf = maxprob[nonbg].mean().item()
    else:
        conf = 0.0

    # ---- 2) optic nerve area score ----
    # "bad" if <2% OR >15%; "fine" between. We'll make it smooth-ish.
    optic_id = class_name_to_id.get("Optic Nerve", None)
    if optic_id is None:
        area_score = 0.0
        optic_frac = 0.0
    else:
        optic_mask = (pred == int(optic_id))
        optic_frac = optic_mask.float().mean().item()  # fraction of image

        A_min = 0.02
        A_max = 0.15
        A_hard = 0.30  # decay to 0 by 30% (tunable)

        if optic_frac <= 0.0:
            area_score = 0.0
        elif optic_frac < A_min:
            area_score = float(optic_frac / A_min)  # ramp up 0..1
        elif optic_frac <= A_max:
            area_score = 1.0
        else:
            # decay from 1 at A_max to 0 at A_hard
            area_score = float(max(0.0, 1.0 - (optic_frac - A_max) / max(1e-6, (A_hard - A_max))))

    # ---- 3) presence score ----
    # "Presence" = are key structures visible?
    # Give globe+retina significantly more weight than dura/subarachnoid.
    # We'll compute per-class presence as:
    #   presence_i = clamp(area_frac / area_thresh, 0..1) * mean_prob_on_pred_pixels
    # where area_thresh is tiny (structure-dependent).
    def _presence_for_name(name: str, area_thresh: float) -> float:
        cid = class_name_to_id.get(name, None)
        if cid is None:
            return 0.0
        cid = int(cid)

        cls_pred = (pred == cid)
        if not cls_pred.any():
            return 0.0

        area_frac = cls_pred.float().mean().item()
        area_term = min(1.0, area_frac / max(1e-6, area_thresh))

        # confidence for this class on pixels predicted as this class
        cls_prob = semseg[0, cid]  # [H,W]
        prob_term = cls_prob[cls_pred].mean().item()

        return float(area_term * prob_term)

    # class-name normalization for your RETINA vs Retina
    # pick whichever key exists in the map
    retina_name = "RETINA" if "RETINA" in class_name_to_id else ("Retina" if "Retina" in class_name_to_id else "retina")

    globe_pres  = _presence_for_name("Globe", area_thresh=0.02)          # globe should be relatively large
    retina_pres = _presence_for_name(retina_name, area_thresh=0.005)     # retina is thinner
    # these are smaller: optic nerve / dura / subarachnoid
    on_pres     = _presence_for_name("Optic Nerve", area_thresh=0.003)

    # combine left/right if present
    rs_pres = _presence_for_name("Right Subarachnoid", area_thresh=0.0015)
    ls_pres = _presence_for_name("Left Subarachnoid",  area_thresh=0.0015)
    rd_pres = _presence_for_name("Right Dura", area_thresh=0.0015)
    ld_pres = _presence_for_name("Left Dura",  area_thresh=0.0015)

    sub_pres = max(rs_pres, ls_pres)  # "either visible"
    dura_pres = max(rd_pres, ld_pres)

    # weights: globe+retina dominate
    w_globe, w_retina, w_on, w_sub, w_dura = 0.38, 0.38, 0.12, 0.06, 0.06
    presence = (
        w_globe * globe_pres +
        w_retina * retina_pres +
        w_on * on_pres +
        w_sub * sub_pres +
        w_dura * dura_pres
    )

    # ---- final score ----
    score = 0.45 * conf + 0.30 * presence + 0.25 * area_score
    # clamp to [0,1] for sanity
    return float(max(0.0, min(1.0, score)))

def main():
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    logs = get_segmentation_logs(args.log_dir)

    test_file_ids = get_test_file_ids(logs)
    class_name_to_id = logs["class_name_to_id"] # {"Globe": 1, "Right Dura": 2, "Left Dura": 3, "Optic Nerve": 4, "Right Subarachnoid": 5, "Left Subarachnoid": 6, "RETINA": 7} - does not include bg, which 0

    num_classes = logs["num_classes"]
    model = get_model(args, num_classes=num_classes).to(device=device)
    model.load_state_dict(torch.load(os.path.join(args.log_dir, "best.pt"), map_location=device))

    img_transforms = get_image_transforms(args=args)

    ds = DICOMKeyframeDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,   # can be None now (dataset must handle)
        context_window=0,
        transform=img_transforms,
        return_keyframes=False
    )

    per_dicom_stats = {}

    model.eval()

    for obj in tqdm(ds, desc="Computing frame scores"):
        dicom_name = obj["dicom"]
        frame_idx = obj["frame_idx"]
        
        img = obj["x"] # context_window=0 should give [1,C,H,W]
        
        is_kf = is_kf = bool(obj["is_keyframe"][0]) if torch.is_tensor(obj["is_keyframe"]) else bool(obj["is_keyframe"])

        file_id = dicom_name.split(".")[0]
        if file_id not in test_file_ids:
            continue

        img = img.to(device=device)

        if dicom_name not in per_dicom_stats:
            per_dicom_stats[dicom_name] = {
                "actual_keyframes": [],
                "frame_scores": {}
            }

        if is_kf:
            per_dicom_stats[dicom_name]["actual_keyframes"].append(frame_idx)

        per_dicom_stats[dicom_name]["frame_scores"][frame_idx] = get_frame_score(model, img=img, class_name_to_id=class_name_to_id)


    json_path = os.path.join(args.log_dir, "frame_segmentation_scores.json")

    with open(json_path, "w") as f:
        json.dump(per_dicom_stats, f, indent=2)

if __name__ == "__main__":
    main()