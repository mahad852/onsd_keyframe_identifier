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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)
    
    parser.add_argument("--evaluation_dir", type=str, default=None)
    
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)

    parser.add_argument("--output_dir", type=str, required=True)

    return parser.parse_args()

def get_model(args):
    if args.model == "usfm":
        model_config_path = "models/USFM/configs/model.yaml"
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        
        config["model"]["model_cfg"]["num_classes"] = 0
        model =build_vit(config["model"]["model_cfg"])

    else:
        raise ValueError(f"Model {args.model} not supported currently.")
    return model

def get_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224,224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
    else:
        raise ValueError(f"Model {args.model} not supported currently.")


def l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def pick_topk_with_nms(scores: np.ndarray, topk: int = 3, nms_radius: int = 3) -> List[Dict[str, Any]]:
    """
    scores: shape [T] (float), higher is better
    Returns list of {"frame_idx": int, "score": float} sorted by frame_idx.
    """
    T = int(scores.shape[0])
    s = scores.copy()
    chosen: List[Tuple[int, float]] = []

    for _ in range(topk):
        t = int(np.argmax(s))
        best = float(s[t])
        if not np.isfinite(best) or best <= -1e8:
            break
        chosen.append((t, float(scores[t])))

        left = max(0, t - nms_radius)
        right = min(T, t + nms_radius + 1)
        s[left:right] = -1e9

    chosen = sorted(chosen, key=lambda x: x[0])
    return [{"frame_idx": t, "score": sc} for t, sc in chosen]


@torch.no_grad()
def score_eval_dataset_and_write_json(
    eval_ds,
    model: nn.Module,
    prototypes_dict: Dict[str, np.ndarray],  # key->(768,)
    output_dir: str,
    device: torch.device,
    tau: float = 2.0,
    topk: int = 3,
    nms_radius: int = 3,
):
    """
    Writes one JSON per DICOM with:
      - per-frame similarity scores
      - best matching prototype key per frame
      - selected frames (topk with NMS)
    Assumes eval_ds returns window clips for all frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # --- build prototype bank ---
    proto_keys = list(prototypes_dict.keys())
    proto_mat = np.stack([prototypes_dict[k] for k in proto_keys], axis=0)  # [M,768]
    proto_t = torch.from_numpy(proto_mat).to(device=device, dtype=torch.float32)  # [M,768]
    proto_t = l2_normalize(proto_t)

    # --- accumulators per dicom ---
    # dicom -> {"scores": {frame_idx: score}, "best_proto": {frame_idx: proto_key}, "num_frames_seen": int}
    per_dicom: Dict[str, Dict[str, Any]] = {}

    for obj in tqdm(eval_ds, desc="Scoring eval frames"):
        dicom = obj["dicom"]
        t = int(obj["frame_idx"])            # global center frame index
        x = obj["x"].to(device=device)       # [P,C,H,W]
        cp = int(obj["center_pos"])          # local center position in [0..P-1]

        feats = model(x)                     # [P,768]
        feats = l2_normalize(feats)

        P = feats.shape[0]
        dist = (torch.arange(P, device=device) - cp).abs().float()
        w = torch.exp(-dist / tau)
        w = w / (w.sum() + 1e-8)

        cand = (w[:, None] * feats).sum(dim=0)    # [768]
        cand = l2_normalize(cand)

        # cosine sim to all prototypes: [M]
        sims = proto_t @ cand                      # [M]
        best_sim, best_idx = torch.max(sims, dim=0)

        score = float(best_sim.item())
        best_key = proto_keys[int(best_idx.item())]

        if dicom not in per_dicom:
            per_dicom[dicom] = {
                "scores": {},
                "best_proto": {},
            }

        per_dicom[dicom]["scores"][t] = score
        per_dicom[dicom]["best_proto"][t] = best_key

    # --- write per-dicom jsons ---
    for dicom, d in per_dicom.items():
        # reconstruct dense arrays (assumes frames are 0..T-1; if not, still works with max index)
        frame_indices = sorted(d["scores"].keys())
        T = frame_indices[-1] + 1

        scores_arr = [None] * T
        best_arr = [None] * T
        for fi in frame_indices:
            scores_arr[fi] = float(d["scores"][fi])
            best_arr[fi] = d["best_proto"][fi]

        # fill missing frames (if any) with -inf for selection
        scores_for_select = np.array([(-1e9 if v is None else v) for v in scores_arr], dtype=np.float32)

        selected = pick_topk_with_nms(scores_for_select, topk=topk, nms_radius=nms_radius)

        out = {
            "dicom": dicom,
            "num_frames_scored": len(frame_indices),
            "T_assumed": T,
            "params": {
                "tau": tau,
                "topk": topk,
                "nms_radius": nms_radius,
                "num_prototypes": len(proto_keys),
            },
            "scores": scores_arr,              # list length T with floats or None
            "best_proto": best_arr,            # list length T with prototype key or None
            "selected": selected,              # list of dicts
        }

        out_path = os.path.join(output_dir, f"{os.path.splitext(dicom)[0]}_scores.json")
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)

    print(f"[OK] Wrote {len(per_dicom)} JSON files to: {output_dir}")

def get_prototype_vectors(ds: Dataset, model: nn.Module, device: torch.device, tau: float = 2.0):
    prototypes = {}
    model.eval()

    with torch.no_grad():
        for obj in tqdm(ds, desc="Deriving mean embedding prototypes"):
            imgs = obj["x"].to(device=device)         # [P,C,H,W]
            feats = model(imgs)                       # expect [P,D] (if not, you need pooling)
            if feats.ndim != 2:
                raise ValueError(f"Expected feats [P,D], got {feats.shape}. Add pooling here.")

            P = feats.shape[0]
            cp = int(obj["center_pos"])               # local index
            dist = (torch.arange(P, device=device) - cp).abs().float()
            w = torch.exp(-dist / tau)
            w = w / (w.sum() + 1e-8)

            proto = (w[:, None] * feats).sum(dim=0)   # [D]

            key = f"{obj['dicom']}_{int(obj['frame_idx'])}"
            prototypes[key] = proto.detach().cpu().numpy()

    return prototypes


def main():
    args = get_args()
    img_transforms = get_transforms(args)

    main_ds = DICOMKeyframeDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,   # can be None now (dataset must handle)
        context_window=10,
        transform=img_transforms,
        return_keyframes=True
    )

    eval_ds = None
    if args.evaluation_dir:
        eval_ds = DICOMKeyframeDataset(
            dicom_dir_path=args.evaluation_dir,
            json_dir_path=None,
            context_window=10,          # important: you want windows for scoring
            transform=img_transforms,
            side_margin=0,
            return_keyframes=False
        )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = get_model(args).to(device)

    main_prototypes = get_prototype_vectors(main_ds, model, device)
    
    score_eval_dataset_and_write_json(
        eval_ds=eval_ds,
        model=model,
        prototypes_dict=main_prototypes,
        output_dir=args.output_dir,
        device=device,
        tau=args.tau,
        topk=args.topk,
        nms_radius=args.nms_radius,
    )
    
if __name__ == "__main__":
    main()