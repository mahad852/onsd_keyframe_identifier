from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pydicom
import torch
from torch.utils.data import Dataset
from dataset.DICOMDataset import to_uint8, crop_ultrasound_frames, np_gray_to_pil
import cv2
from tqdm import tqdm
# ---------- utilities you already have (kept minimal) ----------


def _read_dicom_frames(dicom_path: str) -> Tuple[Any, np.ndarray]:
    """
    Returns (ds, frames) where frames is always (T,H,W) in numpy.
    """
    ds = pydicom.dcmread(dicom_path)
    px = ds.pixel_array  # (H,W) or (T,H,W) depending on file

    if px.shape[-1] == 3:
        if px.ndim == 4:
            px = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in px])
        else:
            px = cv2.cvtColor(px, cv2.COLOR_BGR2GRAY)
        
    if px.ndim == 2:
        px = px[None, ...]
    elif px.ndim == 3:
        pass
    else:
        raise ValueError(f"Unexpected DICOM pixel_array shape: {px.shape}")
    return ds, px


def _apply_transform_frame(frame_hw: np.ndarray, transform) -> torch.Tensor:
    """
    Apply torchvision-style transform.
    - converts to uint8 for PIL-based transforms if needed
    - if transform starts with ToTensor(), it can take numpy uint8 (H,W) or (H,W,C)
    """
    # Keep as grayscale 2D by default.
    frame_u8 = to_uint8(frame_hw) if frame_hw.dtype != np.uint8 else frame_hw
    if transform is None:
        # default: return float tensor [1,H,W] in [0,1]
        t = torch.from_numpy(frame_u8).float() / 255.0
        return t.unsqueeze(0)
    frame_pil = np_gray_to_pil(img_u8=frame_u8)
    return transform(frame_pil)


# ---------- dataset ----------

@dataclass
class _ItemIndex:
    json_fname: str
    dicom_fname: str
    frame_idx: int
    # if return_keyframes=True, this indicates center is a keyframe
    is_keyframe_center: bool


class DICOMKeyframeDataset(Dataset):
    """
    Modes:
      - return_keyframes=True:
          Each sample corresponds to one keyframe center.
          Returns clip of variable length: available frames within [t-k, t+k].
      - return_keyframes=False:
          Each sample corresponds to one frame (no context).
          Returns a single frame.
    Always returns:
      {
        "dicom": <dicom filename>,
        "frame_idx": <center idx>,
        "x": Tensor either [P,C,H,W] (clip) or [C,H,W] (single),
        "is_keyframe": bool or 1D bool tensor length P (for clips)
      }
    Notes:
      - Cropping is applied to frames via Ultrasound Region + side margin.
      - Keyframe indices come from JSON annotations' "keyframe": true fields,
        excluding annotation name "Brest Frame".
    """

    def __init__(
        self,
        dicom_dir_path: str,
        return_keyframes: bool,
        json_dir_path: str = None,
        context_window: int = 5,
        transform=None,
        jsons_to_include: Optional[List[str]] = None,
        side_margin: int = 125,
    ):
        self.dicom_dir_path = dicom_dir_path
        self.json_dir_path = json_dir_path
        self.return_keyframes = bool(return_keyframes)
        self.context_window = int(context_window)
        self.transform = transform
        self.jsons_to_include = set(jsons_to_include) if jsons_to_include else None
        self.side_margin = int(side_margin)

        self._summary = self._build_summary()  # json_fname -> {dicom, num_frames, keyframes:set}
        self._keyframe_lookup = {
            k: set(v["keyframes"]) for k, v in self._summary.items()
        }
        self._index: List[_ItemIndex] = self._build_index()

        self.cache = (None, None)

    def __len__(self) -> int:
        return len(self._index)
    
    def _build_summary(self) -> Dict[str, Dict[str, Any]]:
        if self.json_dir_path:
            return self._build_summary_with_json_dir()
        
        return self._build_summary_without_json_dir()

    def _build_summary_without_json_dir(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}

        for fname in os.listdir(self.dicom_dir_path):
            dicom_fpath = os.path.join(self.dicom_dir_path, fname)

            try:
                ds = pydicom.dcmread(dicom_fpath)
            except Exception:
                continue

            px = ds.pixel_array
            
            if px.shape[-1] == 3:
                num_frames = 1 if px.ndim == 3 else px.shape[0]
            else:
                num_frames = 1 if px.ndim == 2 else px.shape[0]

            summary[fname] = {
                "dicom": fname,
                "num_frames": num_frames,
                "keyframes": [],
            }
        return summary

    def _build_summary_with_json_dir(self) -> Dict[str, Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}

        for json_fname in sorted(os.listdir(self.json_dir_path)):
            if not json_fname.endswith(".json"):
                continue
            if self.jsons_to_include is not None and json_fname not in self.jsons_to_include:
                continue

            json_fpath = os.path.join(self.json_dir_path, json_fname)
            with open(json_fpath, "r") as f:
                obj = json.load(f)

            dicom_name = obj["item"]["name"]
            # Some exports say frame_count=1 even for multiframe; prefer DICOM truth at runtime.
            num_frames = int(obj["item"]["slots"][0].get("frame_count", 1))

            keyframes = set()
            for ann in obj.get("annotations", []):
                name = ann.get("name", "")
                if name == "Brest Frame":
                    continue
                frames = ann.get("frames", {})
                for k, v in frames.items():
                    try:
                        fi = int(k)
                        if isinstance(v, dict) and v.get("keyframe", False):
                            keyframes.add(fi)
                    except Exception:
                        pass

            summary[json_fname] = {
                "dicom": dicom_name,
                "num_frames": num_frames,
                "keyframes": sorted(keyframes),
            }

        if len(summary) == 0:
            raise RuntimeError(f"No JSONs found in {self.json_dir_path} (after filtering).")

        return summary

    def _build_index(self) -> List[_ItemIndex]:
        idx: List[_ItemIndex] = []
        for json_fname, info in self._summary.items():
            dicom_fname = info["dicom"]
            num_frames = int(info["num_frames"])
            kfs = set(info["keyframes"])

            if self.return_keyframes:
                for t in sorted(kfs):
                    idx.append(_ItemIndex(json_fname, dicom_fname, t, True))
            else:
                # all frames, single frame samples
                for t in range(num_frames):
                    idx.append(_ItemIndex(json_fname, dicom_fname, t, t in kfs))
        return idx

    def __getitem__(self, index: int) -> Dict[str, Any]:
        it = self._index[index]
        dicom_path = os.path.join(self.dicom_dir_path, it.dicom_fname)

        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"DICOM not found: {dicom_path}")
        
        if not self.cache[0] or self.cache[0] != it.dicom_fname:
            (ds, frames) = _read_dicom_frames(dicom_path)  # (T,H,W)
            self.cache = (it.dicom_fname, (ds, frames))
        else:
            ds, frames = self.cache[1]        

        frames = crop_ultrasound_frames(ds, frames, side_margin=self.side_margin)  # (T,Hc,Wc)

        T = frames.shape[0]
        keyframes = self._keyframe_lookup.get(it.json_fname, set())

        # --- always return a window (variable-length near boundaries) ---
        k = self.context_window
        t = int(min(max(0, it.frame_idx), T - 1))

        start = max(0, t - k)
        end = min(T, t + k + 1)  # exclusive
        clip_np = frames[start:end]  # (P,H,W)
        P = clip_np.shape[0]

        # transform each frame
        clip_t: List[torch.Tensor] = [
            _apply_transform_frame(clip_np[p], self.transform) for p in range(P)
        ]  # each [C,H,W]
        x = torch.stack(clip_t, dim=0)  # [P,C,H,W]

        # keyframe mask for frames inside this returned window
        clip_frame_indices = range(start, end)
        is_kf = torch.tensor([fi in keyframes for fi in clip_frame_indices], dtype=torch.bool)  # [P]

        center_pos = t - start  # local center index inside returned clip

        return {
            "dicom": it.dicom_fname,
            "frame_idx": t,          # global center frame index
            "x": x,                  # [P,C,H,W]
            "is_keyframe": is_kf,    # [P] bool mask (window-local)
            "center_pos": center_pos # int
        }