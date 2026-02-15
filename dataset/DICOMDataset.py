from torch.utils.data import Dataset
import pydicom
import os
import json
import numpy as np
from typing import Optional, Tuple, Dict, List, Any
import cv2
from collections import OrderedDict
import random
from PIL import Image
import torch

# ---------- your utilities unchanged ----------
def to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    mn = float(np.min(x))
    mx = float(np.max(x))
    if mx <= mn + 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    out = (x - mn) * (255.0 / (mx - mn))
    return np.clip(out, 0, 255).astype(np.uint8)

def ensure_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        g = to_uint8(frame)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[-1] == 3:
        if frame.dtype != np.uint8:
            g = to_uint8(cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2GRAY))
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return frame
    raise ValueError(f"Unexpected frame shape: {frame.shape}")

def get_bmode_region(ds) -> Optional[Tuple[int, int, int, int]]:
    regions = ds.get((0x0018, 0x6011), None)
    if regions is None:
        return None

    best = None
    best_score = -1.0
    for r in regions:
        try:
            x0 = int(getattr(r, "RegionLocationMinX0"))
            y0 = int(getattr(r, "RegionLocationMinY0"))
            x1 = int(getattr(r, "RegionLocationMaxX1"))
            y1 = int(getattr(r, "RegionLocationMaxY1"))
        except Exception:
            continue

        w = max(0, x1 - x0)
        h = max(0, y1 - y0)
        area = float(w * h)

        dx = float(getattr(r, "PhysicalDeltaX", 0.0) or 0.0)
        dy = float(getattr(r, "PhysicalDeltaY", 0.0) or 0.0)

        score = area + (1e12 if (dx > 0 and dy > 0) else 0.0)
        if score > best_score:
            best_score = score
            best = (x0, y0, x1, y1)
    return best

def crop_ultrasound_frames(ds, full: np.ndarray) -> np.ndarray:
    region = get_bmode_region(ds)
    if region is None:
        return full

    x0, y0, x1, y1 = region

    if full.ndim == 2:
        H, W = full.shape
        x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
        # note: assumes x0+125 < x1-125; we guard later
        return full[y0:y1, x0 + 125:x1 - 125]
    elif full.ndim == 3:
        T, H, W = full.shape
        x0 = max(0, min(x0, W)); x1 = max(0, min(x1, W))
        y0 = max(0, min(y0, H)); y1 = max(0, min(y1, H))
        return full[:, y0:y1, x0 + 125:x1 - 125]
    else:
        raise ValueError(f"Unexpected pixel_array shape: {full.shape}")

def np_gray_to_pil(img_u8: np.ndarray) -> Image.Image:
    # img_u8: (H,W) uint8
    if img_u8.dtype != np.uint8:
        img_u8 = img_u8.astype(np.uint8)
    return Image.fromarray(img_u8, mode="L").convert("RGB")

# ---------- dataset ----------
class DICOMDataset(Dataset):
    """
    Returns a context clip of frames centered at a chosen frame index.
    Target is 1 if the *center* frame is a keyframe, else 0.
    (You also get 'keyframe_mask' indicating which frames in the returned context are keyframes.)

    To balance negatives, set zero_prob > 0 to sometimes sample a random non-keyframe center frame.
    """
    def __init__(
        self,
        dicom_dir_path: str,
        json_dir_path: str,
        transform=None,
        context_window: int = 5,
        jsons_to_include: Optional[List[str]] = None,
        zero_prob: float = 0.3,
        pad_mode: str = "edge",  # "edge" | "wrap" | "zero"
        cache_size: int = 8,
        force_multiframe_axis0: bool = True,
    ):
        self.dicom_dir_path = dicom_dir_path
        self.json_dir_path = json_dir_path
        self.jsons_to_include = set(jsons_to_include) if jsons_to_include else None
        self.context_window = int(context_window)
        self.transform = transform
        self.zero_prob = float(zero_prob)
        self.pad_mode = pad_mode
        self.force_multiframe_axis0 = force_multiframe_axis0

        # summary: list of dicts (easier than dict-of-dicts for indexing)
        self.summary = self.get_ds_summary_list()

        # Build a flat index: each element corresponds to a (dicom_idx, frame_idx)
        self.index = self._build_index()

        # LRU cache for decoded/cropped arrays per dicom
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.cache_size = int(cache_size)

        # Precompute per-dicom negatives list for faster sampling
        self._neg_frames: List[List[int]] = []
        for s in self.summary:
            n = int(s["num_frames"])
            kf = set(s["keyframes"])
            neg = [i for i in range(n) if i not in kf]
            self._neg_frames.append(neg)

    # (a) correct len
    def __len__(self) -> int:
        return len(self.index)

    def get_ds_summary_list(self) -> List[Dict[str, Any]]:
        """
        Produces a list, one entry per JSON file (and corresponding DICOM).
        Each entry: dicom filename, num_frames, keyframes (sorted list)
        """
        out = []
        for json_fname in sorted(os.listdir(self.json_dir_path)):
            if not json_fname.endswith(".json"):
                continue
            if self.jsons_to_include and json_fname not in self.jsons_to_include:
                continue

            json_fpath = os.path.join(self.json_dir_path, json_fname)
            with open(json_fpath, "r") as f:
                obj = json.load(f)

            dicom_name = obj["item"]["name"]
            # WARNING: your earlier JSON had frame_count=1 even when cine exists.
            # Here we trust it, but you may want to override with ds.NumberOfFrames when loading.
            num_frames = int(obj["item"]["slots"][0].get("frame_count", 1))

            keyframes = set()
            for ann in obj.get("annotations", []):
                name = ann.get("name", "")
                if name == "Brest Frame":
                    continue
                frames = ann.get("frames", {})
                for k, v in frames.items():
                    try:
                        frame_idx = int(k)
                        if isinstance(v, dict) and v.get("keyframe", False):
                            keyframes.add(frame_idx)
                    except Exception:
                        pass

            out.append({
                "json": json_fname,
                "dicom": dicom_name,
                "num_frames": num_frames,
                "keyframes": sorted(list(keyframes)),
            })
        return out

    def _build_index(self) -> List[Tuple[int, int]]:
        """
        Flat list of (dicom_idx, frame_idx) for all frames in all dicoms.
        This gives you many training samples even with only 43 dicoms.
        """
        idx = []
        for di, s in enumerate(self.summary):
            n = int(s["num_frames"])
            for fi in range(n):
                idx.append((di, fi))
        return idx

    # --------- IO + cache ----------
    def _load_cropped_frames(self, dicom_name: str) -> np.ndarray:
        """
        Returns frames as:
          - (H,W) if single frame
          - (T,H,W) if multi-frame
        Cached by dicom filename.
        """
        if dicom_name in self._cache:
            arr = self._cache.pop(dicom_name)
            self._cache[dicom_name] = arr
            return arr

        dcm_path = os.path.join(self.dicom_dir_path, dicom_name)
        ds = pydicom.dcmread(dcm_path)

        full = ds.pixel_array  # could be (H,W) or (T,H,W) or other vendor layouts

        # Optional: override num_frames using DICOM header if available
        # Some datasets have bad JSON frame_count.
        # NumberOfFrames is a string in some DICOMs.
        # We'll handle axis normalization below.

        # Normalize to (T,H,W) when multi-frame if needed
        if full.ndim == 3 and self.force_multiframe_axis0:
            # common cases: (T,H,W) OR (H,W,T)
            # heuristic: if last dim is "smallish" like <= 400 and first dim looks like H (>= 400),
            # it might be (H,W,T). For your data T ~ 75.
            if full.shape[0] > 256 and full.shape[-1] <= 400:
                # (H,W,T) -> (T,H,W)
                full = np.transpose(full, (2, 0, 1))

        cropped = crop_ultrasound_frames(ds, full)

        # Guard against bad crop (e.g., x0+125 >= x1-125)
        if cropped.ndim == 2 and (cropped.shape[0] == 0 or cropped.shape[1] == 0):
            cropped = full if full.ndim == 2 else full[0]
        if cropped.ndim == 3 and (cropped.shape[1] == 0 or cropped.shape[2] == 0):
            cropped = full

        # cache insert
        self._cache[dicom_name] = cropped
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return cropped

    # --------- context handling ----------
    def _get_context_indices(self, center: int, n_frames: int) -> List[int]:
        """
        Returns indices [center-w ... center+w] with padding when out-of-range.
        pad_mode:
          - "edge": clamp to [0, n-1]
          - "wrap": modulo
          - "zero": mark out-of-range as -1 (we'll create zero frames)
        """
        w = self.context_window
        out = []
        for t in range(center - w, center + w + 1):
            if 0 <= t < n_frames:
                out.append(t)
            else:
                if self.pad_mode == "edge":
                    out.append(min(max(t, 0), n_frames - 1))
                elif self.pad_mode == "wrap":
                    out.append(t % n_frames)
                elif self.pad_mode == "zero":
                    out.append(-1)
                else:
                    raise ValueError(f"Unknown pad_mode: {self.pad_mode}")
        return out

    def _extract_clip(self, frames: np.ndarray, idxs: List[int]) -> np.ndarray:
        """
        frames: (H,W) or (T,H,W)
        returns clip: (L,H,W) where L = 2*w+1
        """
        if frames.ndim == 2:
            # single frame dicom: replicate or zero-pad
            H, W = frames.shape
            clip = []
            for ii in idxs:
                if ii == -1 and self.pad_mode == "zero":
                    clip.append(np.zeros((H, W), dtype=frames.dtype))
                else:
                    clip.append(frames.copy())
            return np.stack(clip, axis=0)

        # multi-frame: frames is (T,H,W)
        T, H, W = frames.shape
        clip = []
        for ii in idxs:
            if ii == -1 and self.pad_mode == "zero":
                clip.append(np.zeros((H, W), dtype=frames.dtype))
            else:
                clip.append(frames[ii])
        return np.stack(clip, axis=0)

    # (b) __getitem__ with targets
    # (c) zero_prob for 0-keyframe samples
    def __getitem__(self, idx: int):
        dicom_idx, frame_idx = self.index[idx]
        s = self.summary[dicom_idx]
        dicom_name = s["dicom"]

        frames = self._load_cropped_frames(dicom_name)

        # Determine actual number of frames from loaded data (more reliable than JSON)
        if frames.ndim == 2:
            n_frames = 1
        else:
            n_frames = frames.shape[0]

        keyframes = set([k for k in s["keyframes"] if 0 <= k < n_frames])

        # With probability zero_prob, force a negative center frame (if possible)
        if self.zero_prob > 0 and random.random() < self.zero_prob:
            neg_list = self._neg_frames[dicom_idx]
            # rebuild neg_list for true n_frames if JSON was wrong
            if len(neg_list) != n_frames:
                neg_list = [i for i in range(n_frames) if i not in keyframes]
            if len(neg_list) > 0:
                frame_idx = random.choice(neg_list)
            # if no negatives exist (rare), just keep original

        # Build context indices and clip
        ctx_idxs = self._get_context_indices(frame_idx, n_frames)
        clip = self._extract_clip(frames, ctx_idxs)  # (L,H,W)

        # target: 1 if CENTER frame is keyframe else 0
        target = 1 if frame_idx in keyframes else 0

        # optional: a mask of keyframes inside the context window (useful later)
        keyframe_mask = np.array([(i in keyframes) if i >= 0 else False for i in ctx_idxs], dtype=np.uint8)

        # Convert to uint8 (or keep float) BEFORE transforms depending on your pipeline.
        # Many torchvision transforms expect PIL/uint8; adjust as needed.
        # Here I keep grayscale uint8.
        clip_u8 = np.stack([to_uint8(f) for f in clip], axis=0)  # (L,H,W), uint8

        if self.transform is not None:
            # You decide transform semantics. Common patterns:
            # 1) apply same transform to each frame
            # 2) apply transform to a stacked tensor (L,C,H,W)
            transformed = []
            for i in range(clip_u8.shape[0]):
                img = clip_u8[i]
                img = np_gray_to_pil(img)
                print(np.array(img).shape)
                out = self.transform(img)  # user-defined
                print("out shape:", out.shape)
                transformed.append(out)
            clip_out = torch.Tensor(transformed)
        else:
            clip_out = clip_u8

        return {
            "clip": clip_out,                 # (L,H,W) uint8 OR list[tensor] if transform used
            "target": int(target),            # center frame label
            "keyframe_mask": keyframe_mask,   # which positions in the clip are keyframes
            "dicom": dicom_name,
            "center_idx": int(frame_idx),
            "ctx_indices": ctx_idxs,
        }
