from torch.utils.data import Dataset
import pydicom, os, json
import numpy as np
import torch
import cv2
from typing import Dict, List, Tuple, Any, Optional
from collections import OrderedDict
from dataset.DICOMDataset import crop_ultrasound_frames, to_uint8, np_gray_to_pil
from PIL import Image

def _maybe_scale_xy(x: float, y: float, W: int, H: int) -> Tuple[int, int]:
    # Same as your helper
    if (0.0 <= x <= 1.5) and (0.0 <= y <= 1.5):
        px = int(round(x * (W - 1)))
        py = int(round(y * (H - 1)))
    else:
        px = int(round(x))
        py = int(round(y))
    px = max(0, min(W - 1, px))
    py = max(0, min(H - 1, py))
    return px, py


def polygon_to_mask(mask: np.ndarray, polygon_obj: dict, class_id: int):
    # polygon_obj: {"paths": [[{"x":..,"y":..}, ...]]}
    H, W = mask.shape
    paths = polygon_obj.get("paths", [])
    if not paths:
        return
    pts = paths[0]
    if len(pts) < 3:
        return
    pts_xy = np.array([_maybe_scale_xy(p["x"], p["y"], W, H) for p in pts], dtype=np.int32)
    cv2.fillPoly(mask, [pts_xy], int(class_id))


def ellipse_to_mask(mask: np.ndarray, ellipse_obj: dict, class_id: int):
    H, W = mask.shape
    c = ellipse_obj["center"]
    r = ellipse_obj["radius"]
    cx, cy = _maybe_scale_xy(c["x"], c["y"], W, H)

    # radii might be normalized; reuse _maybe_scale_xy trick
    rx, _ = _maybe_scale_xy(r["x"], 0.0, W, H)
    _, ry = _maybe_scale_xy(0.0, r["y"], W, H)
    rx, ry = max(1, rx), max(1, ry)

    # Darwin JSON angle appears in radians in your earlier snippet.
    # OpenCV ellipse expects degrees.
    angle_rad = float(ellipse_obj.get("angle", 0.0))
    angle_deg = float(np.degrees(angle_rad))

    cv2.ellipse(
        mask,
        center=(cx, cy),
        axes=(rx, ry),
        angle=angle_deg,
        startAngle=0,
        endAngle=360,
        color=int(class_id),
        thickness=-1
    )


def line_to_mask(mask: np.ndarray, line_obj: dict, class_id: int, thickness: int = 3):
    H, W = mask.shape
    path = line_obj.get("path", [])
    if len(path) < 2:
        return
    pts_xy = np.array([_maybe_scale_xy(p["x"], p["y"], W, H) for p in path], dtype=np.int32)
    cv2.polylines(mask, [pts_xy], isClosed=False, color=int(class_id), thickness=int(thickness))


class DICOMSegmentationDataset(Dataset):
    """
    Returns only annotated frames.
    Output:
      - image: torch.FloatTensor [3,H,W] (or whatever your transform returns)
      - mask:  torch.LongTensor  [H,W] with values in {0..K}
      - meta:  dicom name, frame index
    """

    def __init__(
        self,
        dicom_dir_path: str,
        json_dir_path: str,
        metadata_file: str,
        transform=None,            
        mask_transform=None,
        jsons_to_include: Optional[List[str]] = None,
        ignore_classes: Optional[List[str]] = None,
        class_priority: Optional[List[str]] = None,  # higher later overwrite earlier
        line_thickness: int = 3,
        cache_size: int = 8,
    ):
        self.dicom_dir_path = dicom_dir_path
        self.json_dir_path = json_dir_path
        self.transform = transform
        self.mask_transform = mask_transform
        self.jsons_to_include = set(jsons_to_include) if jsons_to_include else None
        self.ignore_classes = set(ignore_classes) if ignore_classes else set()
        self.class_priority = class_priority
        self.line_thickness = int(line_thickness)

        self.class_type = self._load_metadata_classes(metadata_file)  # name -> type
        self.class_id = self._build_class_id_map()                    # name -> int (>=1)
        self.id_to_name = {v: k for k, v in self.class_id.items()}

        # index entries: (dicom_name, json_path, frame_idx)
        self.index = self._build_index()

        # LRU cache: dicom_name -> cropped pixel array (T,H,W) or (H,W)
        self._cache: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self.cache_size = int(cache_size)

    def get_num_classes(self, include_background: bool = True):
        return len(self.class_id.keys()) + 1
    
    def get_class_name_to_id(self):
        return self.class_id
    
    def _norm_name(self, name: str) -> str:
        return (name or "").strip().lower()

    def _is_line_class(self, class_name: str) -> bool:
        # metadata gives type like "line", "polygon", "ellipse"
        t = self.class_type.get(class_name, None)
        return (t == "line")

    def _load_metadata_classes(self, file_path: str) -> Dict[str, str]:
        with open(file_path, "r") as f:
            metadata = json.load(f)
        classes_info = metadata.get("classes", [])
        out: Dict[str, str] = {}
        for m in classes_info:
            out[m["name"]] = m["type"]  # "ellipse" | "polygon" | "line" | ...
        return out

    def _build_class_id_map(self) -> Dict[str, int]:
        """
        Build class mapping:
        - background = 0
        - ignore line classes
        - case-insensitive dedupe; keep first encountered canonical name
        """
        seen = set()
        kept_names = []

        # iterate in a stable order so "first encountered" is reproducible
        for name in self.class_type.keys():
            n = self._norm_name(name)

            if not n or n == "brest frame":
                continue
            if name in self.ignore_classes or n in {self._norm_name(x) for x in self.ignore_classes}:
                continue
            if self.class_type.get(name) == "line":
                continue
            if n in seen:
                continue  # drop duplicate (RETINA vs retina etc.)
            seen.add(n)
            kept_names.append(name)  # keep original first-seen spelling as canonical

        # assign ids
        return {name: i + 1 for i, name in enumerate(kept_names)}

    def _build_index(self) -> List[Tuple[str, str, int]]:
        idx = []
        for json_fname in sorted(os.listdir(self.json_dir_path)):
            if not json_fname.endswith(".json"):
                continue
            if self.jsons_to_include and json_fname not in self.jsons_to_include:
                continue

            json_path = os.path.join(self.json_dir_path, json_fname)
            with open(json_path, "r") as f:
                obj = json.load(f)

            dicom_name = obj["item"]["name"]
            per_frame_has_ann = set()

            # dedupe within this file (case-insensitive), keep first class only
            seen_classes = set()

            for ann in obj.get("annotations", []):
                raw_name = ann.get("name", "")
                n = self._norm_name(raw_name)
                if not n or n == "brest frame":
                    continue
                if n in seen_classes:
                    continue
                seen_classes.add(n)

                # find canonical name in self.class_id (stored using first-seen spelling)
                # easiest: accept by normalized match
                # (we'll render using the same normalization logic)
                ann_type = self.class_type.get(raw_name)
                if ann_type not in ("ellipse", "polygon"):
                    continue  # excludes "line" and everything else

                frames = ann.get("frames", {})
                for k, v in frames.items():
                    try:
                        fi = int(k)
                    except Exception:
                        continue
                    if ann_type == "polygon" and "polygon" in v:
                        per_frame_has_ann.add(fi)
                    elif ann_type == "ellipse" and "ellipse" in v:
                        per_frame_has_ann.add(fi)

            for fi in sorted(per_frame_has_ann):
                idx.append((dicom_name, json_path, fi))

        return idx

    def __len__(self) -> int:
        return len(self.index)

    def _load_cropped_pixel(self, dicom_name: str) -> np.ndarray:
        if dicom_name in self._cache:
            arr = self._cache.pop(dicom_name)
            self._cache[dicom_name] = arr
            return arr

        dcm_path = os.path.join(self.dicom_dir_path, dicom_name)
        ds = pydicom.dcmread(dcm_path)
        pix = ds.pixel_array

        # normalize cine shape
        if pix.ndim == 2:
            pix = pix[None, ...]  # (1,H,W)
        elif pix.ndim == 3:
            pass
        else:
            raise ValueError(f"Unexpected pixel_array shape: {pix.shape}")

        # crop ultrasound region if you have that helper
        try:
            pix = crop_ultrasound_frames(ds, pix)  # (T,H,W)
            if pix.ndim == 2:
                pix = pix[None, ...]
        except Exception:
            # fallback: no crop
            pass

        self._cache[dicom_name] = pix
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return pix

    def _render_mask(self, obj: dict, frame_idx: int, H: int, W: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)

        seen_classes = set()
        for ann in obj.get("annotations", []):
            raw_name = ann.get("name", "")
            n = self._norm_name(raw_name)

            if not n or n == "brest frame":
                continue
            if n in seen_classes:
                continue  # keep first occurrence only
            seen_classes.add(n)

            # skip line classes
            ann_type = self.class_type.get(raw_name)
            if ann_type not in ("ellipse", "polygon"):
                continue

            # find a canonical class name that exists in self.class_id
            # (match by normalized name)
            canonical = None
            for cname in self.class_id.keys():
                if self._norm_name(cname) == n:
                    canonical = cname
                    break
            if canonical is None:
                continue
            cid = self.class_id[canonical]

            fr = ann.get("frames", {}).get(str(frame_idx))
            if not isinstance(fr, dict):
                continue

            if ann_type == "polygon" and "polygon" in fr:
                polygon_to_mask(mask, fr["polygon"], cid)
            elif ann_type == "ellipse" and "ellipse" in fr:
                ellipse_to_mask(mask, fr["ellipse"], cid)

        return mask

    def __getitem__(self, idx: int):
        dicom_name, json_path, frame_idx = self.index[idx]

        # load json
        with open(json_path, "r") as f:
            obj = json.load(f)

        pix = self._load_cropped_pixel(dicom_name)  # (T,H,W)
        T, H, W = pix.shape
        frame_idx = max(0, min(int(frame_idx), T - 1))

        img_u8 = to_uint8(pix[frame_idx])  # (H,W)
        mask = self._render_mask(obj, frame_idx, H=img_u8.shape[0], W=img_u8.shape[1])  # (H,W)

        # Convert grayscale -> RGB for pretrained encoders
        

        if self.transform is not None:
            img_pil = np_gray_to_pil(img_u8=img_u8)
            img_t = self.transform(img_pil)
        else:
            img_rgb = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2RGB)  # (H,W,3)
            img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # [3,H,W]

        if self.mask_transform is not None:
            mask_pil = Image.fromarray(np.array(mask.astype("uint8")), mode="L")
            mask_t = self.mask_transform(mask_pil)
        else:
            mask_t = torch.from_numpy(mask).long()                               # [H,W]


        return {
            "image": img_t,
            "mask": mask_t,
            "dicom": dicom_name,
            "frame_idx": frame_idx,
        }
