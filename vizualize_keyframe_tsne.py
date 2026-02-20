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
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def plot_tsne(features, labels, graph_dir, *, pca_dim=50, perplexity=30, learning_rate="auto",
              n_iter=1500, random_state=42, title="t-SNE (2D)"):
    """
    features: list of np.ndarray, each (768,)
    labels: list/np.ndarray of 0/1 (same length as features)
    """

    class_idx_to_name = {0: "normal", 1: "keyframe"}

    # Stack into (N, 768)
    X = np.stack([np.asarray(f).reshape(-1) for f in features], axis=0).astype(np.float32)
    y = np.asarray(labels).astype(int)

    if X.ndim != 2 or X.shape[1] != 768:
        raise ValueError(f"Expected X shape (N, 768), got {X.shape}")
    if len(y) != X.shape[0]:
        raise ValueError(f"features length ({X.shape[0]}) != labels length ({len(y)})")
    if not np.all(np.isin(y, [0, 1])):
        raise ValueError("labels must be only 0/1")

    n = X.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 samples for t-SNE.")
    # t-SNE rule of thumb: perplexity < (N-1)/3
    max_perp = max(2, (n - 1) // 3)
    perp = min(perplexity, max_perp)

    # Optional PCA pre-reduction
    X_in = X
    if pca_dim is not None and pca_dim < X.shape[1]:
        pca_dim = min(pca_dim, X.shape[0] - 1)  # PCA components must be < N
        X_in = PCA(n_components=pca_dim, random_state=random_state).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="pca",
        random_state=random_state,
        metric="euclidean",
    )
    Z = tsne.fit_transform(X_in)

    plt.figure(figsize=(7, 6))
    for cls in [0, 1]:
        idx = (y == cls)
        plt.scatter(Z[idx, 0], Z[idx, 1], s=14, alpha=0.8, label=f"{class_idx_to_name[cls]}")
    plt.title(f"{title} | N={n}, perplexity={perp}, PCA={pca_dim}")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(graph_dir, "tsne_kf_vs_non.png"))

    plt.show()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dicom_dir", type=str, required=True)    
    parser.add_argument("--json_dir", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--graph_dir", type=str, required=True)
    return parser.parse_args()

def get_image_transforms(args):
    if args.model == "usfm":
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(224, 224)),
            transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        ])
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

def main():
    args = get_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    num_classes = 8

    model = get_model(args, num_classes=num_classes).to(device=device)
    model.load_state_dict(torch.load(os.path.join("logs", "segmentation", "best.pt"), map_location=device))

    img_transforms = get_image_transforms(args=args)

    ds = DICOMKeyframeDataset(
        dicom_dir_path=args.dicom_dir,
        json_dir_path=args.json_dir,   # can be None now (dataset must handle)
        context_window=0,
        transform=img_transforms,
        return_keyframes=False
    )


    model.eval()
    labels = []
    all_features = []

    for obj in tqdm(ds, desc="Computing frame scores"):        
        img = obj["x"] # context_window=0 should give [1,C,H,W]
        
        is_kf = bool(obj["is_keyframe"][0]) if torch.is_tensor(obj["is_keyframe"]) else bool(obj["is_keyframe"])
        label = 1 if is_kf else 0

        img = img.to(device=device)

        features = model.backbone(model.data_preprocessor(img)) #[1, d]
        features = torch.mean(features[2], dim=[2, 3])

        all_features.append(features[0].detach().cpu().numpy())
        labels.append(label)


    if not os.path.exists(args.graph_dir):
        os.makedirs(args.graph_dir)

    plot_tsne(features=all_features, labels=labels, graph_dir=args.graph_dir)


if __name__ == "__main__":
    main()