import torch
import torch.nn as nn
import yaml
from models.USFM.models import build_vit

class TemporalConvHead(nn.Module):
    def __init__(self, d, hidden=256, k=3, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, hidden, kernel_size=k, padding=k//2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden, 1, kernel_size=1),
        )

    def forward(self, x):  # x: [B,P,D]
        x = x.transpose(1, 2)          # [B,D,P]
        logits = self.net(x).squeeze(1) # [B,P]
        return logits
    
class USFMClip(nn.Module):
    def __init__(self, model_config_path, clip_size: int, finetune_encoder: bool = False):
        super(USFMClip, self).__init__()
        self.load_feature_extractor(model_config_path=model_config_path)
        self.finetune_encoder = finetune_encoder
        self.conv_head = TemporalConvHead(d=768)
        self.clip_size = clip_size
    
    def load_feature_extractor(self, model_config_path: str):
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        model_config["model"]["model_cfg"]["num_classes"] = 0
        self.feature_extractor = build_vit(model_config["model"]["model_cfg"])

    def forward(self, clip: torch.Tensor):
        B, P, C, H, W = clip.shape
        clip = clip.view(B * P, C, H, W)

        features: torch.Tensor = self.feature_extractor(clip)

        features = features.view(B, P, *features.shape[1]) #reshape to Batches, clip_size, embed_dim (e.g. [4, 11, 768])
        logits = self.conv_head(features)

        return logits

    def train(self, mode: bool = True):
        super().train(mode)

        if not self.finetune_encoder:
            # Freeze feature_extrator
            for name, param in self.named_parameters():
                param.requires_grad = ("feature_extractor" not in name)

            return self