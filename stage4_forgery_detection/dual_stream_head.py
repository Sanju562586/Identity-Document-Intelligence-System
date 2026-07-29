"""
stage4_forgery_detection/dual_stream_head.py

Dual-stream forgery detection head:

  RGB stream:  frozen VLM vision encoder → [CLS] embedding (1152-dim)
  ELA stream:  lightweight 3-layer CNN → pooled embedding (256-dim)
  Fusion:      Concat [1408-dim] → MLP(512 → 128 → 2)

The VLM encoder is SigLIP (used by PaliGemma).
If PaliGemma is not available, falls back to timm ViT-B/16.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# ELA CNN Branch
# ──────────────────────────────────────────────────────────────────────────────

class ELACNNBranch(nn.Module):
    """
    Lightweight CNN that encodes a single-channel ELA map (H×W) into
    a fixed-size embedding vector.

    Architecture:
      Conv(1→32, 3×3) → BN → ReLU → MaxPool(2)
      Conv(32→64, 3×3) → BN → ReLU → MaxPool(2)
      Conv(64→128, 3×3) → BN → ReLU → AdaptiveAvgPool(4×4)
      Flatten → Linear(2048→256) → ReLU → Dropout(0.3)
    """

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(4),
        )
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * 4, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, H, W)
        feats = self.features(x)           # (B, 128, 4, 4)
        feats = feats.flatten(1)           # (B, 2048)
        return self.head(feats)            # (B, 256)


# ──────────────────────────────────────────────────────────────────────────────
# MLP Fusion Head
# ──────────────────────────────────────────────────────────────────────────────

class FusionMLP(nn.Module):
    """
    Fuses RGB and ELA embeddings and classifies into genuine / forged.

    in_dim  = rgb_dim + ela_dim
    hidden  = [512, 128]
    out_dim = 2 (logits)
    """

    def __init__(self, in_dim: int, hidden_dims=(512, 128), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ──────────────────────────────────────────────────────────────────────────────
# Vision Encoder Wrapper (frozen)
# ──────────────────────────────────────────────────────────────────────────────

class FrozenVisionEncoder(nn.Module):
    """
    Wraps the VLM's vision encoder (SigLIP / CLIP-style).
    Outputs the [CLS] / pooled embedding.
    Weights are always frozen.
    """

    def __init__(self, model_id: str = "google/paligemma-3b-pt-224"):
        super().__init__()
        self._model_id = model_id
        self._encoder  = None
        self._out_dim  = None

    def _load(self):
        if self._encoder is not None:
            return
        try:
            # Try loading the vision tower from AutoModel
            from transformers import AutoModel, AutoProcessor
            full_model = AutoModel.from_pretrained(
                self._model_id, trust_remote_code=True, torch_dtype=torch.float32
            )
            # PaliGemma: vision_tower attribute
            if hasattr(full_model, "vision_tower"):
                self._encoder = full_model.vision_tower
                self._out_dim = 1152    # SigLIP
            elif hasattr(full_model, "visual"):
                self._encoder = full_model.visual
                self._out_dim = 1024
            else:
                raise AttributeError("Cannot find vision encoder attribute.")
        except Exception:
            # Fallback: timm ViT-B/16
            import timm
            self._encoder = timm.create_model("vit_base_patch16_224", pretrained=True,
                                              num_classes=0)
            self._out_dim = 768

        for p in self._encoder.parameters():
            p.requires_grad_(False)
        self._encoder.eval()

    @property
    def out_dim(self) -> int:
        self._load()
        return self._out_dim

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self._load()
        with torch.no_grad():
            out = self._encoder(pixel_values)
        # Handle different output types
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0]   # CLS token
        return out


# ──────────────────────────────────────────────────────────────────────────────
# Full Dual-Stream Model
# ──────────────────────────────────────────────────────────────────────────────

class DualStreamForgeryDetector(nn.Module):
    """
    Complete forgery detector:
      Input:  RGB image (B,3,H,W) + ELA map (B,1,H,W)
      Output: logits (B,2)  — [genuine_logit, forged_logit]
    """

    def __init__(
        self,
        vision_encoder: Optional[nn.Module] = None,
        rgb_dim:   int = 1152,
        ela_dim:   int = 256,
        hidden_dims=(512, 128),
        dropout:   float = 0.3,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder   # pre-built FrozenVisionEncoder
        self.ela_branch     = ELACNNBranch(out_dim=ela_dim)
        self.fusion         = FusionMLP(
            in_dim=rgb_dim + ela_dim,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
        self._rgb_dim = rgb_dim

    def forward(
        self,
        pixel_values: torch.Tensor,
        ela_maps:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, H, W) — normalised RGB for vision encoder
            ela_maps:     (B, 1, H, W) — ELA single-channel maps
        Returns:
            logits: (B, 2)
        """
        # RGB features from frozen encoder
        if self.vision_encoder is not None:
            rgb_feats = self.vision_encoder(pixel_values)  # (B, rgb_dim)
        else:
            # Fallback: global average pool the raw image
            rgb_feats = pixel_values.mean(dim=(2, 3))
            # Project to expected dim
            rgb_feats = rgb_feats[:, :self._rgb_dim]

        # ELA features
        ela_feats = self.ela_branch(ela_maps)              # (B, ela_dim)

        # Fuse and classify
        combined = torch.cat([rgb_feats, ela_feats], dim=1)  # (B, rgb_dim+ela_dim)
        return self.fusion(combined)                          # (B, 2)

    def predict_proba(
        self,
        pixel_values: torch.Tensor,
        ela_maps:     torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probabilities: (B, 2) — [P(genuine), P(forged)]."""
        logits = self.forward(pixel_values, ela_maps)
        return F.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def build_detector(cfg: dict) -> DualStreamForgeryDetector:
    """Build the full detector from pipeline config."""
    s4 = cfg["stage4"]
    s3 = cfg["stage3"]

    use_fallback = s3.get("use_fallback", False)
    model_id = s3["model_id_fallback"] if use_fallback else s3["model_id"]

    encoder = FrozenVisionEncoder(model_id=model_id)
    rgb_dim  = encoder.out_dim

    detector = DualStreamForgeryDetector(
        vision_encoder=encoder,
        rgb_dim=rgb_dim,
        ela_dim=s4["ela_cnn_out_dim"],
        hidden_dims=tuple(s4["mlp_hidden_dims"]),
    )
    return detector
