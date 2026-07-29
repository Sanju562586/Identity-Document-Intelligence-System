"""
stage4_forgery_detection/gradcam.py

Grad-CAM implementation for the DualStreamForgeryDetector.
Produces pixel-level saliency maps showing which regions contributed most
to the forgery prediction, overlaid on the original image.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ──────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM for a CNN layer.
    Works on the ELA CNN branch of the DualStreamForgeryDetector,
    targeting the last convolutional layer.

    Usage:
        cam = GradCAM(model, target_layer=model.ela_branch.features[-3])
        heatmap = cam(pixel_values, ela_maps, class_idx=1)  # 1=forged
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._gradients:  Optional[torch.Tensor] = None
        self._activations: Optional[torch.Tensor] = None
        self._hooks: list = []
        self._register_hooks()

    def _register_hooks(self):
        def _save_activation(_, __, output):
            self._activations = output.detach()

        def _save_gradient(_, grad_input, grad_output):
            self._gradients = grad_output[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(_save_activation)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_save_gradient)
        )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __call__(
        self,
        pixel_values: torch.Tensor,
        ela_maps: torch.Tensor,
        class_idx: int = 1,    # 1=forged
    ) -> np.ndarray:
        """
        Compute Grad-CAM heatmap.
        Returns: np.ndarray of shape (H, W) normalised to [0, 1].
        """
        self.model.eval()

        # Forward pass (with grad)
        pixel_values = pixel_values.detach().requires_grad_(False)
        ela_maps = ela_maps.detach().requires_grad_(True)

        logits = self.model(pixel_values, ela_maps)
        score  = logits[0, class_idx]

        self.model.zero_grad()
        score.backward()

        if self._gradients is None or self._activations is None:
            return np.zeros((224, 224), dtype=np.float32)

        # Pool gradients across spatial dims → (C,)
        grad = self._gradients[0]         # (C, H, W)
        act  = self._activations[0]       # (C, H, W)
        weights = grad.mean(dim=(1, 2))   # (C,)

        cam = (weights[:, None, None] * act).sum(0)  # (H, W)
        cam = F.relu(cam)

        # Normalise
        cam_np = cam.cpu().numpy()
        if cam_np.max() > 0:
            cam_np = (cam_np - cam_np.min()) / (cam_np.max() - cam_np.min() + 1e-8)
        return cam_np


# ──────────────────────────────────────────────────────────────────────────────
# Overlay helper
# ──────────────────────────────────────────────────────────────────────────────

def overlay_heatmap(
    original_img: Image.Image,
    cam_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> Image.Image:
    """
    Overlay a Grad-CAM heatmap on the original image.

    Args:
        original_img: PIL.Image (RGB)
        cam_map:      np.ndarray (H, W), values in [0, 1]
        alpha:        Blend factor for heatmap (0=original only, 1=heatmap only)
        colormap:     OpenCV colormap constant

    Returns:
        Blended PIL.Image (RGB)
    """
    orig_np = np.array(original_img.convert("RGB"))
    H, W    = orig_np.shape[:2]

    # Resize CAM to image size
    cam_resized = cv2.resize(cam_map, (W, H), interpolation=cv2.INTER_LINEAR)
    cam_uint8   = (cam_resized * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    blended = (alpha * heatmap_rgb + (1 - alpha) * orig_np).astype(np.uint8)
    return Image.fromarray(blended)


# ──────────────────────────────────────────────────────────────────────────────
# Batch visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualise_gradcam_batch(
    model: nn.Module,
    records: List[Dict],
    cfg: Dict,
    out_dir: str,
    n: int = 50,
    device: str = "cpu",
) -> None:
    """
    Generate and save Grad-CAM overlays for N forged samples.
    Saves to out_dir/gradcam/<image_stem>.jpg
    """
    from torchvision import transforms
    from stage4_forgery_detection.ela import compute_ela_gray

    out_dir = Path(out_dir)
    ensure_dir = lambda p: Path(p).mkdir(parents=True, exist_ok=True)
    ensure_dir(out_dir)

    s4 = cfg["stage4"]

    # Target last Conv layer of ELA branch
    target_layer = model.ela_branch.features[-3]   # last Conv2d
    cam = GradCAM(model, target_layer)

    rgb_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    forged_records = [r for r in records if r.get("is_forged", False)][:n]
    saved = 0

    for rec in forged_records:
        img_path = Path(".") / rec["image_path"]
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        pv  = rgb_transform(img).unsqueeze(0).to(device)
        ela = compute_ela_gray(img, quality=s4["ela_quality"], amplify=s4["ela_amplify"])
        ela_t = torch.from_numpy(
            cv2.resize(ela, (224, 224))
        ).float().unsqueeze(0).unsqueeze(0).to(device) / 255.0

        try:
            heatmap = cam(pv, ela_t, class_idx=1)
            vis = overlay_heatmap(img, heatmap, alpha=0.5)
        except Exception as e:
            continue

        stem = img_path.stem
        save_path = out_dir / f"{stem}_gradcam.jpg"
        vis.save(save_path)
        saved += 1

    cam.remove_hooks()
    print(f"[Grad-CAM] Saved {saved} visualisations to {out_dir}")
