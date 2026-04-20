"""
Depth Anything V2 Small — monocular depth estimator.

Converts an RGB frame to a normalised single-channel depth map ready for
use as actor observations in the PPO-CNN drone navigation policy.

Output convention: higher values = closer obstacles (disparity).

Pipeline (per estimate() call):
    RGB (H, W, 3) uint8
        [training] → RGB augmentation (noise, brightness/contrast, blur)
        → Depth Anything V2 Small  (HuggingFace transformers)
        → bilinear resize to target_size
        [training] → scale jitter ±scale_jitter_frac
        → per-frame min-max normalise → float32 [0, 1]

Installation:
    pip install transformers torch torchvision pillow

The model weights (~100 MB) are downloaded automatically on first use and
cached in ~/.cache/huggingface/hub/.
"""

import numpy as np
import cv2
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Thin wrapper around Depth Anything V2 Small for per-frame depth estimation."""

    def __init__(
        self,
        target_size: tuple = (192, 192),
        device: str = None,
        scale_jitter_frac: float = 0.05,
    ):
        """
        Parameters
        ----------
        target_size : (width, height) to resize the output depth map to.
            Must match the environment's img_width / img_height.
        device : 'cuda', 'cpu', or None (auto-detect).
        scale_jitter_frac : ± fraction applied to depth values during training
            to simulate monocular scale ambiguity. Set to 0 to disable.
        """
        self.target_size = target_size
        self.scale_jitter_frac = scale_jitter_frac

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"[DepthEstimator] Loading {MODEL_ID} on {device} ...")
        self._processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        self._model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID)
        self._model.to(device)
        self._model.eval()
        print("[DepthEstimator] Ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate(self, rgb: np.ndarray, training: bool = False) -> np.ndarray:
        """Estimate depth from a single RGB frame.

        Parameters
        ----------
        rgb : (H, W, 3) uint8, RGB channel order.
        training : when True, applies RGB augmentation and scale jitter.

        Returns
        -------
        depth : (target_h, target_w) float32 in [0, 1].
                Higher values indicate closer / more dangerous obstacles.
        """
        if training:
            rgb = self._augment_rgb(rgb)

        pil_img = PILImage.fromarray(rgb)
        inputs = self._processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model(**inputs)
            raw = outputs.predicted_depth  # (1, H, W) — disparity (closer = higher)

        depth = raw.squeeze().cpu().numpy()  # (H, W) float32

        # Resize to target resolution
        w, h = self.target_size
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        # Per-frame min-max normalisation → [0, 1]
        d_min, d_max = float(depth.min()), float(depth.max())
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        # Scale jitter: simulate monocular estimation uncertainty (training only)
        if training and self.scale_jitter_frac > 0.0:
            scale = np.random.uniform(
                1.0 - self.scale_jitter_frac,
                1.0 + self.scale_jitter_frac,
            )
            depth = np.clip(depth * scale, 0.0, 1.0)

        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_rgb(rgb: np.ndarray) -> np.ndarray:
        """Apply training augmentation to the RGB frame before depth estimation.

        Identical to the augmentation pipeline used in v2 for grayscale frames,
        but applied to RGB so the perturbations propagate naturally through the
        depth estimator rather than corrupting depth values directly.

        Operations (in order):
          1. Gaussian noise  σ ~ U(0.5, 3.0)
          2. Contrast jitter α ~ U(0.85, 1.15)
          3. Brightness jitter β ~ U(-20, 20)
          4. 30%-chance GaussianBlur  k ∈ {3, 5}
        """
        img = rgb.astype(np.float32)

        sigma = np.random.uniform(0.5, 3.0)
        img += np.random.normal(0, sigma, img.shape).astype(np.float32)

        alpha = np.random.uniform(0.85, 1.15)
        beta  = np.random.uniform(-20.0, 20.0)
        img   = img * alpha + beta

        img = np.clip(img, 0, 255).astype(np.uint8)

        if np.random.random() < 0.3:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        return img
