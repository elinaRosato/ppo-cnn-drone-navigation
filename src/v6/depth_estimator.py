"""
Depth Anything V2 Small — monocular depth estimator.

Converts an RGB frame to a normalised single-channel depth map ready for
use as actor observations in the PPO-CNN drone navigation policy.

Output convention: higher values = closer obstacles (disparity).

Pipeline (per estimate() call):
    RGB (H, W, 3) uint8
        [training] → RGB augmentation (noise, brightness/contrast, motion blur, occlusions)
        → Depth Anything V2 Small  (HuggingFace transformers)
        → bilinear resize to target_size
        [training] → scale jitter ±scale_jitter_frac
        → per-frame min-max normalise → float32 [0, 1]
        [training] → CLAHE local contrast enhancement

New in v6:
    - Motion blur tied to action_magnitude (simulates lateral camera smear)
    - Random occlusion blobs (simulates dust / water droplets on lens)
    - CLAHE on depth output (improves local contrast in low-light / fog)

Installation:
    pip install transformers torch torchvision pillow
"""

import random

import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class DepthEstimator:
    """Thin wrapper around Depth Anything V2 Small for per-frame depth estimation."""

    def __init__(
        self,
        target_size: tuple = (192, 144),
        device: str = None,
        scale_jitter_frac: float = 0.05,
    ):
        """
        Parameters
        ----------
        target_size : (width, height) to resize the output depth map to.
            Must match the environment's img_width / img_height.
            Default: (192, 144) — IMX219 native 4:3 aspect ratio.
        device : 'cuda', 'cpu', or None (auto-detect).
        scale_jitter_frac : ± fraction applied to depth values during training
            to simulate monocular scale ambiguity. Set to 0 to disable.
        """
        self.target_size = target_size
        self.scale_jitter_frac = scale_jitter_frac

        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

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

    def estimate(self, rgb: np.ndarray, training: bool = False,
                 action_magnitude: float = 0.0) -> np.ndarray:
        """Estimate depth from a single RGB frame.

        Parameters
        ----------
        rgb : (H, W, 3) uint8, RGB channel order.
        training : when True, applies RGB augmentation, scale jitter and CLAHE.
        action_magnitude : |smoothed_action| in [0, 1] — used to scale motion
            blur during training to simulate lateral camera smear.

        Returns
        -------
        depth : (target_h, target_w) float32 in [0, 1].
                Higher values indicate closer / more dangerous obstacles.
        """
        if training:
            rgb = self._augment_rgb(rgb, action_magnitude=action_magnitude)

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

        # CLAHE: local contrast enhancement — helps in fog and low-light (training only)
        if training:
            gray = (depth * 255).astype(np.uint8)
            gray = self._clahe.apply(gray)
            depth = gray.astype(np.float32) / 255.0

        return depth.astype(np.float32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _augment_rgb(rgb: np.ndarray, action_magnitude: float = 0.0) -> np.ndarray:
        """Apply training augmentation to the RGB frame before depth estimation.

        Operations (in order):
          1. Gaussian noise  σ ~ U(0.5, 3.0)
          2. Contrast jitter α ~ U(0.85, 1.15)
          3. Brightness jitter β ~ U(-20, 20)
          4. 30%-chance GaussianBlur  k ∈ {3, 5}
          5. Horizontal motion blur tied to action_magnitude
             — kernel size = clip(action_magnitude × 4, 1, 5) pixels
             — simulates lateral camera smear from drone movement
          6. Random occlusion blobs (0–2 per frame)
             — dark patches simulating dust / water droplets on lens
             — 50% chance each blob is softened with Gaussian blur
        """
        img = rgb.astype(np.float32)

        # Noise + contrast + brightness
        sigma = np.random.uniform(0.5, 3.0)
        img += np.random.normal(0, sigma, img.shape).astype(np.float32)

        alpha = np.random.uniform(0.85, 1.15)
        beta  = np.random.uniform(-20.0, 20.0)
        img   = img * alpha + beta

        img = np.clip(img, 0, 255).astype(np.uint8)

        # Gaussian blur (random)
        if np.random.random() < 0.3:
            k = np.random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # Motion blur tied to lateral action magnitude
        blur_k = int(np.clip(action_magnitude * 4, 0, 4))
        if blur_k >= 2:
            # Horizontal motion blur kernel
            kernel = np.zeros((blur_k, blur_k), dtype=np.float32)
            kernel[blur_k // 2, :] = 1.0 / blur_k
            img = cv2.filter2D(img, -1, kernel)

        # Occlusion blobs (dust / water droplets on lens)
        h, w = img.shape[:2]
        n_blobs = random.randint(0, 2)
        for _ in range(n_blobs):
            bx      = random.randint(0, w - 1)
            by      = random.randint(0, h - 1)
            radius  = random.randint(2, 6)
            # darkness: 0.0 = fully black blob, 0.4 = dim overlay
            darkness = random.uniform(0.0, 0.4)
            blob_val = int((1.0 - darkness) * 255)
            overlay  = img.copy()
            cv2.circle(overlay, (bx, by), radius,
                       (blob_val, blob_val, blob_val), -1)
            # Blend so the edges aren't hard
            alpha_b = 0.6
            img = cv2.addWeighted(overlay, alpha_b, img, 1.0 - alpha_b, 0)
            # 50% chance: soften the blob (water droplet refraction)
            if random.random() < 0.5:
                img = cv2.GaussianBlur(img, (5, 5), 0)

        return img
