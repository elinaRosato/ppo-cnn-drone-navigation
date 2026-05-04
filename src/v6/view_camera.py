"""
Live debug viewer — v6 (8-panel tiled display, 4:3 IMX219).

Shows everything in a single 4×2 grid window:

  Row 0 │ RGB (raw)  │ RGB (augmented) │ Depth est. + CLAHE │ AirSim 3.5m (penalty) │
  Row 1 │ Stack t-2  │ Stack t-1       │ Stack t-0 (latest) │ AirSim 15m  (debug)   │

Image resolution: 192×144 (4:3, FOV 62°).  Tiles displayed at 256×192.

Panel 2 shows the RGB frame after the full training augmentation pipeline:
  noise → contrast/brightness → random blur → motion blur → occlusion blobs.
Panel 3 shows depth estimated from that augmented frame, with CLAHE applied —
exactly what the actor receives during training.

GT AirSim depths (col 4) use COLORMAP_HOT: black=far, white=close.

Usage:
    python3 view_camera.py
    python3 view_camera.py --fps 10
    python3 view_camera.py --topic /airsim_node/SimpleFlight/front_center_Scene/image

Press Q to quit.
"""

import argparse
import os
import sys
import threading

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator

STACK_FRAMES   = 3
IMG_W          = 192    # must match avoidance_env.py / settings.json  (IMX219 4:3, FOV 62°)
IMG_H          = 144
TILE_W         = 256    # display tile size (4:3 at 1.33× source)
TILE_H         = 192
PROX_THRESHOLD = 3.5    # metres — must match avoidance_env.py
DEBUG_RANGE    = 15.0   # metres — shown in panel 8 for visual reference
PROX_WEIGHT    = 0.5


# ── ROS2 receiver ─────────────────────────────────────────────────────────────

class CameraReceiver(Node):
    def __init__(self, scene_topic: str, depth_topic: str):
        super().__init__('view_camera_v4')
        self._bridge       = CvBridge()
        self._lock         = threading.Lock()
        self._latest_rgb   = None   # (H, W, 3) uint8 RGB
        self._latest_depth = None   # (H, W) float32 metres (GT)

        self.create_subscription(Image, scene_topic, self._scene_cb, 10)
        self.create_subscription(Image, depth_topic, self._depth_cb, 10)
        self.get_logger().info(f"Subscribed to {scene_topic}")
        self.get_logger().info(f"Subscribed to {depth_topic}")

    def _scene_cb(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            rgb = cv2.resize(cv_img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest_rgb = rgb
        except Exception as exc:
            self.get_logger().warning(f"Scene callback error: {exc}")

    def _depth_cb(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32)
            depth = cv2.resize(depth, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
            with self._lock:
                self._latest_depth = depth
        except Exception as exc:
            self.get_logger().warning(f"Depth callback error: {exc}")

    def get_latest(self):
        with self._lock:
            rgb   = self._latest_rgb.copy()   if self._latest_rgb   is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
        return rgb, depth


# ── Tile helpers ──────────────────────────────────────────────────────────────

def _label(tile: np.ndarray, text: str, color=(220, 220, 220)) -> np.ndarray:
    cv2.putText(tile, text, (4, TILE_H - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return tile


def rgb_tile(rgb: np.ndarray, label: str) -> np.ndarray:
    """RGB → BGR tile for cv2 display."""
    bgr = cv2.cvtColor(
        cv2.resize(rgb, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA),
        cv2.COLOR_RGB2BGR,
    )
    return _label(bgr, label)


def gray_tile(depth_01: np.ndarray, label: str) -> np.ndarray:
    """Float32 [0,1] depth → true grayscale BGR tile — exactly what the model sees."""
    uint8 = (np.clip(depth_01, 0.0, 1.0) * 255).astype(np.uint8)
    gray  = cv2.resize(uint8, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST)
    bgr   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _label(bgr, label)


def gt_depth_tile(gt: np.ndarray, clip_m: float, label: str,
                  show_roi: bool = False) -> np.ndarray:
    """Metric GT depth → COLORMAP_HOT tile.  Closer = brighter."""
    clipped  = np.clip(gt, 0.0, clip_m)
    uint8    = ((clip_m - clipped) / clip_m * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(
        cv2.resize(uint8, (TILE_W, TILE_H), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_HOT,
    )

    if show_roi:
        # Draw centre-50% rectangle used for penalty computation
        sx = TILE_W / IMG_W
        sy = TILE_H / IMG_H
        qw = IMG_W // 4
        qh = IMG_H // 4
        cv2.rectangle(
            coloured,
            (int(qw * sx), int(qh * sy)),
            (int(3 * qw * sx), int(3 * qh * sy)),
            (255, 255, 255), 1,
        )
        center = clipped[qh: 3 * qh, qw: 3 * qw]
        min_d  = float(np.min(center))
        if min_d < clip_m:
            penalty = (clip_m - min_d) / clip_m * PROX_WEIGHT
            text    = f"{min_d:.2f}m  -{penalty:.3f}"
            color   = (80, 160, 255)
        else:
            text  = f">{clip_m:.0f}m  CLEAR"
            color = (80, 220, 80)
        _label(coloured, text, color)
    else:
        qw    = IMG_W // 4
        qh    = IMG_H // 4
        min_d = float(np.min(clipped[qh: 3 * qh, qw: 3 * qw]))
        _label(coloured, f"{label}  min={min_d:.1f}m")

    return coloured


def blank_tile(text: str = "waiting...") -> np.ndarray:
    tile = np.zeros((TILE_H, TILE_W, 3), dtype=np.uint8)
    cv2.putText(tile, text, (10, TILE_H // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    return tile


def build_grid(row0: list, row1: list) -> np.ndarray:
    """Concatenate two rows of TILE_W×TILE_H BGR tiles into one display image."""
    top    = np.concatenate(row0, axis=1)
    bottom = np.concatenate(row1, axis=1)
    sep    = np.full((2, top.shape[1], 3), 60, dtype=np.uint8)
    return np.concatenate([top, sep, bottom], axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(fps: int, scene_topic: str, depth_topic: str):
    rclpy.init()
    node = CameraReceiver(scene_topic, depth_topic)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Loading Depth Anything V2 Small...")
    estimator = DepthEstimator(target_size=(IMG_W, IMG_H))
    print("Estimator ready.")

    wait_ms     = max(1, int(1000 / fps))
    frame_stack = np.zeros((STACK_FRAMES, IMG_H, IMG_W), dtype=np.float32)

    win = "v6 debug view  [Q = quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TILE_W * 4, TILE_H * 2 + 2)

    print(f"Waiting for frames on {scene_topic} ... Press Q to quit.")

    while True:
        rgb, gt_depth = node.get_latest()

        if rgb is None:
            placeholder = blank_tile("waiting for RGB...")
            grid = build_grid([placeholder] * 4, [placeholder] * 4)
            cv2.imshow(win, grid)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
            continue

        # ── Full training augmentation pipeline ───────────────────────────────
        # action_magnitude=0.3: representative mid-level lateral blur for display
        rgb_aug = DepthEstimator._augment_rgb(rgb, action_magnitude=0.3)

        # Estimate depth from augmented frame (no internal augmentation to avoid double)
        est_depth_raw = estimator.estimate(rgb_aug, training=False)

        # Apply CLAHE — same as training pipeline
        gray_uint8 = (est_depth_raw * 255).astype(np.uint8)
        gray_uint8 = estimator._clahe.apply(gray_uint8)
        est_depth  = gray_uint8.astype(np.float32) / 255.0

        frame_stack = np.roll(frame_stack, shift=-1, axis=0)
        frame_stack[-1] = est_depth

        # ── Row 0: raw RGB | augmented RGB | depth+CLAHE | AirSim penalty ─────
        tile_rgb_raw  = rgb_tile(rgb,     "RGB (raw)")
        tile_rgb_proc = rgb_tile(rgb_aug, "RGB (augmented — model input)")
        tile_est      = gray_tile(est_depth, "depth + CLAHE (model input)")

        if gt_depth is not None:
            tile_penalty = gt_depth_tile(gt_depth, PROX_THRESHOLD,
                                         "AirSim 3.5m", show_roi=True)
        else:
            tile_penalty = blank_tile("no GT depth")

        # ── Row 1: frame stack + AirSim 15m (debug) ──────────────────────────
        stack_tiles = [
            gray_tile(
                frame_stack[i],
                f"stack t-{STACK_FRAMES - 1 - i}"
                + (" (latest)" if i == STACK_FRAMES - 1 else " (oldest)" if i == 0 else ""),
            )
            for i in range(STACK_FRAMES)
        ]

        if gt_depth is not None:
            tile_debug = gt_depth_tile(gt_depth, DEBUG_RANGE,
                                       f"AirSim {DEBUG_RANGE:.0f}m (debug)")
        else:
            tile_debug = blank_tile("no GT depth")

        grid = build_grid(
            [tile_rgb_raw, tile_rgb_proc, tile_est, tile_penalty],
            [*stack_tiles,                              tile_debug],
        )

        cv2.imshow(win, grid)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v6 debug viewer — 8-panel 4:3 display')
    parser.add_argument('--fps',         type=int,  default=10)
    parser.add_argument('--topic',       type=str,
                        default='/airsim_node/SimpleFlight/front_center_Scene/image')
    parser.add_argument('--depth-topic', type=str,
                        default='/airsim_node/SimpleFlight/front_center_DepthPerspective/image')
    args = parser.parse_args()

    main(fps=args.fps, scene_topic=args.topic, depth_topic=args.depth_topic)
