"""
Live debug viewer — v3 (8-panel tiled display).

Shows everything in a single 4×2 grid window:

  Row 0 │ RGB (raw)  │ RGB (processed) │ Depth est. (grayscale) │ AirSim 3.5m (penalty) │
  Row 1 │ Stack t-2  │ Stack t-1       │ Stack t-0 (latest)     │ AirSim 15m  (critic)  │

Depth Anything V2 outputs a single-channel float32 disparity map — not RGB.
The model receives exactly the grayscale tiles shown in cols 3-4, row 1.
No clipping is applied to the estimated depth: what you see is what the model gets.

GT AirSim depths (col 4) use COLORMAP_HOT for visual clarity: black=far, white=close.
These are training-only signals — the actor never sees them.

Usage:
    python3 view_camera.py                      # clean input
    python3 view_camera.py --augment            # apply training RGB augmentation
    python3 view_camera.py --fps 10
    python3 view_camera.py --topic /airsim_node/MyDrone/front_center_Scene/image

Press Q to quit.
"""

import argparse
import sys
import os
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from depth_estimator import DepthEstimator

STACK_FRAMES    = 3
IMG_SIZE        = 192
TILE            = 200       # px per panel (4 wide × 2 tall = 800 × 400 window)
PROX_THRESHOLD  = 3.5       # metres — must match avoidance_env.py
CRITIC_RANGE    = 15.0      # metres — must match avoidance_env.py
PROX_WEIGHT     = 0.5


# ── ROS2 receiver ─────────────────────────────────────────────────────────────

class CameraReceiver(Node):
    def __init__(self, scene_topic: str, depth_topic: str):
        super().__init__('view_camera_v3')
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
            rgb = cv2.resize(cv_img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            with self._lock:
                self._latest_rgb = rgb
        except Exception as exc:
            self.get_logger().warning(f"Scene callback error: {exc}")

    def _depth_cb(self, msg: Image):
        try:
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32)
            depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
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
    cv2.putText(tile, text, (4, TILE - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return tile


def rgb_tile(rgb: np.ndarray, label: str) -> np.ndarray:
    """RGB (stored as RGB) → BGR tile for cv2 display."""
    bgr = cv2.cvtColor(cv2.resize(rgb, (TILE, TILE), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_RGB2BGR)
    return _label(bgr, label)


def gray_tile(depth_01: np.ndarray, label: str) -> np.ndarray:
    """Float32 [0,1] depth → true grayscale BGR tile.  This is what the model sees."""
    uint8 = (np.clip(depth_01, 0.0, 1.0) * 255).astype(np.uint8)
    gray  = cv2.resize(uint8, (TILE, TILE), interpolation=cv2.INTER_NEAREST)
    bgr   = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return _label(bgr, label)


def gt_depth_tile(gt: np.ndarray, clip_m: float, label: str,
                  show_roi: bool = False) -> np.ndarray:
    """Metric GT depth → COLORMAP_HOT tile.  Closer = brighter."""
    clipped    = np.clip(gt, 0.0, clip_m)
    uint8      = ((clip_m - clipped) / clip_m * 255).astype(np.uint8)
    coloured   = cv2.applyColorMap(
        cv2.resize(uint8, (TILE, TILE), interpolation=cv2.INTER_NEAREST),
        cv2.COLORMAP_HOT,
    )

    if show_roi:
        # Draw centre-50% rectangle used for penalty/critic computation
        s = TILE / IMG_SIZE
        q = IMG_SIZE // 4
        cv2.rectangle(coloured,
                      (int(q * s), int(q * s)),
                      (int(3 * q * s), int(3 * q * s)),
                      (255, 255, 255), 1)

        center = clipped[q: 3 * q, q: 3 * q]
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
        # Just show min depth in the centre ROI
        q     = IMG_SIZE // 4
        min_d = float(np.min(clipped[q: 3 * q, q: 3 * q]))
        _label(coloured, f"{label}  min={min_d:.1f}m")

    return coloured


def est_depth_tile(est: np.ndarray) -> np.ndarray:
    """Estimated depth [0,1] as true grayscale — exactly what the model receives."""
    return gray_tile(est, "depth est. (model input)")


def blank_tile(text: str = "waiting...") -> np.ndarray:
    tile = np.zeros((TILE, TILE, 3), dtype=np.uint8)
    cv2.putText(tile, text, (10, TILE // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1, cv2.LINE_AA)
    return tile


def build_grid(row0: list, row1: list) -> np.ndarray:
    """Concatenate two rows of TILE×TILE BGR tiles into one display image."""
    top    = np.concatenate(row0, axis=1)
    bottom = np.concatenate(row1, axis=1)
    # Thin separator line between rows
    sep = np.full((2, top.shape[1], 3), 60, dtype=np.uint8)
    return np.concatenate([top, sep, bottom], axis=0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(fps: int, augment: bool, scene_topic: str, depth_topic: str):
    rclpy.init()
    node = CameraReceiver(scene_topic, depth_topic)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print("Loading Depth Anything V2 Small...")
    estimator = DepthEstimator(target_size=(IMG_SIZE, IMG_SIZE))
    print("Estimator ready.")

    wait_ms     = max(1, int(1000 / fps))
    frame_stack = np.zeros((STACK_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    win = "v3 debug view  [Q = quit]"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, TILE * 4, TILE * 2 + 2)

    aug_label  = "+augment" if augment else "clean"
    print(f"Waiting for frames on {scene_topic} ... Press Q to quit.")

    while True:
        rgb, gt_depth = node.get_latest()

        if rgb is None:
            placeholder = blank_tile("waiting for RGB...")
            grid = build_grid(
                [placeholder] * 4,
                [placeholder] * 4,
            )
            cv2.imshow(win, grid)
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
            continue

        # ── Process RGB ───────────────────────────────────────────────────────
        rgb_aug = DepthEstimator._augment_rgb(rgb) if augment else rgb

        est_depth = estimator.estimate(rgb_aug, training=False)  # augmentation already applied

        frame_stack = np.roll(frame_stack, shift=-1, axis=0)
        frame_stack[-1] = est_depth

        # ── Row 0: camera feeds + depth est + AirSim penalty depth ───────────
        tile_rgb_raw  = rgb_tile(rgb,     "RGB (raw)")
        tile_rgb_proc = rgb_tile(rgb_aug, f"RGB ({aug_label})")
        tile_est      = est_depth_tile(est_depth)

        if gt_depth is not None:
            tile_airsim_35 = gt_depth_tile(gt_depth, PROX_THRESHOLD,
                                           "AirSim 3.5m", show_roi=True)
        else:
            tile_airsim_35 = blank_tile("no GT depth")

        # ── Row 1: frame stack + AirSim critic depth ─────────────────────────
        stack_tiles = [
            gray_tile(frame_stack[i],
                      f"stack t-{STACK_FRAMES - 1 - i}"
                      + (" (latest)" if i == STACK_FRAMES - 1 else " (oldest)" if i == 0 else ""))
            for i in range(STACK_FRAMES)
        ]

        if gt_depth is not None:
            tile_airsim_15 = gt_depth_tile(gt_depth, CRITIC_RANGE,
                                           f"AirSim {CRITIC_RANGE:.0f}m (critic)")
        else:
            tile_airsim_15 = blank_tile("no GT depth")

        grid = build_grid(
            [tile_rgb_raw, tile_rgb_proc, tile_est, tile_airsim_35],
            [*stack_tiles,                              tile_airsim_15],
        )

        cv2.imshow(win, grid)
        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='v3 debug viewer — 8-panel tiled display')
    parser.add_argument('--fps',         type=int,  default=10)
    parser.add_argument('--augment',     action='store_true',
                        help='Apply training RGB augmentation before estimation')
    parser.add_argument('--topic',       type=str,
                        default='/airsim_node/SimpleFlight/front_center_Scene/image')
    parser.add_argument('--depth-topic', type=str,
                        default='/airsim_node/SimpleFlight/front_center_DepthPerspective/image')
    args = parser.parse_args()

    main(fps=args.fps, augment=args.augment,
         scene_topic=args.topic, depth_topic=args.depth_topic)
