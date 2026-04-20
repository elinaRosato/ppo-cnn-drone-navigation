"""
Live camera feed viewer — shows what the model actually sees during training.

Subscribes to the same ROS2 topics as the training script and applies the
identical preprocessing pipeline (BGR→grayscale, resize to 192×192 with
INTER_AREA, CLAHE contrast normalisation) so the display matches exactly
what the model receives.

Depth view (--depth): clips to [0, prox_threshold=3.5 m] and renders with
COLORMAP_HOT (black=far/safe, white=very close/dangerous). The white
rectangle marks the centre-50% region used by avoidance_env.py for both
the proximity penalty and the clear-path bonus. Min depth and the resulting
reward signal are shown as a text overlay.

Augmentation view (--augment): applies the same random noise, brightness/contrast
jitter and blur used during training so you can see how distorted the input
looks to the model.

Usage:
    python view_camera.py
    python view_camera.py --fps 10
    python view_camera.py --stack               # show all 4 stacked frames
    python view_camera.py --depth               # show depth alongside grayscale
    python view_camera.py --augment             # apply training augmentation
    python view_camera.py --topic /airsim_node/MyDrone/front_center_Scene/image
    python view_camera.py --depth-topic /airsim_node/MyDrone/front_center_DepthPerspective/image

Press Q to quit.
"""

import argparse
import threading
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

STACK_FRAMES = 4
IMG_SIZE = 192
DISPLAY_SIZE = 400       # upscale for visibility
PROX_THRESHOLD = 3.5     # metres — must match avoidance_env.py
PROX_WEIGHT = 0.5        # must match avoidance_env.py proximity penalty weight

# CLAHE — identical settings to avoidance_env.py
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def augment_frame(gray: np.ndarray) -> np.ndarray:
    """Apply the same augmentation pipeline as avoidance_env.py._augment_frame.

    Gaussian noise σ∈[0.5, 3.0], brightness/contrast jitter ±20/±15%,
    30%-chance GaussianBlur k∈{3, 5}. Input and output are uint8.
    """
    img = gray.astype(np.float32)
    sigma = np.random.uniform(0.5, 3.0)
    img += np.random.normal(0, sigma, img.shape).astype(np.float32)
    alpha = np.random.uniform(0.85, 1.15)   # contrast
    beta  = np.random.uniform(-20, 20)       # brightness
    img = img * alpha + beta
    img = np.clip(img, 0, 255).astype(np.uint8)
    if np.random.random() < 0.3:
        k = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    return img


class CameraReceiver(Node):
    def __init__(self, scene_topic: str, depth_topic: str = None):
        super().__init__('view_camera')
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest_gray = None
        self._latest_depth = None

        self.create_subscription(Image, scene_topic, self._scene_callback, 10)
        self.get_logger().info(f"Subscribed to scene:  {scene_topic}")

        if depth_topic:
            self.create_subscription(Image, depth_topic, self._depth_callback, 10)
            self.get_logger().info(f"Subscribed to depth:  {depth_topic}")

    def _scene_callback(self, msg: Image):
        try:
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            gray = _CLAHE.apply(gray)   # match avoidance_env.py preprocessing
            with self._lock:
                if self._latest_gray is None:
                    self.get_logger().info(
                        f"First scene frame — shape={gray.shape} "
                        f"min={gray.min()} max={gray.max()}"
                    )
                self._latest_gray = gray
        except Exception as exc:
            self.get_logger().warning(f"Scene callback error: {exc}")

    def _depth_callback(self, msg: Image):
        try:
            # AirSim publishes depth as 32FC1 (float32 metres)
            depth = self._bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            depth = depth.astype(np.float32)
            depth = cv2.resize(depth, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            with self._lock:
                if self._latest_depth is None:
                    self.get_logger().info(
                        f"First depth frame — shape={depth.shape} "
                        f"min={depth.min():.2f}m max={depth.max():.2f}m"
                    )
                self._latest_depth = depth
        except Exception as exc:
            self.get_logger().warning(f"Depth callback error: {exc}")

    def get_latest(self):
        with self._lock:
            gray = self._latest_gray.copy() if self._latest_gray is not None else None
            depth = self._latest_depth.copy() if self._latest_depth is not None else None
        return gray, depth


def render_depth(depth: np.ndarray, display_size: int) -> np.ndarray:
    """Render depth with the same clip used by the reward function.

    - Clips to [0, PROX_THRESHOLD] metres — identical to avoidance_env.py
    - Closer = brighter/hotter (COLORMAP_HOT): black=safe, white=dangerous
    - White rectangle: centre-50% region sampled for penalty/bonus
    - Text overlay: min depth in that region + reward signal value
    """
    clipped = np.clip(depth, 0.0, PROX_THRESHOLD)

    # 0 m (very close) → 255 (hot), PROX_THRESHOLD (far) → 0 (black)
    normalised = ((PROX_THRESHOLD - clipped) / PROX_THRESHOLD * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_HOT)
    coloured = cv2.resize(coloured, (display_size, display_size), interpolation=cv2.INTER_NEAREST)

    # Centre-50% region coordinates (matches avoidance_env.py exactly)
    ry1, ry2 = IMG_SIZE // 4, 3 * IMG_SIZE // 4
    rx1, rx2 = IMG_SIZE // 4, 3 * IMG_SIZE // 4

    scale = display_size / IMG_SIZE
    cv2.rectangle(
        coloured,
        (int(rx1 * scale), int(ry1 * scale)),
        (int(rx2 * scale), int(ry2 * scale)),
        (255, 255, 255), 2
    )

    # Compute the same reward signal as avoidance_env.py
    min_d = float(np.min(clipped[ry1:ry2, rx1:rx2]))
    if min_d < PROX_THRESHOLD:
        penalty = (PROX_THRESHOLD - min_d) / PROX_THRESHOLD * PROX_WEIGHT
        label = f"min={min_d:.2f}m  penalty=-{penalty:.3f}"
        colour = (80, 160, 255)   # orange in BGR
    else:
        label = f"min>{PROX_THRESHOLD:.1f}m  CLEAR  bonus=+0.00..0.05"
        colour = (80, 220, 80)    # green in BGR

    cv2.putText(coloured, label, (8, display_size - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    return coloured


def main(fps: int, show_stack: bool, show_depth: bool, augment: bool,
         scene_topic: str, depth_topic: str):
    rclpy.init()
    node = CameraReceiver(scene_topic, depth_topic if show_depth else None)

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    wait_ms = max(1, int(1000 / fps))
    frame_stack = np.zeros((STACK_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    scene_win = "Drone Camera — 192×192 grayscale" + (" [AUGMENTED]" if augment else " (model input)")
    cv2.namedWindow(scene_win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(scene_win, DISPLAY_SIZE, DISPLAY_SIZE)

    if show_depth:
        cv2.namedWindow("Depth — 0–3.5 m clipped (reward signal)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth — 0–3.5 m clipped (reward signal)",
                         DISPLAY_SIZE, DISPLAY_SIZE)

    print(f"Waiting for frames on {scene_topic} ... Press Q to quit.")

    try:
        while True:
            gray, depth = node.get_latest()

            if gray is not None:
                display_gray = augment_frame(gray) if augment else gray
                frame_stack = np.roll(frame_stack, shift=-1, axis=0)
                frame_stack[-1] = display_gray

                if show_stack:
                    frames = [cv2.resize(frame_stack[i], (200, 200))
                              for i in range(STACK_FRAMES)]
                    cv2.imshow("Frame Stack (oldest → newest)",
                               np.concatenate(frames, axis=1))
                else:
                    cv2.imshow(
                        scene_win,
                        cv2.resize(display_gray, (DISPLAY_SIZE, DISPLAY_SIZE),
                                   interpolation=cv2.INTER_NEAREST),
                    )

            if show_depth and depth is not None:
                cv2.imshow("Depth — 0–3.5 m clipped (reward signal)",
                           render_depth(depth, DISPLAY_SIZE))

            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live camera feed viewer (ROS2)')
    parser.add_argument('--fps', type=int, default=10,
                        help='Display refresh rate in FPS (default: 10)')
    parser.add_argument('--stack', action='store_true',
                        help='Show all 4 stacked frames side-by-side')
    parser.add_argument('--depth', action='store_true',
                        help='Show depth camera with penalty region highlighted')
    parser.add_argument('--augment', action='store_true',
                        help='Apply training augmentation (noise, jitter, blur)')
    parser.add_argument('--topic', type=str,
                        default='/airsim_node/SimpleFlight/front_center_Scene/image',
                        help='ROS2 scene image topic')
    parser.add_argument('--depth-topic', type=str,
                        default='/airsim_node/SimpleFlight/front_center_DepthPerspective/image',
                        help='ROS2 depth topic (used with --depth)')
    args = parser.parse_args()

    main(fps=args.fps, show_stack=args.stack, show_depth=args.depth,
         augment=args.augment, scene_topic=args.topic, depth_topic=args.depth_topic)
