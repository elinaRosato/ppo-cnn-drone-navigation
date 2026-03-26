"""
Live camera feed viewer — shows what the model actually sees during training.

Subscribes to the same ROS2 topic as the training script and applies the
identical preprocessing pipeline (BGR→grayscale, resize to 84×84 with
INTER_AREA) so the display matches exactly what the model receives.

Displays the grayscale frame (128×128, upscaled for visibility) plus,
optionally, the 4-frame stack side-by-side for debugging temporal context.

Usage:
    python view_camera.py
    python view_camera.py --fps 10
    python view_camera.py --stack     # show all 4 stacked frames
    python view_camera.py --topic /airsim_node/MyDrone/front_center_Scene/image

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
IMG_SIZE = 128
DISPLAY_SIZE = 400  # upscale for visibility


class FrameReceiver(Node):
    def __init__(self, topic: str):
        super().__init__('view_camera')
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None

        self.create_subscription(Image, topic, self._callback, 10)
        self.get_logger().info(f"Subscribed to {topic}")

    def _callback(self, msg: Image):
        try:
            # Topic publishes rgb8 — convert to gray directly
            cv_img = self._bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            with self._lock:
                if self._latest is None:
                    self.get_logger().info(f"First frame received — shape={gray.shape} min={gray.min()} max={gray.max()}")
                self._latest = gray
        except Exception as exc:
            self.get_logger().warning(f"Callback error: {exc}")

    def get_latest(self):
        with self._lock:
            return self._latest.copy() if self._latest is not None else None


def main(fps: int, show_stack: bool, topic: str):
    rclpy.init()
    node = FrameReceiver(topic)

    # Spin ROS2 in a background thread so OpenCV display runs on the main thread
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    wait_ms = max(1, int(1000 / fps))
    frame_stack = np.zeros((STACK_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    cv2.namedWindow("Drone Camera (128x128 grayscale — model input)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Drone Camera (128x128 grayscale — model input)", DISPLAY_SIZE, DISPLAY_SIZE)

    print(f"Waiting for frames on {topic} ... Press Q to quit.")

    try:
        while True:
            gray = node.get_latest()

            if gray is not None:
                frame_stack = np.roll(frame_stack, shift=-1, axis=0)
                frame_stack[-1] = gray

                if show_stack:
                    frames = [cv2.resize(frame_stack[i], (200, 200)) for i in range(STACK_FRAMES)]
                    display = np.concatenate(frames, axis=1)
                    cv2.imshow("Frame Stack (oldest -> newest)", display)
                else:
                    # INTER_NEAREST keeps the exact 84x84 pixels visible without blurring
                    cv2.imshow(
                        "Drone Camera (128x128 grayscale — model input)",
                        cv2.resize(gray, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_NEAREST),
                    )

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
    parser.add_argument('--topic', type=str,
                        default='/airsim_node/SimpleFlight/front_center_Scene/image',
                        help='ROS2 image topic to subscribe to')
    args = parser.parse_args()

    main(fps=args.fps, show_stack=args.stack, topic=args.topic)
