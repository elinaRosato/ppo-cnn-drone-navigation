"""
ROS2 Camera Bridge for AirSim — v3 (RGB output)

Thread-safe subscriber that caches the latest RGB frame and depth frame
from the AirSim ROS2 image topics.  In v3 the RGB frame is kept as a
3-channel (H, W, 3) uint8 array so that the DepthEstimator can consume
it directly — contrast to v2 which converted to grayscale here.

Usage
-----
    from ros2_bridge import ROS2CameraBridge
    from depth_estimator import DepthEstimator

    bridge = ROS2CameraBridge()
    bridge.start()

    estimator = DepthEstimator()
    env = ObstacleAvoidanceEnv(ros2_bridge=bridge, depth_estimator=estimator)
    bridge.stop()

Requirements
------------
    pip install opencv-python transformers
    source /opt/ros/humble/setup.bash
    sudo apt install ros-humble-cv-bridge

Topics (default vehicle SimpleFlight):
    /airsim_node/SimpleFlight/front_center_Scene/image            (sensor_msgs/Image)
    /airsim_node/SimpleFlight/front_center_DepthPerspective/image (sensor_msgs/Image)
"""

import threading
from typing import Optional
import numpy as np
import cv2


class ROS2CameraBridge:
    """
    Subscribes to AirSim Scene and DepthPerspective topics in a background
    thread, caching the latest frames:
      - RGB   → (H, W, 3) uint8  — passed directly to DepthEstimator
      - Depth → (H, W)   float32 in metres — GT depth for the privileged critic

    All getters are thread-safe.
    """

    def __init__(
        self,
        image_topic: str = '/airsim_node/SimpleFlight/front_center_Scene/image',
        depth_topic: str = '/airsim_node/SimpleFlight/front_center_DepthPerspective/image',
        target_size: tuple = (256, 144),
    ):
        """
        Parameters
        ----------
        image_topic : ROS2 topic for the AirSim RGB camera stream.
        depth_topic : ROS2 topic for the AirSim depth camera stream.
            Set to None to disable (depth will fall back to the AirSim API).
        target_size : (width, height) to resize incoming frames to. Must match
            the environment's img_width / img_height (default 192×192).
        """
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.target_size = target_size

        self._latest_frame: Optional[np.ndarray] = None   # (H, W, 3) uint8 RGB
        self._latest_depth: Optional[np.ndarray] = None   # (H, W) float32 metres
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the ROS2 subscriber in a background daemon thread."""
        if self._running:
            print("[ROS2Bridge] Already running — ignoring start()")
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._spin,
            name='ros2_camera_bridge',
            daemon=True,
        )
        self._thread.start()
        topics = self.image_topic
        if self.depth_topic:
            topics += f" + {self.depth_topic}"
        print(f"[ROS2Bridge] Subscriber thread started → {topics}")

    def stop(self):
        """Signal the background thread to shut down and wait for it."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        print("[ROS2Bridge] Subscriber stopped")

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the most recent RGB frame (H, W, 3) uint8, or None.

        Returns None until the first image has been received from the topic.
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_depth(self) -> Optional[np.ndarray]:
        """Return the most recent depth frame (H, W) float32 in metres, or None."""
        with self._lock:
            return self._latest_depth.copy() if self._latest_depth is not None else None

    @property
    def has_frame(self) -> bool:
        """True once at least one RGB frame has been received."""
        with self._lock:
            return self._latest_frame is not None

    @property
    def has_depth(self) -> bool:
        """True once at least one depth frame has been received."""
        with self._lock:
            return self._latest_depth is not None

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _spin(self):
        """Initialise rclpy, create the subscriber node, and spin until stopped."""
        try:
            import rclpy
            from rclpy.node import Node
            from sensor_msgs.msg import Image
            from cv_bridge import CvBridge
        except ImportError as exc:
            print(f"[ROS2Bridge] ERROR: missing ROS2 dependency — {exc}")
            print("[ROS2Bridge] Make sure ROS2 is sourced and cv_bridge is installed:")
            print("             source /opt/ros/humble/setup.bash")
            print("             sudo apt install ros-humble-cv-bridge")
            self._running = False
            return

        rclpy.init()
        cv_bridge = CvBridge()

        target_size = self.target_size
        lock = self._lock
        outer = self

        class _CameraNode(Node):
            def __init__(self):
                super().__init__('airsim_camera_bridge_v3')

                self.create_subscription(
                    Image,
                    outer.image_topic,
                    self._image_cb,
                    10,
                )
                self.get_logger().info(
                    f"Subscribed to {outer.image_topic} — waiting for RGB frames…"
                )

                if outer.depth_topic:
                    self.create_subscription(
                        Image,
                        outer.depth_topic,
                        self._depth_cb,
                        10,
                    )
                    self.get_logger().info(
                        f"Subscribed to {outer.depth_topic} — waiting for depth frames…"
                    )

            def _image_cb(self, msg: Image):
                try:
                    # Keep as RGB (3-channel) for DepthEstimator — do NOT convert to grayscale
                    cv_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    w, h = target_size
                    rgb = cv2.resize(cv_img, (w, h), interpolation=cv2.INTER_AREA)
                    with lock:
                        outer._latest_frame = rgb
                except Exception as exc:
                    self.get_logger().warning(f"RGB callback error: {exc}")

            def _depth_cb(self, msg: Image):
                try:
                    # AirSim publishes depth as 32FC1 (float32 metres)
                    depth = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                    w, h = target_size
                    depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
                    with lock:
                        outer._latest_depth = depth
                except Exception as exc:
                    self.get_logger().warning(f"Depth callback error: {exc}")

        node = _CameraNode()

        try:
            while self._running and rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
