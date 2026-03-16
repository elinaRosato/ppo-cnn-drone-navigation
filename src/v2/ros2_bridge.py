"""
ROS2 Camera Bridge for AirSim

Thread-safe subscriber that caches the latest grayscale RGB frame and depth
frame from the AirSim ROS2 image topics. Designed as a drop-in image source
for ObstacleAvoidanceEnv, replacing both simGetImages() calls with a
continuous stream at the native render rate (~20-30 Hz).

Usage
-----
    from ros2_bridge import ROS2CameraBridge

    bridge = ROS2CameraBridge()
    bridge.start()

    env = ObstacleAvoidanceEnv(ros2_bridge=bridge)
    # ... train / run episodes ...
    bridge.stop()

Requirements
------------
    pip install opencv-python
    # ROS2 (Humble or Foxy) must be sourced before running:
    source /opt/ros/humble/setup.bash
    # cv_bridge (ROS2 package):
    sudo apt install ros-humble-cv-bridge

The AirSim ROS2 bridge must be running before bridge.start() is called:
    ros2 launch airsim_ros_pkgs airsim_node.launch.py host:=<AIRSIM_IP>

Topics published by the bridge (default vehicle SimpleFlight):
    /airsim_node/SimpleFlight/front_center_Scene/image            (sensor_msgs/Image)
    /airsim_node/SimpleFlight/front_center_DepthPerspective/image (sensor_msgs/Image)
"""

import threading
from typing import Optional
import numpy as np
import cv2


class ROS2CameraBridge:
    """
    Subscribes to the AirSim Scene and DepthPerspective image topics in a
    background thread, caching the latest frames:
      - RGB  → grayscale (H, W) uint8
      - Depth → float32  (H, W) in metres

    All getters are thread-safe and can be called from any thread,
    including the training loop.
    """

    def __init__(
        self,
        image_topic: str = '/airsim_node/SimpleFlight/front_center_Scene/image',
        depth_topic: str = '/airsim_node/SimpleFlight/front_center_DepthPerspective/image',
        target_size: tuple = (84, 84),
    ):
        """
        Parameters
        ----------
        image_topic : str
            ROS2 topic name for the AirSim RGB camera stream.
            Adjust the vehicle name (SimpleFlight) to match settings.json.
        depth_topic : str
            ROS2 topic name for the AirSim depth camera stream.
            Set to None to disable depth subscription (depth will then fall
            back to the AirSim Python API in the environment).
        target_size : tuple
            (width, height) to resize incoming frames to. Must match the
            environment's img_width / img_height (default 84x84).
        """
        self.image_topic = image_topic
        self.depth_topic = depth_topic
        self.target_size = target_size

        self._latest_frame: Optional[np.ndarray] = None
        self._latest_depth: Optional[np.ndarray] = None
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self):
        """Start the ROS2 subscriber in a background daemon thread.

        Returns immediately. Frames will start arriving once the AirSim
        ROS2 bridge is publishing on image_topic / depth_topic.
        """
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
        """Return the most recent grayscale frame (H, W) uint8, or None.

        Returns None until the first image has been received from the topic.
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def get_latest_depth(self) -> Optional[np.ndarray]:
        """Return the most recent depth frame (H, W) float32 in metres, or None.

        Returns None if depth_topic is disabled or no frame has arrived yet.
        """
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

        # Capture outer-scope references for use inside the nested class
        target_size = self.target_size
        lock = self._lock
        outer = self

        class _CameraNode(Node):
            def __init__(self):
                super().__init__('airsim_camera_bridge')

                self.create_subscription(
                    Image,
                    outer.image_topic,
                    self._image_cb,
                    10,  # QoS depth
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
                    cv_img = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
                    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
                    with lock:
                        outer._latest_frame = gray
                except Exception as exc:
                    self.get_logger().warning(f"RGB callback error: {exc}")

            def _depth_cb(self, msg: Image):
                try:
                    # AirSim publishes depth as 32FC1 (single-channel float32, metres)
                    depth = cv_bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
                    depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
                    with lock:
                        outer._latest_depth = depth
                except Exception as exc:
                    self.get_logger().warning(f"Depth callback error: {exc}")

        node = _CameraNode()

        try:
            while self._running and rclpy.ok():
                # spin_once with a short timeout keeps the loop responsive
                # to stop() without blocking indefinitely
                rclpy.spin_once(node, timeout_sec=0.05)
        finally:
            node.destroy_node()
            rclpy.try_shutdown()
