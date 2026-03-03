# AirSim ROS2 Bridge Setup Guide

Using the AirSim ROS2 bridge instead of the Python API allows camera images to be
published as a continuous stream at the simulation's native render rate (20–60 Hz),
eliminating the TCP round-trip bottleneck of `simGetImages()`. This enables training
and inference loops that closely match the 30 Hz target of real-world deployment.

---

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Ubuntu | 20.04 or 22.04 (WSL2 on Windows is possible but adds latency) |
| ROS2 | Humble (22.04) or Foxy (20.04) |
| AirSim | Built from source (the ROS2 wrapper is not in pre-built binaries) |
| Python | 3.8+ |
| CUDA | Optional but recommended for model inference |

> **Windows note**: AirSim itself runs on Windows, but the ROS2 bridge runs on Linux.
> The bridge connects to AirSim over the network, so running AirSim on Windows and
> the bridge on WSL2 or a separate Linux machine is a supported configuration.

---

## Step 1: Build AirSim from Source

The ROS2 bridge is located in `AirSim/ros2/` and must be compiled alongside AirSim.

```bash
git clone https://github.com/microsoft/AirSim.git
cd AirSim
./setup.sh       # installs dependencies
./build.sh       # builds AirSim core libraries
```

---

## Step 2: Install ROS2

Follow the official ROS2 installation guide for your Ubuntu version:
- **Ubuntu 22.04**: install ROS2 Humble
- **Ubuntu 20.04**: install ROS2 Foxy

After installation, source the setup file in your `.bashrc`:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 3: Build the AirSim ROS2 Package

```bash
cd AirSim/ros2
source /opt/ros/humble/setup.bash

# Install ROS2 dependencies
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
source install/setup.bash
```

---

## Step 4: Configure AirSim Camera

In your AirSim `settings.json`, configure the camera to publish at the desired resolution
and frame rate. The ROS2 bridge will pick up this configuration automatically.

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 1.0,
  "Vehicles": {
    "SimpleFlight": {
      "VehicleType": "SimpleFlight",
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 84,
              "Height": 84,
              "FOV_Degrees": 120
            }
          ],
          "X": 0.50,
          "Y": 0,
          "Z": -0.25,
          "Pitch": 0,
          "Roll": 0,
          "Yaw": 0
        }
      }
    }
  }
}
```

---

## Step 5: Launch the ROS2 Bridge

Start AirSim (Unreal Engine), then in a Linux terminal:

```bash
source AirSim/ros2/install/setup.bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py \
  output:=screen \
  host:=<AIRSIM_HOST_IP>   # use 127.0.0.1 if on the same machine / WSL2
```

The bridge will begin publishing topics including:

| Topic | Type | Description |
|-------|------|-------------|
| `/airsim_node/SimpleFlight/front_center/Scene` | `sensor_msgs/Image` | RGB camera stream |
| `/airsim_node/SimpleFlight/front_center/DepthPerspective` | `sensor_msgs/Image` | Depth stream |
| `/airsim_node/SimpleFlight/odom_local_ned` | `nav_msgs/Odometry` | Drone position/velocity |

Verify topics are publishing:

```bash
ros2 topic hz /airsim_node/SimpleFlight/front_center/Scene
# Should show ~20-60 Hz depending on hardware
```

---

## Step 6: Rewrite the Training Loop as ROS2 Nodes

The training environment must change from a blocking request-response model to an
event-driven subscriber model.

### Architecture

```
AirSim → ROS2 Image Topic (30 Hz)
              ↓
        Image Subscriber Node
              ↓ latest frame (cached)
        RL Step Loop (runs at subscriber rate)
              ↓ action
        Velocity Command Publisher
              ↓
        /airsim_node/vel_cmd_body_frame topic
              ↓
        AirSim drone
```

### Key ROS2 Topics to Publish/Subscribe

**Subscribe (inputs to the model):**
```python
# Latest RGB frame
self.create_subscription(Image, '/airsim_node/SimpleFlight/front_center/Scene',
                         self.image_callback, 10)

# Drone odometry (position for goal check)
self.create_subscription(Odometry, '/airsim_node/SimpleFlight/odom_local_ned',
                         self.odom_callback, 10)
```

**Publish (model output):**
```python
# Body-frame velocity command
self.vel_pub = self.create_publisher(
    VelCmd, '/airsim_node/SimpleFlight/vel_cmd_body_frame', 10)
```

### Minimal Node Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from airsim_interfaces.msg import VelCmd
import cv2
import numpy as np
from cv_bridge import CvBridge
from stable_baselines3 import PPO

class AvoidanceNode(Node):
    def __init__(self):
        super().__init__('avoidance_node')
        self.bridge = CvBridge()
        self.model = PPO.load('path/to/model.zip')
        self.latest_frame = None
        self.frame_stack = np.zeros((4, 84, 84), dtype=np.uint8)
        self.position = None

        self.create_subscription(Image,
            '/airsim_node/SimpleFlight/front_center/Scene',
            self.image_callback, 10)
        self.create_subscription(Odometry,
            '/airsim_node/SimpleFlight/odom_local_ned',
            self.odom_callback, 10)
        self.vel_pub = self.create_publisher(VelCmd,
            '/airsim_node/SimpleFlight/vel_cmd_body_frame', 10)

        # Run inference at 30 Hz
        self.create_timer(1.0 / 30.0, self.inference_step)

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (84, 84))
        # Update frame stack
        self.frame_stack = np.roll(self.frame_stack, shift=-1, axis=0)
        self.frame_stack[-1] = gray

    def odom_callback(self, msg):
        self.position = msg.pose.pose.position

    def inference_step(self):
        if self.latest_frame is None or self.position is None:
            return
        action, _ = self.model.predict(self.frame_stack, deterministic=True)
        cmd = VelCmd()
        cmd.vx = 3.0 + float(action[0])  # base_speed + lateral correction
        cmd.vy = 0.0
        cmd.vz = float(action[1]) * 0.5  # vertical correction
        self.vel_pub.publish(cmd)

def main():
    rclpy.init()
    node = AvoidanceNode()
    rclpy.spin(node)

if __name__ == '__main__':
    main()
```

---

## Step 7: Training with the ROS2 Bridge

For RL training, the environment class must be adapted to use the ROS2 subscriber
pattern. The main differences from the Python API version:

1. **Image capture is non-blocking** — the latest frame is always available from the
   last callback. The step loop runs at the timer rate, not the capture rate.
2. **Reward and collision info** still come from the AirSim Python API (or ROS2
   service calls) — the bridge does not expose collision events as topics by default.
3. **Reset** still requires the AirSim Python API (`client.reset()`).

A hybrid approach is common: use the ROS2 bridge for high-frequency image streaming,
and the Python API only for environment management (reset, collision check, takeoff).

---

## Expected Improvement

| Method | Effective Hz | Scene change at 3 m/s |
|--------|-------------|----------------------|
| Python API, 256×256 | <1 Hz | >300 cm |
| Python API, 84×84 | ~5 Hz | ~60 cm |
| ROS2 bridge, 84×84 | ~20–30 Hz | ~10–15 cm |
| Real deployment (RealSense) | 30 Hz | 10 cm |

With the ROS2 bridge at 84×84, the simulation closely matches real deployment conditions
without requiring speed domain randomization as a compensatory technique.

---

## Complexity vs. Benefit Summary

| Approach | Setup complexity | Sim Hz | Sim-to-real gap |
|----------|-----------------|--------|-----------------|
| Python API + speed randomization | Low | ~5 Hz | Moderate |
| Python API, 84×84 RGB-only | Low | ~8 Hz | Moderate |
| ROS2 bridge | High | ~20–30 Hz | Small |

The ROS2 bridge is the right long-term architecture if the real drone also runs ROS2
(e.g., Jetson + MAVROS + RealSense driver), as the simulation and real software stacks
become nearly identical.
