# AirSim ROS2 Bridge Setup — Windows + WSL2

This guide covers the complete setup to run the v2 training script with the
ROS2 bridge on a **Windows machine running AirSim**, using **WSL2 (Ubuntu)**
for the ROS2 bridge and the training script.

## Why bother?

The AirSim Python API `simGetImages()` takes ~500 ms per call (TCP round-trip
+ render sync). This limits training to ~1.5 Hz — meaning the 4-frame stack
spans ~2.7 seconds of real time. On a real drone at 30 Hz, the same 4 frames
span ~133 ms. The model trained in simulation sees a completely different
temporal signal than it will encounter on the real drone.

The ROS2 bridge publishes camera frames at the Unreal Engine render rate
(~20–30 Hz). The training loop reads from a local memory cache (< 1 ms) and
the bottleneck disappears.

| Method | Step rate | 4-frame span |
|--------|-----------|--------------|
| Python API | ~1.5 Hz | ~2.7 s |
| ROS2 bridge | ~30 Hz | ~133 ms |
| Real drone (RealSense) | 30 Hz | 133 ms |

---

## Architecture

```
Windows
│
├── AirSim (Unreal Engine)
│   └── publishes camera + physics at render rate
│
└── WSL2 (Ubuntu 22.04)
    ├── ROS2 Humble
    │   └── airsim_ros_pkgs  ← connects to AirSim over localhost
    │       ├── /airsim_node/SimpleFlight/front_center/Scene        (RGB ~30 Hz)
    │       └── /airsim_node/SimpleFlight/front_center/DepthPerspective (depth ~30 Hz)
    │
    └── Python training script (train.py)
        ├── ROS2CameraBridge  ← subscribes to above topics (background thread)
        └── AirSim Python API ← reset, arm, takeoff, collision, velocity commands
            └── connects to AirSim via Windows host IP (172.x.x.x)
```

---

## Step 1: Enable WSL2 on Windows

Open PowerShell as Administrator and run:

```powershell
wsl --install
```

This installs WSL2 with Ubuntu by default. Restart when prompted.

After restart, open the Ubuntu terminal from the Start menu and complete the
initial user setup (username + password).

Verify WSL2 is the default version:

```powershell
wsl --set-default-version 2
wsl -l -v   # should show Ubuntu with VERSION 2
```

---

## Step 2: Install ROS2 Humble in WSL2

Open the Ubuntu WSL2 terminal and run the official install:

```bash
# Set locale
sudo apt update && sudo apt install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# Add ROS2 apt repo
sudo apt install -y software-properties-common curl
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
  http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
  | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS2 Humble
sudo apt update
sudo apt install -y ros-humble-desktop python3-colcon-common-extensions python3-rosdep

# Source ROS2 in every new shell
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Initialise rosdep
sudo rosdep init
rosdep update
```

---

## Step 3: Build AirSim from Source in WSL2

The ROS2 bridge (`airsim_ros_pkgs`) is only available by building from source.

```bash
# Dependencies
sudo apt install -y \
  build-essential cmake git \
  ros-humble-cv-bridge ros-humble-image-transport \
  ros-humble-mavros-msgs ros-humble-geographic-msgs

# Clone AirSim
cd ~
git clone https://github.com/microsoft/AirSim.git
cd AirSim

# Build AirSim core libraries (needed by the ROS2 package)
./setup.sh
./build.sh
```

Then build the ROS2 package:

```bash
cd ~/AirSim/ros2
source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
echo "source ~/AirSim/ros2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

---

## Step 4: Install Python Dependencies in WSL2

```bash
# Python deps for training
pip install \
  airsim \
  gymnasium \
  stable-baselines3[extra] \
  opencv-python \
  numpy \
  torch torchvision   # or install the CUDA build from pytorch.org

# cv_bridge Python bindings (already installed via ROS2 apt above,
# but make sure it's accessible in your Python environment)
sudo apt install -y python3-cv-bridge
```

---

## Step 5: Configure AirSim on Windows

Copy `settings_sample.json` to `C:\Users\<YourName>\Documents\AirSim\settings.json`.

Key settings to verify:

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
            { "ImageType": 0, "Width": 256, "Height": 256, "FOV_Degrees": 120 },
            { "ImageType": 2, "Width": 256, "Height": 256, "FOV_Degrees": 120 }
          ],
          "X": 0.50, "Y": 0, "Z": -0.25,
          "Pitch": 0, "Roll": 0, "Yaw": 0
        }
      }
    }
  }
}
```

> **ClockSpeed must be 1.0 when using the ROS2 bridge.** Setting it higher
> speeds up physics but the ROS2 image topics still publish at the real-time
> render rate, creating desynchronisation between the physics and the images
> the model sees.

Both `ImageType 0` (Scene/RGB) and `ImageType 2` (DepthPerspective) must be
listed so the AirSim ROS2 bridge publishes both topics.

---

## Step 6: Find the Windows Host IP (from WSL2)

WSL2 runs in a virtual network. To connect from WSL2 to AirSim running on
Windows, you need the Windows host IP as seen from inside WSL2.

```bash
# Run this inside WSL2
export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $AIRSIM_HOST   # typically 172.x.x.x
```

Add it to your `~/.bashrc` so you don't have to set it every session:

```bash
echo 'export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '"'"'{print $2}'"'"')' >> ~/.bashrc
source ~/.bashrc
```

> **Windows Firewall:** AirSim listens on port 41451. If the connection is
> refused, add an inbound firewall rule in Windows:
> Settings → Windows Defender Firewall → Advanced Settings →
> Inbound Rules → New Rule → Port 41451 → Allow.

---

## Step 7: Launch AirSim on Windows

Start your Unreal Engine project (or a pre-built AirSim binary) on the Windows
side as normal. Wait for the simulation to fully load before proceeding.

---

## Step 8: Launch the ROS2 Bridge in WSL2

In a WSL2 terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/AirSim/ros2/install/setup.bash

ros2 launch airsim_ros_pkgs airsim_node.launch.py \
  output:=screen \
  host:=$AIRSIM_HOST
```

You should see log lines like:
```
[airsim_node]: Connected to AirSim!
[airsim_node]: Publishing image on /airsim_node/SimpleFlight/front_center/Scene
```

Verify both topics are publishing at the expected rate:

```bash
# In a second WSL2 terminal
ros2 topic hz /airsim_node/SimpleFlight/front_center/Scene
# Expected: ~20-30 Hz

ros2 topic hz /airsim_node/SimpleFlight/front_center/DepthPerspective
# Expected: ~20-30 Hz
```

---

## Step 9: Run Training in WSL2

Open a third WSL2 terminal, navigate to the `src/v2` directory, and run:

```bash
cd /path/to/ppo-cnn-drone-navigation/src/v2

# Make sure AIRSIM_HOST is set (if not already in .bashrc)
export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')

# Start training with the ROS2 bridge
python train.py --ros2

# Optional: specify total steps
python train.py --ros2 --steps 500000

# Resume a previous run with the bridge
python train.py --ros2 --resume
```

The script will:
1. Start `ROS2CameraBridge` in a background thread
2. Wait up to 10 seconds for the first frame to arrive
3. Print the confirmed AirSim connection host
4. Show the per-step timing breakdown every 25 steps so you can verify the
   bridge is working (`images=` should be < 5 ms, not ~600 ms)

---

## Verifying It Works

Look for these signs in the training output:

**Bridge connected:**
```
[ROS2Bridge] Subscriber thread started → .../Scene + .../DepthPerspective
ROS2 bridge ready.
Connected to AirSim at 172.x.x.x!
```

**Step timing with bridge active (images < 5 ms):**
```
[TIMING ms] pos1=4.1 move=1.0 images=0.3 collision=3.9 pos2=4.2 total=13.5 (~74.1 Hz)
[IMG-ROS2 ms] rgb_cache=0.1 depth_ros2=0.1
```

**Step timing without bridge (images ~600 ms):**
```
[TIMING ms] pos1=4.2 move=1.1 images=612.3 collision=3.8 pos2=4.0 total=625.4 (~1.6 Hz)
[IMG-API ms] simGetImages=609.7
```

If `depth_ros2` shows as `depth_api` in the output, the depth topic hasn't
received its first frame yet — this is normal for the first few steps and
resolves automatically.

---

## Troubleshooting

**`Connection refused` on AirSim Python API:**
- Check `AIRSIM_HOST` is set correctly and that AirSim is running
- Add a Windows Firewall inbound rule for TCP port 41451

**ROS2 bridge receives no frames:**
- Confirm the `airsim_node` launched without errors
- Confirm `host:=$AIRSIM_HOST` in the launch command matches the Windows IP
- Run `ros2 topic list` — if the topics don't appear, the bridge isn't connected to AirSim

**`ImportError: No module named 'rclpy'`:**
- Make sure you sourced ROS2 before running: `source /opt/ros/humble/setup.bash`
- Make sure you sourced the workspace: `source ~/AirSim/ros2/install/setup.bash`

**Low frame rate despite bridge (< 10 Hz):**
- Check GPU load on Windows — AirSim render rate drops under high GPU load
- Lower `Width`/`Height` in `settings.json` (the bridge renders at native
  resolution; downsampling to 84×84 happens in Python)

**`ClockSpeed` > 1.0 + ROS2 bridge = wrong observations:**
- Keep `ClockSpeed: 1.0` when using the bridge. The bridge publishes images
  at wall-clock render rate regardless of simulation speed, so a faster
  clock means physics advance faster than images update.
