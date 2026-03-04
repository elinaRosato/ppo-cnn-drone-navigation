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
initial user setup (username + password). **Save this password** — you will
need it for `sudo` commands throughout this guide.

Verify WSL2 is the default version:

```powershell
wsl --set-default-version 2
wsl -l -v   # should show Ubuntu with VERSION 2
```

> **Forgotten password?** Log in as root without a password and reset it:
> ```powershell
> wsl -d Ubuntu-22.04 -u root
> ```
> Then inside WSL2:
> ```bash
> cat /etc/passwd | grep /home   # find your username
> passwd yourname                # set a new password
> exit
> ```

---

## Step 2: Install ROS2 Humble in WSL2

Open the Ubuntu WSL2 terminal and run:

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

# Install ROS2 Humble + all packages needed by airsim_ros_pkgs
sudo apt update
sudo apt install -y \
  ros-humble-desktop \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-cv-bridge \
  ros-humble-image-transport \
  ros-humble-mavros-msgs \
  ros-humble-geographic-msgs \
  ros-humble-tf2 \
  ros-humble-tf2-ros \
  ros-humble-tf2-geometry-msgs \
  ros-humble-tf2-sensor-msgs \
  ros-humble-tf2-eigen

# Source ROS2 in every new shell
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Initialise rosdep
sudo rosdep init
rosdep update
```

> **`rosdep install` warning about `message_runtime`** is harmless — it is a
> ROS1 package name left in AirSim's `package.xml`. Ignore it and continue.

---

## Step 3: Build AirSim from Source in WSL2

The ROS2 bridge (`airsim_ros_pkgs`) is only available by building from source.

### 3a. Install build dependencies and clone

```bash
sudo apt install -y build-essential cmake git

cd ~
git clone https://github.com/microsoft/AirSim.git
cd AirSim
```

### 3b. Patch setup.sh for Ubuntu 22.04

Two package names changed in Ubuntu 22.04 and are not available under their
old names:

```bash
# vulkan-utils was renamed to vulkan-tools
# clang-8 is not available; use clang-12
sudo sed -i \
  's/vulkan-utils/vulkan-tools/g;
   s/clang-8/clang-12/g;
   s/clang++-8/clang++-12/g;
   s/libc++-8-dev/libc++-12-dev/g;
   s/libc++abi-8-dev/libc++abi-12-dev/g' \
  ~/AirSim/setup.sh

./setup.sh
```

### 3c. Patch build.sh for Ubuntu 22.04

`build.sh` also hardcodes `clang-8`:

```bash
sed -i 's/clang-8/clang-12/g; s/clang++-8/clang++-12/g' ~/AirSim/build.sh
./build.sh
```

> **`build.sh` exits with Error 2** near the end (after 82% — non-AirLib
> targets fail to compile with clang-12). This is fine. The only file the
> ROS2 package needs is `build_release/output/lib/libAirLib.a`, which is
> built successfully. Verify it exists:
> ```bash
> ls -lh ~/AirSim/build_release/output/lib/libAirLib.a
> # should show ~3-4 MB
> ```

### 3d. Patch the ROS2 CMakeLists.txt for ROS2 Humble include layout

ROS2 Humble changed the include directory layout: each package now has its
own subdirectory under `/opt/ros/humble/include/`. The AirSim ROS2 package
was written for Foxy and does not account for this.

```bash
python3 -c "
content = open('/home/$USER/AirSim/ros2/src/airsim_ros_pkgs/CMakeLists.txt').read()
content = content.replace(
    'include_directories(\${INCLUDE_DIRS})',
    '# ROS2 Humble: each package has its own include subdir\nfile(GLOB _ros2_includes /opt/ros/humble/include/*)\ninclude_directories(\${INCLUDE_DIRS} \${_ros2_includes})'
)
open('/home/$USER/AirSim/ros2/src/airsim_ros_pkgs/CMakeLists.txt', 'w').write(content)
print('Patched')
"
```

> **Important:** replace `$USER` with your actual Linux username if the
> Python one-liner doesn't expand it automatically (e.g. `/home/elina/...`).

### 3e. Build the ROS2 package

```bash
cd ~/AirSim/ros2
source /opt/ros/humble/setup.bash

rosdep install --from-paths src --ignore-src -r -y

colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the workspace
echo "source ~/AirSim/ros2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

The build will print many `-Wdeprecated-copy` warnings from Eigen — these are
harmless. A successful build ends with:
```
Summary: 2 packages finished [~2 min]
  1 package had stderr output: airsim_ros_pkgs
```
(stderr output = warnings only, not errors)

---

## Step 4: Install Python Dependencies in WSL2

> **NumPy version constraint:** `cv_bridge` (installed by apt as part of
> ROS2 Humble) was compiled against NumPy 1.x. Installing NumPy 2.x via pip
> will cause an `AttributeError: _ARRAY_API not found` at runtime.

```bash
# Install msgpack-rpc-python first — airsim's setup.py requires it at
# metadata-generation time, so it must be present before airsim is installed
pip install msgpack-rpc-python

# Install remaining deps; pin numpy below 2.0
pip install \
  airsim \
  gymnasium \
  "stable-baselines3[extra]" \
  opencv-python \
  "numpy<2" \
  torch torchvision
```

> **`numpy<2` vs opencv-python 4.13 conflict warning** is expected and
> harmless. opencv-python 4.13 declares `numpy>=2` as a requirement but
> works correctly for all operations used in this project.

> **`python` not found?** On Ubuntu 22.04, use `python3` instead of `python`
> for all commands.

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
export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $AIRSIM_HOST   # typically 172.x.x.x
```

Add it to your `~/.bashrc` so you don't have to set it every session:

```bash
echo 'export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '"'"'{print $2}'"'"')' >> ~/.bashrc
source ~/.bashrc
```

---

## Step 7: Windows Firewall Rule (one-time)

AirSim listens on TCP port 41451. WSL2 connections from Linux will be blocked
unless an inbound firewall rule is added. Run this in **PowerShell as
Administrator** on Windows:

```powershell
New-NetFirewallRule -DisplayName "AirSim" -Direction Inbound -Protocol TCP -LocalPort 41451 -Action Allow
```

---

## Step 8: Launch AirSim on Windows

Start your Unreal Engine project on the Windows side. **Hit Play** in the
editor before proceeding — AirSim only starts accepting connections once the
simulation is running.

---

## Step 9: Launch the ROS2 Bridge in WSL2

In a WSL2 terminal (terminal 1):

```bash
source /opt/ros/humble/setup.bash
source ~/AirSim/ros2/install/setup.bash

ros2 launch airsim_ros_pkgs airsim_node.launch.py \
  output:=screen \
  host:=$AIRSIM_HOST
```

You should see:
```
[airsim_node-2] Connected!
[airsim_node-2] [INFO] [...] [airsim_node]: AirsimROSWrapper Initialized!
```

---

## Step 10: Run Training in WSL2

Open a second WSL2 terminal (terminal 2):

```bash
# Windows project files are accessible under /mnt/c/
cd /mnt/c/Users/<YourName>/Sites/uav-simulation-training/src/v2

# Make sure AIRSIM_HOST is set
export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')

# Start training with the ROS2 bridge
python3 train.py --ros2

# Optional: specify total steps
python3 train.py --ros2 --steps 500000

# Resume a previous run with the bridge
python3 train.py --ros2 --resume
```

---

## Verifying It Works

**Bridge connected:**
```
[ROS2Bridge] Subscriber thread started → .../Scene + .../DepthPerspective
ROS2 bridge ready.
Connected to AirSim at 172.x.x.x!
```

**Step timing with bridge active — `compute` < 10 ms, effective ~30 Hz:**
```
[TIMING ms] pos1=1.0 move=0.1 images=2.2 collision=0.5 pos2=0.8 compute=4.6 (~29.9 Hz)
[IMG-ROS2 ms] rgb_cache=0.0 depth_ros2=0.0
```

The `compute=` field shows the actual processing time before the 30 Hz rate
limiter sleep. The Hz shown is the effective wall-clock step rate after
sleeping.

**Step timing without bridge (Python API fallback):**
```
[TIMING ms] pos1=4.2 move=1.1 images=612.3 collision=3.8 pos2=4.0 compute=625.4 (~1.6 Hz)
[IMG-API ms] simGetImages=609.7
```

If `depth_ros2` shows as `depth_api` in the first few steps, the depth topic
hasn't received its first frame yet — this resolves automatically.

---

## Troubleshooting

**`Connection refused` on AirSim Python API:**
- Check `AIRSIM_HOST` is set correctly and that AirSim is running
- Confirm the Windows Firewall rule for TCP port 41451 exists
- Confirm Play is active in the Unreal Editor

**ROS2 bridge receives no frames:**
- Confirm the `airsim_node` launched without errors
- Confirm `host:=$AIRSIM_HOST` in the launch command matches the Windows IP
- Run `ros2 topic list` — if the topics don't appear, the bridge isn't connected to AirSim

**`AttributeError: _ARRAY_API not found` in cv_bridge:**
- NumPy 2.x is installed; downgrade: `pip install "numpy<2"`

**`ModuleNotFoundError: No module named 'msgpackrpc'`:**
- Install before airsim: `pip install msgpack-rpc-python`

**`ImportError: No module named 'rclpy'`:**
- Source ROS2 before running: `source /opt/ros/humble/setup.bash`
- Source the workspace: `source ~/AirSim/ros2/install/setup.bash`

**`build.sh` fails with `clang-8 not found`:**
- Patch the compiler references: `sed -i 's/clang-8/clang-12/g; s/clang++-8/clang++-12/g' ~/AirSim/build.sh`

**`colcon build` fails with missing tf2/mavros headers:**
- The CMakeLists.txt patch in Step 3d was not applied or was applied incorrectly
- Verify the file contains `file(GLOB _ros2_includes /opt/ros/humble/include/*)` before `include_directories`

**Low frame rate despite bridge (< 10 Hz):**
- Check GPU load on Windows — AirSim render rate drops under high GPU load
- Lower `Width`/`Height` in `settings.json` (the bridge renders at native
  resolution; downsampling to 84×84 happens in Python)

**`ClockSpeed` > 1.0 + ROS2 bridge = wrong observations:**
- Keep `ClockSpeed: 1.0` when using the bridge. The bridge publishes images
  at wall-clock render rate regardless of simulation speed, so a faster
  clock means physics advance faster than images update.
