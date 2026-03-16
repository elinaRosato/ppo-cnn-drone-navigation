# AirSim ROS2 Bridge Setup

This guide covers two configurations:

- **[Linux Native](#linux-native-ubuntu-2404--ros2-jazzy)** — Ubuntu 24.04, Cosys-AirSim running directly in Unreal Engine
- **[Windows + WSL2](#windows--wsl2-ubuntu-2204--ros2-humble)** — AirSim on Windows, ROS2 bridge and training in WSL2

## Why bother?

The AirSim Python API `simGetImages()` takes ~500 ms per call (TCP round-trip + render sync). This limits training to ~1.5 Hz — the 4-frame stack spans ~2.7 seconds. On a real drone at 30 Hz, the same 4 frames span ~133 ms. The model trained in simulation sees a completely different temporal signal than it will on the real drone.

The ROS2 bridge publishes frames at the Unreal Engine render rate (~20–30 Hz). The training loop reads from a local memory cache (< 1 ms) and the bottleneck disappears.

| Method      | Step rate | 4-frame span |
|-------------|-----------|--------------|
| Python API  | ~1.5 Hz   | ~2.7 s       |
| ROS2 bridge | ~30 Hz    | ~133 ms      |
| Real drone  | 30 Hz     | 133 ms       |

---

## Linux Native (Ubuntu 24.04 + ROS2 Jazzy)

### Step 1: Install ROS2 Jazzy

```bash
sudo apt update && sudo apt install -y software-properties-common curl

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

sudo mkdir -p /etc/apt/sources.list.d
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'

sudo apt update
sudo apt install -y ros-jazzy-desktop python3-colcon-common-extensions python3-rosdep
```

### Step 2: Install bridge dependencies

```bash
sudo apt install -y \
  ros-jazzy-cv-bridge \
  ros-jazzy-image-transport \
  ros-jazzy-tf2 \
  ros-jazzy-tf2-ros \
  ros-jazzy-tf2-geometry-msgs \
  ros-jazzy-tf2-sensor-msgs \
  ros-jazzy-tf2-eigen \
  ros-jazzy-mavros-msgs \
  ros-jazzy-geographic-msgs \
  ros-jazzy-pcl-ros \
  libpcl-dev
```

### Step 3: Build the Cosys-AirSim ROS2 packages

ROS2 Jazzy renamed several headers from `.h` to `.hpp`. The Cosys-AirSim source
uses the old names and must be patched before building.

```bash
# Patch header includes
sed -i \
  's|tf2_geometry_msgs/tf2_geometry_msgs.h|tf2_geometry_msgs/tf2_geometry_msgs.hpp|g; \
   s|tf2_sensor_msgs/tf2_sensor_msgs.h|tf2_sensor_msgs/tf2_sensor_msgs.hpp|g; \
   s|cv_bridge/cv_bridge.h|cv_bridge/cv_bridge.hpp|g' \
  ~/Cosys-AirSim/ros2/src/airsim_ros_pkgs/include/pd_position_controller_simple.h \
  ~/Cosys-AirSim/ros2/src/airsim_ros_pkgs/include/airsim_ros_wrapper.h \
  ~/Cosys-AirSim/ros2/src/airsim_ros_pkgs/include/utils.h \
  ~/Cosys-AirSim/ros2/src/airsim_ros_pkgs/src/airsim_ros_wrapper.cpp

# Initialise rosdep
sudo rosdep init
rosdep update

# Install ROS dependencies
source /opt/ros/jazzy/setup.bash
cd ~/Cosys-AirSim/ros2
rosdep install --from-paths src --ignore-src -r -y

# Build
colcon build
```

> **`rosdep` warning about `pcl`** is expected — install it manually (Step 2 above). Everything else resolves correctly.

> **`stderr output` in build summary** is warnings only, not errors. A successful build ends with `2 packages finished`.

### Step 4: Source ROS2 permanently

```bash
echo "source /opt/ros/jazzy/setup.bash" >> ~/.bashrc
echo "source ~/Cosys-AirSim/ros2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### Step 5: Configure AirSim settings

Copy `settings_sample.json` to `~/Documents/AirSim/settings.json`:

```bash
mkdir -p ~/Documents/AirSim
cp ~/Sites/ppo-cnn-drone-navigation/src/v2/settings_sample.json ~/Documents/AirSim/settings.json
```

> **`ClockSpeed` must be `1.0` when using the ROS2 bridge.** A higher value speeds up physics but the ROS2 topics still publish at real-time render rate, causing desynchronisation between physics and images.

### Step 6: Install Python dependencies

The venv must be created with `--system-site-packages` so it can access the
ROS2 Python packages (`rclpy`, `cv_bridge`) installed system-wide.

```bash
cd ~/Sites/ppo-cnn-drone-navigation
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 7: Launch

**Terminal 1 — Unreal Engine:** Open and hit Play in the Blocks environment.

**Terminal 2 — ROS2 AirSim node:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/Cosys-AirSim/ros2/install/setup.bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py host:=localhost
```

**Terminal 3 — Training:**
```bash
cd ~/Sites/ppo-cnn-drone-navigation
source venv/bin/activate
python3 src/v2/train.py --ros2
```

---

## Windows + WSL2 (Ubuntu 22.04 + ROS2 Humble)

### Step 1: Enable WSL2 on Windows

Open PowerShell as Administrator:

```powershell
wsl --install
wsl --set-default-version 2
```

Restart when prompted. After restart, open the Ubuntu terminal from the Start
menu and complete the initial user setup.

> **Forgotten password?**
> ```powershell
> wsl -d Ubuntu-22.04 -u root
> ```
> Then inside WSL2:
> ```bash
> passwd yourname
> exit
> ```

### Step 2: Install ROS2 Humble in WSL2

```bash
sudo apt update && sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg

sudo mkdir -p /etc/apt/sources.list.d
sudo sh -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list'

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
  ros-humble-tf2-eigen \
  ros-humble-pcl-ros \
  libpcl-dev

echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc

sudo rosdep init
rosdep update
```

### Step 3: Build Cosys-AirSim ROS2 packages in WSL2

```bash
sudo apt install -y build-essential cmake git

# Clone Cosys-AirSim (or copy from Windows if already cloned)
cd ~
git clone https://github.com/Cosys-Lab/Cosys-AirSim.git
cd Cosys-AirSim

# Build AirLib
./setup.sh
./build.sh

# Build ROS2 package
source /opt/ros/humble/setup.bash
cd ~/Cosys-AirSim/ros2
rosdep install --from-paths src --ignore-src -r -y
colcon build

echo "source ~/Cosys-AirSim/ros2/install/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

> **`build.sh` exits with Error 2** near the end — this is expected. The only file needed is `build_release/output/lib/libAirLib.a`. Verify it exists before proceeding.

### Step 4: Install Python dependencies in WSL2

The venv must be created with `--system-site-packages` to access ROS2 Python
packages (`rclpy`, `cv_bridge`) installed system-wide.

```bash
cd ~/Sites/ppo-cnn-drone-navigation   # or your project path
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 5: Configure AirSim settings on Windows

Copy `settings_sample.json` to `C:\Users\<YourName>\Documents\AirSim\settings.json`.

> **`ClockSpeed` must be `1.0` when using the ROS2 bridge.**

### Step 6: Find the Windows host IP

WSL2 runs in a virtual network. AirSim runs on Windows, so you need the host IP:

```bash
export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
echo $AIRSIM_HOST   # typically 172.x.x.x
```

Persist it:
```bash
echo 'export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '"'"'{print $2}'"'"')' >> ~/.bashrc
```

### Step 7: Windows Firewall rule (one-time)

Run in **PowerShell as Administrator**:

```powershell
New-NetFirewallRule -DisplayName "AirSim" -Direction Inbound -Protocol TCP -LocalPort 41451 -Action Allow
```

### Step 8: Launch

**Windows:** Start Unreal Engine and hit Play.

**WSL2 Terminal 1 — ROS2 AirSim node:**
```bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py host:=$AIRSIM_HOST output:=screen
```

**WSL2 Terminal 2 — Training:**
```bash
cd ~/Sites/ppo-cnn-drone-navigation
source venv/bin/activate
python3 src/v2/train.py --ros2
```

---

## Verifying It Works

**Bridge connected:**
```
[ROS2Bridge] Subscriber thread started → .../Scene + .../DepthPerspective
ROS2 bridge ready.
Connected to AirSim at localhost!
```

**Step timing with bridge — effective ~30 Hz:**
```
[TIMING ms] pos1=1.0 move=0.1 images=2.2 collision=0.5 pos2=0.8 compute=4.6 (~29.9 Hz)
[IMG-ROS2 ms] rgb_cache=0.0 depth_ros2=0.0
```

**Step timing without bridge (Python API fallback):**
```
[TIMING ms] pos1=4.2 move=1.1 images=612.3 collision=3.8 pos2=4.0 compute=625.4 (~1.6 Hz)
[IMG-API ms] simGetImages=609.7
```

---

## Troubleshooting

**`bind: Address already in use` when hitting Play:**
```bash
kill $(ss -tlnp | grep 41451 | awk '{print $6}' | grep -oP 'pid=\K[0-9]+')
```

**`ModuleNotFoundError: No module named 'rclpy'`:**
- The venv was created without `--system-site-packages`. Recreate it:
  ```bash
  rm -rf venv
  python3 -m venv --system-site-packages venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

**`ModuleNotFoundError: No module named 'backports'`:**
```bash
pip install backports.ssl-match-hostname
```

**`AttributeError: _ARRAY_API not found` in cv_bridge:**
- NumPy 2.x is installed; downgrade:
  ```bash
  pip install "numpy<2"
  ```

**`Connection refused` on AirSim Python API:**
- Confirm Play is active in Unreal Editor
- (Windows) Confirm the firewall rule for TCP 41451 exists
- (Windows) Confirm `AIRSIM_HOST` matches the Windows IP seen from WSL2

**ROS2 bridge receives no frames after 10s:**
- Confirm `airsim_node` launched without errors
- Run `ros2 topic list` — if topics don't appear, the node isn't connected to AirSim
- (Linux) Verify `host:=localhost` is correct

**`colcon build` fails: header not found:**
- On Jazzy, apply the header patches from Step 3 (Linux section)
- On Humble, this should not occur — check that all apt packages installed correctly

**Low frame rate despite bridge (< 10 Hz):**
- Check GPU load — AirSim render rate drops under high GPU utilisation
- Lower `Width`/`Height` in `settings.json`

**`ClockSpeed > 1.0` + ROS2 bridge = wrong observations:**
- Keep `ClockSpeed: 1.0`. Physics advance faster than images update at higher values.
