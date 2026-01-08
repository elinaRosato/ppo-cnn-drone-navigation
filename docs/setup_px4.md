# PX4 Setup Guide for AirSim on Windows

## Step 1: Install WSL2 (Windows Subsystem for Linux)

Open PowerShell as **Administrator** and run:

```powershell
wsl --install -d Ubuntu-22.04
```

**Or** if WSL is already installed but you need Ubuntu:
```powershell
wsl --install -d Ubuntu-22.04
```

Restart your computer when prompted.

After restart, Ubuntu will launch automatically. Create a username and password when prompted.

---

## Step 2: Install PX4 Dependencies in WSL2

Open Ubuntu (search "Ubuntu" in Windows Start menu) and run these commands:

```bash
# Update package lists
sudo apt update
sudo apt upgrade -y

# Install required dependencies
sudo apt install -y \
    git \
    make \
    cmake \
    g++ \
    gcc \
    python3 \
    python3-pip \
    python3-jinja2 \
    python3-empy \
    ninja-build

# Install additional Python packages
pip3 install --user pyros-genmsg packaging toml numpy
```

---

## Step 3: Clone PX4 Autopilot

```bash
# Navigate to home directory
cd ~

# Clone PX4 (this will take a few minutes)
git clone https://github.com/PX4/PX4-Autopilot.git --recursive

# Navigate into the directory
cd PX4-Autopilot
```

---

## Step 4: Build PX4 for AirSim

```bash
# Make the setup script executable
chmod +x Tools/setup/ubuntu.sh

# Run the setup script (installs more dependencies)
./Tools/setup/ubuntu.sh

# Build PX4 SITL for AirSim (first build takes 10-15 minutes)
make px4_sitl_default none_iris
```

---

## Step 5: Running PX4 with AirSim

### Terminal 1 (WSL2 - Ubuntu): Start PX4

```bash
cd ~/PX4-Autopilot
make px4_sitl_default none_iris
```

Wait for the message: `INFO [mavlink] partner IP: 127.0.0.1`

### Terminal 2 (Windows): Start AirSim

Navigate to your AirSim blocks folder and double-click `Blocks.exe`

Or from command prompt:
```cmd
cd C:\Users\Dator\Documents\AirSim
start Blocks.exe
```

### Terminal 3 (Windows): Start Training

```cmd
cd C:\Users\Dator\Sites\uav-simulation-training
python train_ppo_v3_baby_steps.py
```

---

## AirSim settings.json for PX4

Make sure your `C:\Users\Dator\Documents\AirSim\settings.json` contains:

```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 5.0,
  "Vehicles": {
    "PX4": {
      "VehicleType": "PX4Multirotor",
      "UseSerial": false,
      "UseTcp": true,
      "TcpPort": 4560,
      "ControlPortLocal": 14540,
      "ControlPortRemote": 14580,
      "Enabled": true,
      "LocalHostIp": "127.0.0.1",
      "UdpIp": "127.0.0.1",
      "UdpPort": 14560,
      "Cameras": {
        "front_center": {
          "CaptureSettings": [
            {
              "ImageType": 0,
              "Width": 256,
              "Height": 144,
              "FOV_Degrees": 90
            }
          ],
          "X": 0.5,
          "Y": 0.0,
          "Z": 0.0,
          "Pitch": 0.0,
          "Roll": 0.0,
          "Yaw": 0.0
        }
      }
    }
  }
}
```

---

## Troubleshooting

### PX4 won't connect to AirSim
- Make sure PX4 is running BEFORE starting AirSim
- Check that ports 14540, 14580, 4560 are not blocked by firewall
- In WSL2, ensure localhost networking is working: `ping 127.0.0.1`

### "Permission denied" errors in WSL2
```bash
chmod +x Tools/setup/ubuntu.sh
```

### Build fails with "submodule" errors
```bash
cd ~/PX4-Autopilot
git submodule update --init --recursive
```

### WSL2 networking issues
```powershell
# In PowerShell as Administrator
wsl --shutdown
# Then restart Ubuntu
```

---

## Quick Start Commands (After Initial Setup)

Every time you want to train:

1. **Terminal 1 (WSL2):**
   ```bash
   cd ~/PX4-Autopilot && make px4_sitl_default none_iris
   ```

2. **Terminal 2 (Windows):** Start `Blocks.exe`

3. **Terminal 3 (Windows):**
   ```cmd
   cd C:\Users\Dator\Sites\uav-simulation-training
   python train_ppo_v3_baby_steps.py
   ```

---

## Advantages of PX4 over SimpleFlight

- **Realistic flight dynamics**: Same autopilot used in real drones
- **Real-world transfer**: Trained models work better on actual hardware
- **Advanced sensors**: Better IMU, GPS, barometer simulation
- **Industry standard**: Used by DJI, Auterion, and many drone companies
