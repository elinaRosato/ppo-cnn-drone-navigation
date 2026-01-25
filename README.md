# UAV Navigation Training with Reinforcement Learning

Deep reinforcement learning system for autonomous drone navigation using PPO and AirSim. The model learns to navigate to goal positions while avoiding obstacles and adapting to different height constraints.

## Features

- **Goal-Aware Navigation**: Navigates to any target position, not memorized routes
- **Height-Aware Adaptation**: Adapts to different altitude constraints (different forests/environments)
- **Obstacle Avoidance**: Uses camera vision for real-time obstacle detection
- **Altitude Maintenance**: Maintains consistent cruising altitude for efficient flight
- **Transfer Learning**: Trained in simulation, deployable to different environments

## SimpleFlight vs PX4

This project supports two flight modes:

**SimpleFlight (Recommended for Training):**
- AirSim's built-in flight controller
- Faster training iterations
- Simpler setup (no WSL/Ubuntu needed)
- Use `settings_simpleFlight_sample.json`

**PX4 (Recommended for Testing):**
- Realistic flight controller physics
- Validates real-world readiness
- Tests with GPS, sensor noise, and failsafes
- Use `settings_px4_sample.json`
- Requires WSL2/Ubuntu on Windows

**Workflow:** Train with SimpleFlight → Test with PX4 → Deploy to real drone

## Quick Start

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/uav-simulation-training.git
cd uav-simulation-training
```

2. **Create virtual environment**:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Training

**Training uses SimpleFlight** (AirSim's built-in flight controller) for faster iteration and simpler setup.

1. **Configure AirSim** (`C:\Users\Dator\Documents\AirSim\settings.json`):

Copy `settings_simpleFlight_sample.json` from this repository to your AirSim settings folder:

```bash
cp settings_simpleFlight_sample.json C:/Users/Dator/Documents/AirSim/settings.json
```

Or use this minimal configuration:
```json
{
  "SettingsVersion": 1.2,
  "SimMode": "Multirotor",
  "ClockSpeed": 5.0
}
```

2. **Launch AirSim**: Open Unreal Editor and hit Play

3. **Activate Python environment and start training**:
```bash
cd C:\Users\Dator\Sites\uav-simulation-training
venv\Scripts\activate
python train_ppo_v3_baby_steps.py
```

Training time: 6-10 hours for 500k timesteps with ClockSpeed=5

**Note:** Uses ultra-easy curriculum (3m → 35m goals over 300k steps) for faster initial learning.

## Staged Training System

The project includes a progressive staged training system located in `src/stage1/`, `src/stage2/`, `src/stage3/`, and `src/stage4/`. Each stage builds upon the previous one, adding complexity incrementally.

### Training Stages Overview

| Stage | Script | Goal | Default Steps |
|-------|--------|------|---------------|
| 1 | `train_stage1_forward.py` | Learn forward movement | 50,000 |
| 2 | `train_stage2_goals.py` | Add goal seeking | 200,000 |
| 3 | `train_stage3_altitude.py` | Add altitude constraints | 250,000 |
| 4 | `train_stage4_obstacles.py` | Add obstacle avoidance | 300,000 |

### How Staged Training Works

Each stage:
1. Loads the final model from the previous stage (if available)
2. Trains on a new environment with added complexity
3. Saves checkpoints every N steps
4. Creates timestamped run folders to preserve all training attempts

### Realistic Training (No Visual Markers)

By default, goal markers are **hidden** during training. This ensures the drone learns to navigate using **coordinate inputs** (like GPS), not by visually seeing a target marker.

The drone receives goal information through the observation vector:
- Relative position to goal (X, Y)
- Distance to goal
- Direction to goal (yaw angle)
- Relative yaw (how much to turn to face goal)

To enable visual markers for **debugging or testing**:
```python
# In the environment initialization
env = AirSimStage2Env(show_visual_marker=True)
```

### Running Training Scripts

**Start a new training run:**
```bash
cd src/stage1
python train_stage1_forward.py
```

**Resume from the latest checkpoint:**
```bash
python train_stage1_forward.py --resume
```

**Train for a specific number of total steps:**
```bash
python train_stage1_forward.py --steps 100000
```

**Combine flags (resume and train to 100k total):**
```bash
python train_stage1_forward.py --resume --steps 100000
```

### Command Line Flags

| Flag | Description | Example |
|------|-------------|---------|
| `--resume` | Resume from the latest checkpoint in the most recent run | `--resume` |
| `--steps N` | Set target total timesteps (overrides default) | `--steps 100000` |

### Folder Structure

Each training run creates timestamped folders to preserve all training attempts:

```
models_stage1/
  run_2026-01-10_14-30-00/       # First training attempt
    checkpoints/
      stage1_forward_10000_steps.zip
      stage1_forward_20000_steps.zip
      ...
    stage1_forward_final.zip
  run_2026-01-11_09-15-22/       # Second training attempt
    checkpoints/
      ...
    stage1_forward_final.zip

logs_stage1/
  run_2026-01-10_14-30-00/
    tensorboard/
      PPO_1/
        events.out.tfevents...
```

### Resume Behavior

When using `--resume`:
- Finds the **latest run folder** (most recent timestamp)
- Loads the **highest checkpoint** from that run
- Continues training from that step count
- Saves new checkpoints to the **same run folder**

**Example:**
```
# First run - trains from 0 to 20k, then interrupted
python train_stage1_forward.py

# Resume - continues from 20k checkpoint
python train_stage1_forward.py --resume

Output:
  Resuming from checkpoint: ./models_stage1/run_2026-01-10_14-30-00/checkpoints/stage1_forward_20000_steps.zip
  Resuming from step: 20,000
  Run directory: ./models_stage1/run_2026-01-10_14-30-00

  Target timesteps: 50,000
  Current progress: 20,000
  Remaining steps: 30,000
```

### Extending Training Beyond Default Steps

To train beyond the default target:
```bash
# Stage 1 defaults to 50k, train to 100k instead
python train_stage1_forward.py --steps 100000

# Or resume and extend
python train_stage1_forward.py --resume --steps 100000
```

### Stage Progression

After completing each stage, move to the next:
```bash
# Complete Stage 1
cd src/stage1
python train_stage1_forward.py

# Move to Stage 2 (automatically loads Stage 1 model)
cd ../stage2
python train_stage2_goals.py

# Move to Stage 3 (automatically loads Stage 2 model)
cd ../stage3
python train_stage3_altitude.py
```

Each stage script automatically:
1. Looks for the final model from the previous stage in the latest run folder
2. Falls back to the old path structure if not found
3. Creates a new model from scratch if no previous stage model exists (with a warning)

### Testing

**Quick Test with SimpleFlight:**
```bash
python test_ppo_v3.py --model ./models_v3_baby_steps/best_model/best_model.zip
```

**Testing with PX4 (Optional - For realistic validation):**

After training, you can test your model with PX4 for more realistic flight physics:

1. **Install WSL2 and Ubuntu** (Windows only):
```bash
wsl --install
```

2. **Install PX4-Autopilot in Ubuntu**:
```bash
# In WSL/Ubuntu terminal
git clone https://github.com/PX4/PX4-Autopilot.git --recursive
cd PX4-Autopilot
bash ./Tools/setup/ubuntu.sh
```

3. **Get your Windows host IP from WSL**:
```bash
ip route show | grep -i default | awk '{ print $3}'
# Example output: 172.22.32.1
```

4. **Configure AirSim for PX4**:

Copy `settings_px4_sample.json` to your AirSim settings folder:
```bash
cp settings_px4_sample.json C:/Users/Dator/Documents/AirSim/settings.json
```

**Note**: Update `LocalHostIp: "0.0.0.0"` in the settings to allow WSL to connect to AirSim.

5. **Start PX4** (replace with your Windows IP from step 3):
```bash
cd ~/PX4-Autopilot
PX4_SIM_HOSTNAME=172.22.32.1 COM_ARM_WO_GPS=1 make px4_sitl_default none_iris
```

6. **Launch AirSim**: Open Unreal Editor and hit Play

7. **Run your test** (modify test script to use `use_px4=True`)

**Why test with PX4?**
- Validates model works with realistic flight controller
- Tests with GPS, sensor noise, and realistic physics
- Proves readiness for real-world deployment

## Project Structure

```
uav-simulation-training/
├── airsim_env_v3.py              # Gym environment with height awareness
├── train_ppo_v3_baby_steps.py   # Ultra-easy curriculum (3m→35m) - RECOMMENDED
├── train_ppo_v3_curriculum.py   # Standard curriculum (10m→30m)
├── rescue_training.py            # Rescue failed training with baby steps
├── resume_training_v3.py         # Resume from checkpoint
├── test_ppo_v3.py               # Test trained model
├── main.py                      # Basic AirSim demo
├── requirements.txt             # Python dependencies
├── docs/                        # Documentation
│   ├── TRAINING_EVOLUTION.md    # Development history
│   ├── VERSION_COMPARISON.md    # V1 vs V2 vs V3
│   └── ...
└── archive/                     # Previous versions
    ├── v1/                      # Camera-only (deprecated)
    └── v2/                      # Goal-aware (limited)
```

## Model Architecture

- **Policy**: PPO with MultiInputPolicy
- **Observation**:
  - Camera image: 84x84x3 RGB
  - State vector: 13 values (goal position, orientation, height bounds, distances)
- **Network**: 256x256 dual-layer architecture
- **Training**: 500k timesteps with randomized goals and height constraints

## Observation Space

The model receives:
- **Image**: Front camera view (84x84 RGB)
- **Vector** (13 values):
  - [0-2]: Relative position to goal (x, y, z)
  - [3]: Distance to goal
  - [4]: Yaw angle to goal (world frame)
  - [5]: Current height
  - [6]: Goal height
  - [7-8]: Height bounds (max/min safe altitude)
  - [9-10]: Distance to ceiling/floor
  - [11]: Current drone yaw orientation
  - [12]: Relative yaw (how much to turn to face goal)

## Action Space

The drone can perform 4 continuous actions:
- **[0]**: X velocity (-1 to 1) → ±3.0 m/s
- **[1]**: Y velocity (-1 to 1) → ±3.0 m/s
- **[2]**: Z velocity (-1 to 1) → ±0.3 m/s (slower for safety)
- **[3]**: Yaw rate (-1 to 1) → ±45°/s (rotation)

This allows the drone to move in any direction AND rotate to point its camera toward the goal.

## Key Parameters

### Training Configuration
```python
# Environment
# Drone always starts at origin (0, 0) at optimal altitude
# Goals randomized at curriculum-controlled distance
curriculum_start_distance = 3.0m    # Ultra-easy start
curriculum_end_distance = 35.0m     # Full difficulty
curriculum_timesteps = 300000       # Gradual progression

# Height bounds (randomized per episode, provides wiggle room)
max_height = (-1.0, -1.2)    # Ceiling varies
min_height = (-3.5, -3.7)    # Floor varies (increased for larger drones)
# Total vertical space: ~2.5m (allows natural drone movement and wobbling)
# Extreme ceiling: ~-13.6m (terminates if exceeded - 5x range above ceiling)

# Visual aids
goal_radius = 3.0m           # Success zone (3D distance)
# Flight corridor: green (center), cyan (ceiling), orange (floor)

# PPO hyperparameters
n_steps = 2048
batch_size = 64
learning_rate = 3e-4
gamma = 0.99
ent_coef = 0.1              # High exploration (10x default)
```

### Reward Structure
**Progress Rewards (ONLY when in bounds):**
- 3D progress toward goal: +20.0 per meter
- Horizontal (XY) progress: +10.0 per meter
- Forward velocity bonus: +5.0 per m/s (moving where camera sees)
- Backward velocity penalty: -10.0 per m/s (flying blind)
- Alignment bonus: +3.0 (max when facing goal, encourages camera use)

**Altitude Rewards:**
- In-bounds survival: +0.5 per step
- Optimal altitude bonus: +5.0 (quadratic, peaked at center)
- Altitude progress: +10.0 per meter (directional - rewards moving toward target altitude)
- Out-of-bounds penalty: Exponential 50.0 * (e^(2x) - 1)

**Terminal Rewards:**
- Goal reached: +500.0
- Collision: -500.0
- Extreme altitude: -300.0 (flew too high)
- Step penalty: -0.1 (efficiency incentive)

**Behavioral Controls:**
- First 10 steps: Forced hover (vz=0) for stabilization, rotation allowed
- Vertical velocity: Limited to ±0.3 m/s (prevents instant crashes)
- Horizontal velocity: Full ±3.0 m/s (efficient navigation)
- Yaw rate: ±45°/s (moderate rotation speed)

**Design Note:** Progress rewards are ONLY given when within altitude bounds, eliminating out-of-bounds exploitation. The alignment bonus encourages rotating to face the goal, making camera-based navigation essential. Directional altitude rewards provide clear gradients for returning to target altitude when out of bounds.

## Real-World Deployment

```python
from stable_baselines3 import PPO

# Load model
model = PPO.load("models_v3_quality/best_model/best_model.zip")

# Get sensor data
camera_image = drone.get_camera()
position = gps.get_position()
goal = waypoint_planner.get_next()

# Get height bounds from terrain map
terrain_map = load_map("forest_region.map")
ceiling = terrain_map.get_canopy_height(position)
floor = terrain_map.get_ground_height(position)

# Build observation
observation = build_observation(camera_image, position, goal, ceiling, floor)

# Get action
action = model.predict(observation)
drone.set_velocity(action)
```

## Documentation

- **[TRAINING_EVOLUTION.md](docs/TRAINING_EVOLUTION.md)**: Complete development history
- **[VERSION_COMPARISON.md](docs/VERSION_COMPARISON.md)**: Comparison of V1, V2, and V3
- **[SPEED_OPTIMIZATION.md](docs/SPEED_OPTIMIZATION.md)**: Training speed techniques
- **[GOAL_AWARE_EXPLANATION.md](docs/GOAL_AWARE_EXPLANATION.md)**: Why goal awareness matters

## Resume Training

If training is interrupted:

```bash
# Auto-detect latest checkpoint
python resume_training_v3.py

# Or specify checkpoint
python resume_training_v3.py --checkpoint ./models_v3_quality/checkpoints/ppo_quality_100000_steps.zip
```

## Monitoring

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir=./logs_v3_quality/tensorboard/
```

Open http://localhost:6006

Key metrics:
- `rollout/ep_rew_mean`: Average episode reward (should increase)
- `rollout/ep_len_mean`: Episode length
- `time/fps`: Training speed

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- AirSim
- CUDA-capable GPU (recommended)

## License

MIT License

## Citation

If you use this code in your research, please cite:

```
@software{uav_navigation_training,
  title = {UAV Navigation Training with Reinforcement Learning},
  year = {2025},
  url = {https://github.com/yourusername/uav-simulation-training}
}
```
