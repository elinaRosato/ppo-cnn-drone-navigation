# UAV Obstacle Avoidance Training

Reinforcement learning system that teaches a drone to avoid obstacles using a front-facing RGB camera. A simple controller handles navigation (fly toward goal), while the RL model learns lateral and vertical corrections to dodge obstacles along the way.

## Architecture

```
Controller (hardcoded):  compute world-frame velocity toward goal
RL Model (learned):      4 stacked grayscale frames → lateral + vertical correction
Combined output:         world velocity = forward_toward_goal + lateral_perpendicular
Yaw:                     faces actual movement direction (camera always looks ahead)
```

The model receives 4 stacked grayscale frames (converted from RGB) and outputs two continuous values:
- **Lateral correction** (-1 to 1): steer left/right perpendicular to the goal direction
- **Vertical correction** (-1 to 1): adjust altitude up/down

Navigation is not learned — the controller always flies toward the goal. The model only learns to avoid what it sees in the camera. Frame stacking gives the model temporal context so it can detect motion and approaching obstacles.

## Quick Start

### 1. Setup

```bash
git clone https://github.com/yourusername/uav-simulation-training.git
cd uav-simulation-training
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure AirSim

Copy the settings file to your AirSim config directory:

```bash
cp src/simplified/settings_sample.json C:/Users/<YourUser>/Documents/AirSim/settings.json
```

Key settings:
```json
{
  "ClockSpeed": 10.0,
  "Cameras": {
    "front_center": {
      "CaptureSettings": [
        { "ImageType": 0, "Width": 84, "Height": 84, "FOV_Degrees": 145 },
        { "ImageType": 2, "Width": 84, "Height": 84, "FOV_Degrees": 145 }
      ]
    }
  }
}
```

- **ImageType 0**: Scene (RGB) — used as the model observation (converted to grayscale)
- **ImageType 2**: Depth perspective — used only for the soft proximity penalty reward, not fed to the model
- **84x84**: Matches CnnPolicy input, fast to capture
- **145 FOV**: Wide field of view for better obstacle detection
- **ClockSpeed 10**: Simulation runs 10x faster than real-time

Both image types must be listed in `CaptureSettings` so AirSim pre-allocates the buffers. Both are requested in a single `simGetImages` call per step.

### 3. Train

```bash
cd src/simplified
python train.py                     # New training, 200k steps
python train.py --steps 500000      # New training, 500k steps
python train.py --resume            # Resume from latest checkpoint
python train.py --resume --steps 400000  # Resume, train to 400k total
```

### 4. Test

```bash
python test.py                          # Test latest model, 5 episodes
python test.py --episodes 10            # More episodes
python test.py --model path/to/model.zip  # Specific model
```

### 5. Fly a Mission

```bash
python fly_mission.py                         # Latest model, default waypoints
python fly_mission.py --model path/to/model.zip --speed 3.0
```

### 6. Monitor Training

```bash
# Live camera feed — shows the grayscale RGB feed the model sees
python view_camera.py
python view_camera.py --stack    # Show all 4 stacked frames side-by-side

# Drone/goal marker visualization
python monitor_drones.py

# TensorBoard graphs
tensorboard --logdir ./logs_simplified
# Open http://localhost:6006
```

## Hyperparameters

### Environment

| Parameter | Value | Description |
|-----------|-------|-------------|
| `base_speed` | 1.0 m/s | Constant forward speed toward goal |
| `lateral_scale` | 1.0 | Max lateral correction velocity (m/s) |
| `vertical_scale` | 0.5 | Max vertical correction velocity (m/s) |
| `cruising_altitude` | -5.0 m | Target altitude (NED, negative = up) |
| `goal_distance_range` | (50, 50) | Goal distance from origin in meters |
| `goal_radius` | 1.5 m | Distance to consider goal reached |
| `max_steps` | 500 | Steps before episode truncation |
| `prox_threshold` | 5.0 m | Depth distance that triggers proximity penalty |
| `stack_frames` | 4 | Number of consecutive frames stacked as observation |
| `action_momentum` | 0.5 | Blend ratio with previous action (smoothing) |

### PPO (Stable-Baselines3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy` | CnnPolicy | Convolutional policy for image input |
| `learning_rate` | 1e-4 | Adam optimizer learning rate |
| `n_steps` | 2048 | Steps per rollout buffer |
| `batch_size` | 256 | Minibatch size for gradient updates |
| `n_epochs` | 5 | Passes over rollout data per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.01 | Entropy coefficient (exploration) |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

`batch_size=256` with `n_epochs=5` gives 40 gradient updates per rollout (2048/256 × 5), compared to 320 with the previous 64/10 combination. Larger batches reduce gradient variance for CNN policies.

### Camera / Observation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Source | RGB Scene (ImageType 0) | Captured as colour, converted to grayscale |
| Resolution | 84×84 | Per frame, matches CnnPolicy input |
| Channels | 4 stacked frames | Observation shape: (4, 84, 84) CHW |
| FOV | 145 degrees | Wide angle for obstacle detection |
| Camera position | X=0.25, Z=-0.1 | Front-center, slightly above body |
| Depth (reward only) | ImageType 2, raw metres | Used for proximity penalty, not fed to model |

The observation format `(4, 84, 84)` is channels-first (CHW). SB3's `CnnPolicy` detects this automatically and does not apply any transposition — no `VecTransposeImage` wrapper is needed.

## Reward Function

| Event | Reward | Notes |
|-------|--------|-------|
| Goal reached | +10.0 | Terminal — episode ends |
| Collision | -10.0 | Terminal — episode ends immediately |
| Proximity penalty | 0 to -2.0 | Scales with closeness: `-(threshold - depth) / threshold * 2` |
| Action norm | -0.05 × \|\|action\|\| | Discourages unnecessary corrections |

The proximity penalty uses the minimum depth in the centre 50% of the depth image. It activates when any obstacle is within `prox_threshold` (5 m) and creates a smooth gradient pushing the drone to stay clear, rather than only penalising at the hard collision boundary. Depth is queried in the same `simGetImages` call as RGB — no extra API overhead.

Action smoothing blends each output 50% with the previous action before execution, preventing abrupt velocity reversals and stabilising AirSim physics interactions.

## Movement System

The controller computes a world-frame velocity rather than yawing and flying in body frame:

```
ux, uy = unit vector toward goal
px, py = perpendicular (left) to goal direction

vel_x = base_speed * ux + lateral_correction * px
vel_y = base_speed * uy + lateral_correction * py

yaw = atan2(vel_y, vel_x)   # drone faces its actual movement direction
body_vx = sqrt(vel_x² + vel_y²)
body_vy = 0
```

This ensures the camera always faces the direction of travel regardless of whether the model is applying a correction, so what the model sees is always what is directly ahead of it.

### Altitude Hold

A P-controller maintains cruising altitude:
```
altitude_error = cruising_altitude - current_z
base_vz = clip(altitude_error * 0.5, -1.0, 1.0)
body_vz = base_vz + vertical_correction
```

The model can add vertical corrections on top, but is clamped when already below cruising altitude to prevent ground crashes.

## File Structure

```
src/simplified/
  avoidance_env.py      # Gymnasium environment (core training logic)
  train.py              # Training script with checkpoints and eval
  test.py               # Test trained model with visual markers
  fly_mission.py        # Multi-waypoint mission with trained model
  fly_baseline.py       # Baseline: controller only, no model
  view_camera.py        # Live grayscale RGB feed (with --stack option)
  monitor_drones.py     # Visual drone/goal marker overlay
  settings_sample.json  # AirSim configuration (RGB + Depth capture)
```

### Output Directories

```
models_simplified/
  run_YYYY-MM-DD_HH-MM-SS/
    checkpoints/                    # Saved every 20k steps
    best_model/                     # Best evaluation performance
    eval/                           # Eval logs
    simplified_avoidance_final.zip  # Final model

logs_simplified/
  run_YYYY-MM-DD_HH-MM-SS/
    tensorboard/                    # TensorBoard logs
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- AirSim (with Unreal Engine environment)
- CUDA GPU (recommended)
