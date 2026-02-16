# UAV Obstacle Avoidance Training

Reinforcement learning system that teaches a drone to avoid obstacles using only a depth camera. A simple controller handles navigation (fly toward goal), while the RL model learns lateral and vertical corrections to dodge obstacles along the way.

## Architecture

```
Controller (hardcoded):  yaw toward goal, fly forward at base_speed
RL Model (learned):      depth image -> lateral + vertical correction
Combined output:         body_velocity = (forward, lateral, altitude_hold + vertical)
```

The model receives a single-channel depth image and outputs two continuous values:
- **Lateral correction** (-1 to 1): steer left/right in body frame
- **Vertical correction** (-1 to 1): adjust altitude up/down

Navigation is not learned -- the controller always flies toward the goal. The model only learns to avoid what it sees in the camera.

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
      "ImageType": 2,
      "Width": 84,
      "Height": 84,
      "FOV_Degrees": 145
    }
  }
}
```

- **ImageType 2**: Depth perspective (float distance in meters)
- **84x84**: Matches CnnPolicy input, fast to capture
- **145 FOV**: Wide field of view for better obstacle detection
- **ClockSpeed 10**: Simulation runs 10x faster than real-time

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
# Live camera feed (run alongside training)
python view_camera.py

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
| `base_speed` | 1.0 m/s | Constant forward speed (body frame) |
| `lateral_scale` | 1.0 | Max lateral correction velocity |
| `vertical_scale` | 0.5 | Max vertical correction velocity |
| `cruising_altitude` | -5.0 m | Target altitude (NED, negative = up) |
| `goal_distance_range` | (50, 50) | Goal distance from origin in meters |
| `goal_radius` | 1.5 m | Distance to consider goal reached |
| `max_steps` | 500 | Steps before episode truncation |

### PPO (Stable-Baselines3)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy` | CnnPolicy | Convolutional policy for image input |
| `learning_rate` | 3e-4 | Adam optimizer learning rate |
| `n_steps` | 2048 | Steps per rollout buffer |
| `batch_size` | 64 | Minibatch size for updates |
| `n_epochs` | 10 | Passes over rollout data per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE lambda for advantage estimation |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.01 | Entropy coefficient (exploration) |
| `vf_coef` | 0.5 | Value function loss coefficient |
| `max_grad_norm` | 0.5 | Gradient clipping |

### Camera

| Parameter | Value | Description |
|-----------|-------|-------------|
| Resolution | 84x84 | Single-channel depth image |
| FOV | 145 degrees | Wide angle for obstacle detection |
| Depth range | 0-100 m | Clamped and normalized to 0-255 uint8 |
| Position | X=0.25, Z=-0.1 | Front-center, slightly above body |

## Reward Function

The reward is intentionally simple -- the model should learn to stay out of trouble:

- **Collision**: -25.0 per collision
- **Action penalty**: -2.0 * ||action|| (penalizes unnecessary corrections)

No progress reward. The controller handles navigation; the model is only rewarded for flying smoothly without hitting things.

## Movement System

Movement uses **body frame** (`moveByVelocityBodyFrameAsync`):
- **X (forward)**: always `base_speed` (1.0 m/s)
- **Y (lateral)**: model's lateral correction * `lateral_scale`
- **Z (vertical)**: altitude P-controller + model's vertical correction

The controller yaws the drone to face the goal, so body-frame forward always points toward the target. The velocity command uses a long duration (5s) without `.join()` so the drone continues flying during camera capture -- the next step's command overrides the previous one.

### Altitude Hold

A P-controller maintains cruising altitude:
```
altitude_error = cruising_altitude - current_z
base_vz = clip(altitude_error * 0.5, -1.0, 1.0)
```
The model can add vertical corrections on top, but is clamped when too far below cruising altitude to prevent ground crashes.

## File Structure

```
src/simplified/
  avoidance_env.py      # Gymnasium environment (core training logic)
  train.py              # Training script with checkpoints and eval
  test.py               # Test trained model with visual markers
  fly_mission.py        # Multi-waypoint mission with trained model
  fly_baseline.py       # Baseline: controller only, no model
  view_camera.py        # Live depth camera feed viewer
  monitor_drones.py     # Visual drone/goal marker overlay
  settings_sample.json  # AirSim configuration
```

### Output Directories

```
models_simplified/
  run_YYYY-MM-DD_HH-MM-SS/
    checkpoints/                    # Every 20k steps
    best_model/                     # Best eval performance
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
