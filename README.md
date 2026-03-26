# UAV Obstacle Avoidance Training

Reinforcement learning system that teaches a drone to avoid obstacles using a front-facing RGB camera. A simple controller handles navigation (fly toward goal), while the RL model learns lateral and vertical corrections to dodge obstacles along the way.

> **Active version: `src/v2/`** — see [`src/v2/README.md`](src/v2/README.md) for full documentation.

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

```bash
source venv/bin/activate

# Train (1M steps, ROS2 bridge for 30 Hz images)
python3 src/v2/train.py --ros2 --steps 1000000

# Resume from latest checkpoint
python3 src/v2/train.py --resume --ros2

# Test latest model
python3 src/v2/test.py --episodes 20

# Live camera feed (same preprocessing as the model)
python3 src/v2/view_camera.py --ros2 --stack

# TensorBoard
tensorboard --logdir logs_v2
```

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Observation | (4, 128, 128) grayscale CHW |
| Action | lateral + vertical correction [-1, 1] |
| Speed | randomised 1.0–3.0 m/s per episode |
| Goal distance | 50 m |
| Max steps | 2000 |
| Cruising altitude | -1.5 m (NED) |
| Checkpoint frequency | every 30k steps |
| Image source | ROS2 bridge (~30 Hz) or AirSim API (~1.5 Hz) |

## Reward Function

| Event | Reward |
|-------|--------|
| Goal reached | +10.0 |
| Collision | -10.0 |
| Proximity (depth < 5 m) | 0 to -2.0 |
| Action norm | -0.05 × ‖action‖ |

## Environment Randomisation

Each episode randomises goal direction, episode speed, forest density, and sun position (time of day). Trees are repositioned into an elliptical corridor aligned with the goal path using the AirSim `simSetObjectPose` API.

## Requirements

- Python 3.10+, PyTorch 2.0+, Stable-Baselines3 2.0+
- Cosys-AirSim + Unreal Engine 5.5
- ROS2 Jazzy (optional, recommended for training)
- CUDA GPU (recommended)

See [`src/v2/README.md`](src/v2/README.md) for the full design documentation.
