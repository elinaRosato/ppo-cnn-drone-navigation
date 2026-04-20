# v2 — Obstacle Avoidance via PPO + CNN

This folder contains the second-generation training pipeline for the drone
obstacle avoidance system. It is a ground-up redesign focused on closing the
**sim-to-real gap** that caused v1 to require hundreds of thousands of steps
without converging.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AirSim (Unreal Engine)               │
│   Physics sim + RGB camera + depth camera (front_center)    │
└───────────┬────────────────────────┬────────────────────────┘
            │ AirSim Python API      │ AirSim ROS2 Bridge
            │ (reset, arm, takeoff,  │ (continuous image stream
            │  collision, velocity)  │  at ~30 Hz)
            │                        │
            ▼                        ▼
┌─────────────────────┐   ┌──────────────────────────┐
│  avoidance_env.py   │◄──│    ros2_bridge.py         │
│  (Gymnasium Env)    │   │  ROS2CameraBridge         │
│                     │   │  - caches latest RGB      │
│  Controller:        │   │  - caches latest depth    │
│   navigate to goal  │   │  - thread-safe getters    │
│                     │   └──────────────────────────┘
│  RL model:          │
│   avoid obstacles   │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────────────┐
│         train.py            │
│  PPO + AsymmetricAC policy  │
│  DummyVecEnv + VecNormalize │
│  CheckpointCallback         │
│  SuccessRateCallback        │
│  ValidationCallback         │
│  launch_training.py         │
└─────────────────────────────┘
```

The key split: **the controller handles navigation, the RL model handles only
obstacle avoidance.** This separation keeps the learning problem focused and
lets the model generalise across different goal positions without needing to
see goal-relative information.

---

## Files

| File | Purpose |
|------|---------|
| `avoidance_env.py` | Gymnasium environment — physics, rewards, frame stack, forest generation |
| `train.py` | PPO training loop with checkpoint/resume/validation/curriculum support |
| `ros2_bridge.py` | Thread-safe ROS2 image subscriber (optional, high-frequency path) |
| `launch_training.py` | One-command launcher — waits for AirSim, starts ROS2 node, TensorBoard, then training |
| `test.py` | Evaluate a trained model over N episodes |
| `tune.py` | Bayesian hyperparameter optimisation using Optuna |
| `view_camera.py` | Debug tool — display live camera feed identical to what the model sees |
| `fly_mission.py` | Fly a multi-waypoint mission with a trained model |
| `fly_baseline.py` | Controller-only baseline (no RL) for comparison |
| `test_density.py` | Visual tool — place trees at a chosen density to inspect spacing before training |
| `monitor_drones.py` | Runtime telemetry monitor |
| `settings_sample.json` | AirSim `settings.json` template |
| `AIRSIM_ROS2_BRIDGE_SETUP.md` | Step-by-step ROS2 bridge setup guide |

---

## Design Decisions

### 1. Separation of Navigation and Avoidance

The RL model does **not** navigate. A deterministic controller computes a
forward velocity toward the goal every step. The model outputs only a lateral
correction (left/right, perpendicular to the goal direction). Vertical altitude
is held by a P-controller and is not part of the action space.

**Why:** Goal-conditioned navigation is a much harder problem that requires
the model to understand its position relative to the goal. By handing
navigation to a simple controller, the model can focus entirely on what it
can actually see in the camera — obstacles. It also makes the policy
directly reusable on a real drone with any navigation stack.

### 2. RGB Grayscale Instead of Depth as Actor Observation

The actor receives **4 stacked grayscale frames** derived from the RGB camera,
not the depth image.

**Why:** Depth sensors on real drones (e.g. Intel RealSense) have a very
different noise profile and range from AirSim's simulated depth. RGB cameras
are far more consistent between simulation and reality. Using RGB as the
observation eliminates one major source of the sim-to-real gap.

Depth is still fetched every step but is used in two ways:
- **Proximity penalty reward** — clipped to `prox_threshold = 3.5 m`
- **Privileged critic observation** — clipped to `critic_depth_range = 15.0 m` (see §6)

### 3. CLAHE Contrast Normalisation

After grayscale conversion and resize, each frame passes through CLAHE
(`clipLimit=2.0, tileGridSize=(8,8)`). This boosts local contrast and makes
the preprocessing robust to different lighting conditions (time-of-day
randomisation, indoor vs. outdoor scenes).

### 4. Image Augmentation (Training Only)

During training (`training_mode=True`) each captured frame is randomly
augmented before being added to the frame stack:

- **Gaussian noise** σ ∈ [0.5, 3.0]
- **Brightness/contrast jitter** ±20 / ±15%
- **Gaussian blur** k ∈ {3, 5}, applied with 30% probability

Augmentation is disabled during validation and test. This is a form of domain
randomisation that prevents the model from overfitting to clean sim textures.

Use `view_camera.py --augment` to visualise the augmented input in real time.

### 5. Frame Stacking — (4, 192, 192) CHW

Four consecutive grayscale frames are stacked along the channel axis,
producing a `(4, 192, 192)` observation. This gives the CNN temporal context —
it can infer whether an obstacle is approaching or receding, and detect motion
even from a monocular camera.

**Frame update:** the stack is a rolling window — each step, the oldest frame
is dropped and the newest is appended at index `[-1]` using `np.roll`.

### 6. Asymmetric Actor-Critic

The policy uses an asymmetric design where the actor and critic receive
different information:

```
Actor:  image (4,192,192) + state (2,)      → CNN + MLP → 288 features → MLP[64,64] → action
Critic: same 288 features + privileged (1,) →                            MLP[128,128] → value
```

**Actor state vector** `[speed_norm, lateral_offset_norm]`:
- `speed_norm` = episode speed / max speed — helps the model time dodge manoeuvres
- `lateral_offset_norm` = signed perpendicular drift from the spawn→goal line / goal distance

**Privileged critic observation** `[min_depth_norm]`:
- Minimum depth in the centre 50% of the frame, clipped to `critic_depth_range = 15.0 m`
- Normalised to [0, 1]: 0 = obstacle touching, 1 = 15 m+ clear
- The longer clip range (vs 3.5 m for the penalty) lets the critic anticipate obstacles
  before they affect the reward, producing better value estimates
- Only the critic sees this during training; **the deployed actor needs only the RGB camera**

**Why asymmetric:** The critic is only used during training to compute advantages
(GAE). At deployment only the actor runs, so it is fine to give the critic
privileged sensor information that would be unavailable or unreliable on a real
drone.

### 7. Lateral Drift Penalty

A small penalty is applied each step when the drone drifts more than 1 m
perpendicular to the spawn→goal line:

```
drift_penalty = -(|lateral_offset| - 1.0) × 0.005   when |offset| > 1 m
```

**Why:** Without this, the model learns to dodge by strafing far to one side
and never returning. The penalty encourages it to re-centre after avoidance
manoeuvres, producing tighter, more efficient trajectories.

### 8. Yaw-to-Movement-Direction

At each step, the drone's yaw is set to face the direction of the combined
velocity vector (controller forward + model lateral correction), not simply
toward the goal. This ensures the **camera always looks at what is directly
ahead of the drone**, regardless of lateral drift.

At episode reset, `rotateToYawAsync` aligns the drone to face the goal before
the first frame is captured, so the initial frame stack contains a consistent
forward view from step 0.

### 9. Speed Randomisation

Each episode draws a random cruising speed from `speed_range = (1.0, 3.0) m/s`.
The controller moves at this speed; the model's corrections are applied on top.
The normalised speed is included in the state vector so the model can adapt its
dodge timing to the current episode speed.

**Why:** A model trained at a single speed may overfit to the temporal signature
of obstacles approaching at that exact rate. Randomising speed forces the model
to learn appearance-based avoidance rather than motion-timing-based avoidance.

### 10. Action Smoothing

The action applied to AirSim is a 50/50 blend of the current model output and
the previous action:

```python
action = 0.5 * new_action + 0.5 * prev_action
```

**Why:** AirSim's physics engine can become unstable with abrupt velocity
reversals. Smoothing prevents oscillation and produces more natural flight
trajectories.

### 11. Reward Structure

| Signal | Value | Condition |
|--------|-------|-----------|
| Goal reached | +10.0 | Distance to goal < `goal_radius` (5 m) |
| Collision | -100.0 | AirSim reports `has_collided` |
| Soft proximity penalty | 0 to −0.5 | Min depth in centre 50% < `prox_threshold` (3.5 m) |
| Clear-path bonus | 0 to +0.05 | Min depth ≥ 3.5 m, scales with how straight the action is |
| Action norm penalty | −0.1 × ‖action‖ | Every step |
| Lateral drift penalty | −(‖offset‖ − 1.0) × 0.005 | When lateral offset > 1 m |

### 12. ROS2 Bridge — Closing the Temporal Frequency Gap

The single largest source of the sim-to-real gap in v1 was **capture
frequency**. The AirSim Python API `simGetImages()` takes ~500 ms per call
(TCP round-trip + render). This means:

- Training frequency: ~1.5 Hz → 4 frames span ~2.7 seconds
- Real deployment (RealSense): 30 Hz → 4 frames span ~133 ms

`ros2_bridge.py` solves this by subscribing to the AirSim ROS2 bridge topics,
which publish at the Unreal Engine render rate (~30 Hz). The training loop
reads the latest frame from memory (sub-millisecond), so the bottleneck shifts
away from image capture entirely.

```
Without ROS2 bridge:
  simGetImages() ≈ 500 ms  →  step ≈ 600 ms  →  ~1.5 Hz

With ROS2 bridge:
  get_latest_frame() ≈ 0.1 ms  →  step ≈ 10-20 ms  →  ~50-100 Hz
```

The bridge is **optional** — pass `ros2_bridge=None` (default) to fall back to
the Python API.

### 13. Dynamic Forest Generation

Each episode, trees are repositioned into an **ellipse aligned along the goal
direction**. The ellipse spans the full path length with a ±10 m lateral
corridor. A density curriculum controls tree spacing:

| Stage | Min spacing | Mix |
|-------|-------------|-----|
| 0 | 8.0 m | sparse only |
| 1 | 6.5 m | sparse + medium |
| 2 | 5.0 m | all densities |

The curriculum advances automatically after two consecutive validation runs
above 80% success rate.

**Background trees:** approximately 30% of available tree actors are placed
outside the flight corridor (>5 m lateral clearance) as visual background.
This makes the scene look more realistic and prevents the model from learning
to treat a sparse background as a shortcut signal.

Trees that don't fit in the obstacle zone are parked at a remote location.

### 14. Environment Randomisation

Each episode randomises:
- **Goal direction** — uniform random angle, fixed distance (50 m)
- **Episode speed** — uniform draw from `speed_range` (1.0–3.0 m/s)
- **Forest density** — randomly chosen for the current curriculum stage
- **Sun position** — random time of day between 06:00 and 19:59 via `simSetTimeOfDay`

### 15. VecNormalize Reward Normalisation

Rewards are normalised online using `VecNormalize(norm_obs=False,
norm_reward=True, gamma=0.99, clip_reward=10.0)`. This stabilises training
across the large reward range (goal +10 vs. collision −100) without changing
the reward signal semantics.

The normaliser statistics are saved alongside every checkpoint as
`vecnormalize_<step>_steps.pkl` and automatically loaded on resume.

---

## Observation Space

```
Type:   Dict
{
    "image":     (4, 192, 192) uint8   — 4 stacked CLAHE grayscale frames (CHW)
    "state":     (2,)          float32 — [speed_norm, lateral_offset_norm]
    "privileged":(1,)          float32 — [min_depth_norm] — critic only
}
```

The actor uses `"image"` and `"state"` only. `"privileged"` is passed to the
critic during training and is ignored at deployment.

## Action Space

```
Shape:  (1,)
dtype:  float32
Range:  [-1.0, 1.0]

action[0]: lateral correction — scaled by lateral_scale (1.5 m/s)
```

The controller always provides full forward velocity toward the goal.
Vertical altitude is held by a P-controller independently.

---

## Hyperparameters

### Environment

| Parameter | Value | Description |
|-----------|-------|-------------|
| `speed_range` | (1.0, 3.0) m/s | Randomised forward speed per episode |
| `lateral_scale` | 1.5 | Max lateral correction velocity (m/s) |
| `cruising_altitude` | −1.5 m | Target altitude (NED, negative = up) |
| `goal_distance` | 50 m | Fixed goal distance from spawn |
| `goal_radius` | 5.0 m | Distance to consider goal reached |
| `max_steps` | 2000 | Steps before episode truncation |
| `prox_threshold` | 3.5 m | Depth clip range for proximity penalty |
| `critic_depth_range` | 15.0 m | Depth clip range for privileged critic obs |
| `stack_frames` | 4 | Number of consecutive frames stacked |
| `action_momentum` | 0.5 | Blend ratio with previous action (smoothing) |
| `bg_tree_fraction` | 0.30 | Fraction of tree actors used as background |

### PPO (Stable-Baselines3) — Optuna-tuned

| Parameter | Value | Description |
|-----------|-------|-------------|
| `policy` | AsymmetricActorCriticPolicy | Custom asymmetric actor-critic |
| `learning_rate` | 1.47e-4 | Adam optimizer learning rate |
| `n_steps` | 4096 | Steps per rollout buffer |
| `batch_size` | 512 | Minibatch size for gradient updates |
| `n_epochs` | 9 | Passes over rollout data per update |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.940 | GAE lambda for advantage estimation |
| `clip_range` | 0.2 | PPO clipping range |
| `ent_coef` | 0.00183 | Entropy coefficient (exploration) |
| `vf_coef` | 1.0 | Value function loss coefficient |
| `max_grad_norm` | 0.993 | Gradient clipping |
| `target_kl` | 0.033 | Early-stop KL threshold per update |
| `log_std_init` | −0.111 | Initial log std for action distribution |

### Camera / Observation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Source | RGB Scene (ImageType 0) | Captured as colour, converted to grayscale |
| Resolution | 192×192 | Per frame, after resize |
| Channels | 4 stacked frames | Observation shape: (4, 192, 192) CHW |
| Preprocessing | CLAHE (clipLimit=2.0) | Applied after grayscale conversion |
| FOV | 120 degrees | Wide angle for obstacle detection |
| Depth (reward + critic) | ImageType 2, raw metres | Not fed to the actor |

---

## Quick Start

Activate the venv from the project root before running any script:

```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
```

### Recommended: one-command launcher

`launch_training.py` automates all setup steps after you hit Play in UE5:
waits for AirSim, starts the ROS2 node, waits for the camera topic, starts
TensorBoard, then starts training.

```bash
# New training run
python3 src/v2/launch_training.py

# Resume from latest checkpoint
python3 src/v2/launch_training.py --resume

# Custom step count
python3 src/v2/launch_training.py --steps 1000000

# Hyperparameter tuning
python3 src/v2/launch_training.py --tune --trials 30
```

### Manual

```bash
# Train (new run)
python3 src/v2/train.py --ros2

# Resume from latest checkpoint
python3 src/v2/train.py --resume --ros2

# Test a trained model (5 episodes)
python3 src/v2/test.py

# Test a specific model (20 episodes)
python3 src/v2/test.py --model models_v2/run_<timestamp>/simplified_avoidance_final.zip --episodes 20

# View the live camera feed (same preprocessing as the model sees)
python3 src/v2/view_camera.py
python3 src/v2/view_camera.py --stack              # show all 4 stacked frames
python3 src/v2/view_camera.py --depth              # show depth with penalty region
python3 src/v2/view_camera.py --augment            # show augmented input (training noise/jitter)
python3 src/v2/view_camera.py --augment --stack    # augmented frame stack
```

### Hyperparameter Tuning

```bash
# 30 trials at stage 0 (sparse forest), 200k steps each
python3 src/v2/tune.py --trials 30 --ros2

# Show best config found so far without running new trials
python3 src/v2/tune.py --show-best

# Tune at stage 1 starting from a good stage-0 checkpoint
python3 src/v2/tune.py --trials 20 --stage 1 \
    --checkpoint models_v2/run_<timestamp>/checkpoints/best.zip --ros2
```

Monitor with Optuna Dashboard:
```bash
optuna-dashboard sqlite:///tune_stage0.db
# open http://localhost:8080
```

---

### Manual multi-terminal setup (without launch_training.py)

**Terminal 1 — Unreal Engine:** Open your environment and hit Play.

**Terminal 2 — ROS2 AirSim node:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/Cosys-AirSim/ros2/install/setup.bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py host:=localhost
```

**Terminal 3 — Training:**
```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
python3 src/v2/train.py --ros2 --steps 1000000
```

**Terminal 4 — TensorBoard:**
```bash
tensorboard --logdir src/v2/logs_v2
# open http://localhost:6006
```

---

## TensorBoard Metrics

**rollout/**
| Metric | Description |
|--------|-------------|
| `ep_rew_mean` | Average episode reward — main signal, should trend upward |
| `ep_len_mean` | Average episode length |
| `success_rate` | Rolling success rate over last 50 training episodes |
| `episodes` | Total training episodes completed |
| `lateral_avg` | Rolling mean signed lateral correction (bias indicator, should be near 0) |
| `lateral_abs_avg` | Rolling mean absolute lateral correction (how much the model steers) |

**validation/**
| Metric | Description |
|--------|-------------|
| `success_rate` | Deterministic success rate over 30 episodes, logged every ~74k steps |
| `lateral_avg` | Mean signed lateral correction during deterministic validation |
| `lateral_abs_avg` | Mean absolute lateral correction during deterministic validation |

**reward/**
| Metric | Description |
|--------|-------------|
| `proximity` | Rolling mean per-episode proximity penalty (negative) |
| `straight_bonus` | Rolling mean per-episode clear-path bonus (positive) |
| `action_norm` | Rolling mean per-episode action norm penalty (negative) |
| `drift` | Rolling mean per-episode lateral drift penalty (negative, should approach 0) |

**train/**
| Metric | Description |
|--------|-------------|
| `policy_gradient_loss` | How much the policy changes each update |
| `value_loss` | How well the critic estimates future rewards |
| `entropy_loss` | Exploration level |
| `approx_kl` | Policy drift per update — spikes signal unstable updates |
| `clip_fraction` | Fraction of updates hitting the PPO clip boundary |
| `explained_variance` | Critic quality — closer to 1.0 is better |
| `std` | Action distribution standard deviation |

**curriculum/**
| Metric | Description |
|--------|-------------|
| `density_stage` | Current forest density stage (0=sparse, 1=sparse+medium, 2=all) |

`rollout/success_rate` tracks stochastic training episodes (includes exploration noise).
`validation/success_rate` uses `deterministic=True` and is the better indicator of true
policy quality.

---

## AirSim Settings

Use `settings_sample.json` as your `~/Documents/AirSim/settings.json`.
Key parameters:

```json
"ClockSpeed": 10.0    // Speed up simulation for faster training
"ImageType": 0        // Scene (RGB) — used for observations
"ImageType": 2        // DepthPerspective — used for proximity penalty and critic
"Width": 256          // Native render resolution (downsampled to 192×192 in code)
"Height": 256
"TargetFPS": 30       // Cap render rate to match ROS2 bridge target frequency
"FOV_Degrees": 120    // Wide FOV for better peripheral obstacle detection
```

---

## Output Directories

```
models_v2/
  run_YYYY-MM-DD_HH-MM-SS/
    checkpoints/
      simplified_avoidance_<N>_steps.zip   # Saved every 30k steps
      vecnormalize_<N>_steps.pkl           # VecNormalize stats (saved alongside each checkpoint)
    simplified_avoidance_final.zip         # Final model
    vecnormalize_final.pkl                 # Final VecNormalize stats

logs_v2/
  run_YYYY-MM-DD_HH-MM-SS/
    tensorboard/                           # TensorBoard logs
```

---

## Step Timing

Per-step timing is printed every 25 steps to help diagnose bottlenecks:

```
[TIMING ms] pos1=4.2 move=1.1 images=612.3 collision=3.8 pos2=4.0 compute=625.4 (~1.6 Hz)
[IMG-API ms] simGetImages=609.7
```

With the ROS2 bridge active, `images=` drops from ~600 ms to < 1 ms:

```
[TIMING ms] pos1=3.1 move=0.9 images=0.1 collision=3.5 pos2=3.2 compute=10.8 (~30 Hz)
[IMG-ROS2 ms] rgb_cache=0.1 depth_ros2=0.1
```

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Stable-Baselines3 2.0+
- Cosys-AirSim Python client (`cosysairsim`)
- Unreal Engine 5.5 with Cosys-AirSim plugin
- ROS2 Jazzy (optional, for 30 Hz image capture)
- CUDA GPU (recommended)
