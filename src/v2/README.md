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
┌─────────────────────┐
│      train.py       │
│   PPO + CnnPolicy   │
│   DummyVecEnv       │
│   CheckpointCallback│
│   SuccessRateCallback│
│   ValidationCallback│
└─────────────────────┘
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
| `train.py` | PPO training loop with checkpoint/resume/validation support |
| `ros2_bridge.py` | Thread-safe ROS2 image subscriber (optional, high-frequency path) |
| `test.py` | Evaluate a trained model over N episodes |
| `fly_mission.py` | Fly a multi-waypoint mission with a trained model |
| `fly_baseline.py` | Controller-only baseline (no RL) for comparison |
| `view_camera.py` | Debug tool — display live camera feed identical to what the model sees |
| `tune.py` | Bayesian hyperparameter optimisation using Optuna — finds the best PPO config for a given density stage |
| `test_density.py` | Visual tool — place trees at a chosen density to inspect spacing before training |
| `monitor_drones.py` | Runtime telemetry monitor |
| `settings_sample.json` | AirSim `settings.json` template |
| `AIRSIM_ROS2_BRIDGE_SETUP.md` | Step-by-step ROS2 bridge setup guide |

---

## Design Decisions

### 1. Separation of Navigation and Avoidance

The RL model does **not** navigate. A deterministic controller computes a
forward velocity toward the goal every step. The model outputs only two
corrections: lateral (left/right, perpendicular to the goal direction) and
vertical (up/down). These are added on top of the controller velocity.

**Why:** Goal-conditioned navigation is a much harder problem that requires
the model to understand its position relative to the goal. By handing
navigation to a simple controller, the model can focus entirely on what it
can actually see in the camera — obstacles. It also makes the policy
directly reusable on a real drone with any navigation stack.

### 2. RGB Grayscale Instead of Depth as Observation

The model receives **4 stacked grayscale frames** derived from the RGB camera,
not the depth image.

**Why:** Depth sensors on real drones (e.g. Intel RealSense) have a very
different noise profile and range from AirSim's simulated depth. RGB cameras
are far more consistent between simulation and reality. Using RGB as the
observation eliminates one major source of the sim-to-real gap.

Depth is still fetched every step but is only used internally for the **soft
proximity penalty reward** (see below). It is never fed to the model.

### 3. Frame Stacking — (4, 128, 128) CHW

Four consecutive grayscale frames are stacked along the channel axis,
producing a `(4, 128, 128)` observation. This gives the CNN temporal context —
it can infer whether an obstacle is approaching or receding, and detect motion
even from a monocular camera.

The observation is already channels-first (CHW). Stable-Baselines3's
`CnnPolicy` detects this automatically because the first dimension (4) is
smaller than the spatial dimensions (128), so no `VecTransposeImage` wrapper
is needed.

**Frame update:** the stack is a rolling window — each step, the oldest frame
is dropped and the newest is appended at index `[-1]` using `np.roll`.

### 4. Yaw-to-Movement-Direction

At each step, the drone's yaw is set to face the direction of the combined
velocity vector (controller forward + model lateral correction), not simply
toward the goal. This ensures the **camera always looks at what is directly
ahead of the drone**, regardless of lateral drift.

At episode reset, `rotateToYawAsync` aligns the drone to face the goal before
the first frame is captured, so the initial frame stack contains a consistent
forward view from step 0.

### 5. Speed Randomisation

Each episode draws a random cruising speed from `speed_range = (1.0, 3.0) m/s`.
The controller moves at this speed; the model's corrections are applied on top.

**Why:** This is a form of domain randomisation. A model trained at a single
speed may overfit to the temporal signature of obstacles approaching at that
exact rate. Randomising speed forces the model to learn appearance-based
avoidance rather than motion-timing-based avoidance, which transfers better to
real deployment where wind and payload vary.

### 6. Action Smoothing

The action applied to AirSim is a 50/50 blend of the current model output and
the previous action:

```python
action = 0.5 * new_action + 0.5 * prev_action
```

**Why:** AirSim's physics engine can become unstable with abrupt velocity
reversals (e.g. hard left followed immediately by hard right). Smoothing
prevents oscillation and produces more natural flight trajectories. It also
acts as implicit regularisation — the model cannot exploit high-frequency
jitter to rack up large corrections.

### 7. Reward Structure

| Signal | Value | Condition |
|--------|-------|-----------|
| Goal reached | +10.0 | Distance to goal < `goal_radius` (5 m) |
| Collision | -10.0 | AirSim reports `has_collided` |
| Soft proximity penalty | 0 to -2.0 | Closest obstacle in centre 50% of depth image < `prox_threshold` (5 m) |
| Action norm penalty | -0.05 × ‖action‖ | Every step |

**Goal reward:** necessary to distinguish a successful episode from a timeout.
Without it, the model cannot tell the difference between reaching the goal and
running out of steps.

**Collision terminates immediately:** the episode ends on first collision
rather than continuing. This prevents the model from learning to tolerate
collisions by grinding out small rewards afterward.

**Soft proximity penalty:** gives a smooth gradient signal well before a
collision occurs. The penalty scales linearly: `0` at `prox_threshold` metres,
`-2.0` at `0` metres. Only the centre 50% of the depth image is used (the
quarter-crop on each side), so objects only visible in peripheral vision do not
dominate. Depth is clamped to `[0, prox_threshold]` before computing the
minimum, which (a) ignores background objects beyond the threshold, and (b)
guards against `NaN`/`inf` values that AirSim returns for open sky.

**Action norm penalty:** encourages the model to output small corrections when
no obstacle is present (i.e. trust the controller). The optimal policy for an
obstacle-free flight is `[0, 0]`.

### 8. ROS2 Bridge — Closing the Temporal Frequency Gap

The single largest source of the sim-to-real gap in v1 was **capture
frequency**. The AirSim Python API `simGetImages()` takes ~500 ms per call
(TCP round-trip + render). This means:

- Training frequency: ~1.5 Hz → 4 frames span ~2.7 seconds
- Real deployment (RealSense): 30 Hz → 4 frames span ~133 ms

The model trained at 1.5 Hz learns to react to motion that has already
finished by the time it executes an action on a real drone at 30 Hz.

`ros2_bridge.py` solves this by subscribing to the AirSim ROS2 bridge topics,
which publish at the Unreal Engine render rate (~30 Hz). Both the RGB and
depth streams are subscribed in a single background thread and cached. The
training loop reads the latest frame from memory (a mutex-protected dict
lookup, sub-millisecond), so the bottleneck shifts away from image capture
entirely.

```
Without ROS2 bridge:
  simGetImages() ≈ 500 ms  →  step ≈ 600 ms  →  ~1.5 Hz

With ROS2 bridge:
  get_latest_frame() ≈ 0.1 ms  →  step ≈ 10-20 ms  →  ~50-100 Hz (capped by sim tick)
```

The bridge is **optional** — pass `ros2_bridge=None` (default) to fall back to
the Python API. This preserves compatibility on machines without ROS2.

**Depth fallback:** if the bridge's depth topic has not yet received a frame
(or `depth_topic=None`), the environment automatically falls back to a single
`simGetImages` depth-only API call. Once the topic starts delivering frames,
the fallback is no longer used.

### 9. Dynamic Forest Generation

Each episode, trees are repositioned into an **ellipse aligned along the
goal direction**. The ellipse spans the full path length (semi-major axis)
with a ±10 m lateral corridor (semi-minor axis). Tree density is randomly
selected each episode:

| Config | Min spacing | Approx trees active |
|--------|-------------|---------------------|
| Dense  | 2.0 m       | ~195                |
| Medium | 3.0 m       | ~87                 |
| Sparse | 4.0 m       | ~50                 |

Trees that don't fit the chosen density are parked 500 m above the scene.

Tree actors are discovered at startup using `simListSceneObjects()`, filtered
by `TREE_ACTOR_FILTER = 'StaticMeshActor_UAID_'`. This matches all manually
placed static mesh actors while excluding landscape, HLOD, PCG, and system
actors which use different prefixes.

**Minimum recommended trees:** 200. With fewer trees, dense configurations
will be limited and the forest may look sparse.

### 10. Environment Randomisation

Each episode randomises:
- **Goal direction** — uniform random angle, fixed distance (50 m)
- **Episode speed** — uniform draw from `speed_range` (1.0–3.0 m/s)
- **Forest density** — randomly chosen from dense / medium / sparse
- **Sun position** — random time of day between 06:00 and 19:59 via `simSetTimeOfDay`

---

## Observation Space

```
Shape:  (4, 128, 128)   — channels-first (CHW)
dtype:  uint8
Range:  [0, 255]

Channel 0: oldest grayscale frame  (t - 3)
Channel 1:                         (t - 2)
Channel 2:                         (t - 1)
Channel 3: newest grayscale frame  (t)
```

## Action Space

```
Shape:  (2,)
dtype:  float32
Range:  [-1.0, 1.0]

action[0]: lateral correction  — scaled by lateral_scale (1.0 m/s)
action[1]: vertical correction — scaled by vertical_scale (1.0 m/s)
```

The controller always provides full forward velocity toward the goal.
The model adds corrections perpendicular to the goal direction.

---

## Hyperparameters

### Environment

| Parameter | Value | Description |
|-----------|-------|-------------|
| `speed_range` | (1.0, 3.0) m/s | Randomised forward speed per episode |
| `lateral_scale` | 1.0 | Max lateral correction velocity (m/s) |
| `vertical_scale` | 1.0 | Max vertical correction velocity (m/s) |
| `cruising_altitude` | -1.5 m | Target altitude (NED, negative = up) |
| `goal_distance_range` | (50, 50) | Goal distance from origin in metres |
| `goal_radius` | 5.0 m | Distance to consider goal reached |
| `max_steps` | 2000 | Steps before episode truncation |
| `prox_threshold` | 3.5 m | Depth distance that triggers proximity penalty |
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

### Camera / Observation

| Parameter | Value | Description |
|-----------|-------|-------------|
| Source | RGB Scene (ImageType 0) | Captured as colour, converted to grayscale |
| Resolution | 128×128 | Per frame, matches CnnPolicy input |
| Channels | 4 stacked frames | Observation shape: (4, 128, 128) CHW |
| FOV | 120 degrees | Wide angle for obstacle detection |
| Camera position | X=0.25, Z=-0.1 | Front-center, slightly above body |
| Depth (reward only) | ImageType 2, raw metres | Used for proximity penalty, not fed to model |

---

## Altitude Hold

A P-controller maintains cruising altitude:
```
altitude_error = cruising_altitude - current_z
base_vz = clip(altitude_error * 0.2, -1.0, 1.0)
body_vz = base_vz + vertical_correction
```

The model can add vertical corrections on top, but is clamped when already
above cruising altitude to prevent runaway climbs.

---

## Quick Start

Activate the venv from the project root before running any script:

```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
```

```bash
# Train (new run, 200k steps)
python3 src/v2/train.py

# Train with ROS2 bridge (recommended — 30 Hz image capture)
python3 src/v2/train.py --ros2

# Train for 1M steps
python3 src/v2/train.py --ros2 --steps 1000000

# Resume from latest checkpoint
python3 src/v2/train.py --resume --ros2

# Test a trained model (5 episodes)
python3 src/v2/test.py

# Test a specific model (20 episodes)
python3 src/v2/test.py --model models_v2/run_<timestamp>/simplified_avoidance_final.zip --episodes 20

# View the live camera feed (same preprocessing as the model sees)
python3 src/v2/view_camera.py --ros2
python3 src/v2/view_camera.py --ros2 --stack   # show all 4 stacked frames
```

### Hyperparameter Tuning

Run Bayesian hyperparameter search with Optuna before committing to a full
training run. Each trial trains for `--trial-steps` steps and is evaluated
by validation success rate. Bad trials are pruned early to save compute.
Results persist in a SQLite database so the sweep can be stopped and resumed.

```bash
# 30 trials at stage 0 (sparse forest), 200k steps each (~2-3 hrs/trial)
python3 src/v2/tune.py --trials 30 --ros2

# Resume an existing sweep (adds more trials on top)
python3 src/v2/tune.py --trials 20 --ros2

# Show best config found so far without running new trials
python3 src/v2/tune.py --show-best

# Tune at stage 1 starting from a good stage-0 checkpoint
python3 src/v2/tune.py --trials 20 --stage 1 \
    --checkpoint models_v2/run_<timestamp>/checkpoints/best.zip --ros2
```

**Monitor progress with Optuna Dashboard:**
```bash
optuna-dashboard sqlite:///tune_stage0.db
```
Then open `http://localhost:8080` to see trial history, parameter importances,
and pruned trials — updates live as new trials complete.

**Tuned hyperparameters:** `learning_rate`, `clip_range`, `n_epochs`,
`gae_lambda`, `vf_coef`, `target_kl`, `ent_coef`, `log_std_init`,
`n_steps`, `batch_size`, `gamma`, `max_grad_norm`.

Best params are saved to `best_params_stage0.json`. After tuning, update
`train.py` with the best values and run a full training.

---

### With ROS2 Bridge (recommended for training)

**Terminal 1 — Unreal Engine:** Open your environment and hit Play.

**Terminal 2 — ROS2 AirSim node:**
```bash
source /opt/ros/jazzy/setup.bash
source ~/Cosys-AirSim/ros2/install/setup.bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py host_ip:=localhost
```

**Terminal 3 — Training:**
```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
python3 src/v2/train.py --ros2 --steps 1000000
```

**Terminal 4 — TensorBoard:**
```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
tensorboard --logdir logs_v2
```

Then open `http://localhost:6006` in your browser.

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
| `success_rate` | Deterministic success rate over 30 episodes, logged every 70k steps |
| `lateral_avg` | Mean signed lateral correction during deterministic validation |
| `lateral_abs_avg` | Mean absolute lateral correction during deterministic validation |

**reward/**
| Metric | Description |
|--------|-------------|
| `proximity` | Rolling mean per-episode proximity penalty (negative — should not dominate) |
| `straight_bonus` | Rolling mean per-episode straight-path bonus (positive) |
| `action_norm` | Rolling mean per-episode action norm penalty (negative) |

**train/**
| Metric | Description |
|--------|-------------|
| `policy_gradient_loss` | How much the policy changes each update |
| `value_loss` | How well the critic estimates future rewards |
| `entropy_loss` | Exploration level |
| `approx_kl` | Policy drift per update — spikes signal unstable updates |
| `clip_fraction` | Fraction of updates hitting the PPO clip boundary |
| `explained_variance` | Critic quality — closer to 1.0 is better |
| `std` | Action distribution standard deviation — should stay ≤ 1.0 with BoundedStdCnnPolicy |

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
"ImageType": 2        // DepthPerspective — used for proximity penalty
"Width": 256          // Native render resolution (downsampled to 128x128 in code)
"Height": 256
"TargetFPS": 30       // Cap render rate to match ROS2 bridge target frequency
"FOV_Degrees": 120    // Wide FOV for better peripheral obstacle detection
```

---

## Output Directories

```
models_v2/
  run_YYYY-MM-DD_HH-MM-SS/
    checkpoints/                    # Saved every 30k steps
    simplified_avoidance_final.zip  # Final model

logs_v2/
  run_YYYY-MM-DD_HH-MM-SS/
    tensorboard/                    # TensorBoard logs
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
