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
            │  collision, velocity)  │  at ~20-30 Hz)
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
| `avoidance_env.py` | Gymnasium environment — physics, rewards, frame stack |
| `train.py` | PPO training loop with checkpoint/resume support |
| `ros2_bridge.py` | Thread-safe ROS2 image subscriber (optional, high-frequency path) |
| `test.py` | Evaluate a trained model over N episodes |
| `fly_mission.py` | Fly a multi-waypoint mission with a trained model |
| `fly_baseline.py` | Controller-only baseline (no RL) for comparison |
| `view_camera.py` | Debug tool — display live camera feed from AirSim |
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

### 3. Frame Stacking — (4, 84, 84) CHW

Four consecutive grayscale frames are stacked along the channel axis,
producing a `(4, 84, 84)` observation. This gives the CNN temporal context —
it can infer whether an obstacle is approaching or receding, and detect motion
even from a monocular camera.

The observation is already channels-first (CHW). Stable-Baselines3's
`CnnPolicy` detects this automatically because the first dimension (4) is
smaller than the spatial dimensions (84), so no `VecTransposeImage` wrapper
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

Each episode draws a random cruising speed from `speed_range = (1.0, 3.0)
m/s`. The controller moves at this speed; the model's corrections are applied
on top.

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
which publish at the Unreal Engine render rate (~20–30 Hz). Both the RGB and
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

---

## Observation Space

```
Shape:  (4, 84, 84)   — channels-first (CHW)
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
action[1]: vertical correction — scaled by vertical_scale (0.5 m/s)
```

The controller always provides full forward velocity toward the goal.
The model adds corrections perpendicular to the goal direction.

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

# Resume from latest checkpoint
python3 src/v2/train.py --resume

# Test a trained model
python3 src/v2/test.py --model src/v2/models_v2/run_<timestamp>/simplified_avoidance_final.zip

# Fly a waypoint mission
python3 src/v2/fly_mission.py --model src/v2/models_v2/run_<timestamp>/simplified_avoidance_final.zip --speed 2.0
```

### With ROS2 Bridge (recommended for training)

**Terminal 1 — Unreal Engine:** Open Blocks environment and hit Play.

**Terminal 2 — ROS2 AirSim node (Linux):**
```bash
source /opt/ros/jazzy/setup.bash
source ~/Cosys-AirSim/ros2/install/setup.bash
ros2 launch airsim_ros_pkgs airsim_node.launch.py host:=localhost
```

**Terminal 3 — Training:**
```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
python3 src/v2/train.py --ros2
```

**Terminal 4 — TensorBoard:**
```bash
source ~/Sites/ppo-cnn-drone-navigation/venv/bin/activate
cd ~/Sites/ppo-cnn-drone-navigation
tensorboard --logdir logs_v2
```

Then open `http://localhost:6006` in your browser.

### TensorBoard Metrics

TensorBoard shows the standard PPO metrics logged automatically by Stable-Baselines3.

**rollout/**
| Metric | Description |
|--------|-------------|
| `ep_rew_mean` | Average episode reward — the main signal to watch, should trend upward |
| `ep_len_mean` | Average episode length (steps per episode) |

**train/**
| Metric | Description |
|--------|-------------|
| `policy_gradient_loss` | How much the policy is changing each update |
| `value_loss` | How well the critic estimates future rewards |
| `entropy_loss` | Exploration level — higher means more random actions |
| `approx_kl` | How far the new policy drifted from the old one per update |
| `clip_fraction` | Fraction of updates that hit the PPO clip boundary |
| `explained_variance` | How well the value function predicts actual returns — closer to 1.0 is better |
| `learning_rate` | Constant at `1e-4` |

The most important metrics are **`ep_rew_mean`** (is the drone learning to avoid obstacles and reach goals?) and **`explained_variance`** (is the critic learning?). If `ep_rew_mean` is flat or falling, the drone is not improving.

See `AIRSIM_ROS2_BRIDGE_SETUP.md` for full setup instructions (Linux native and Windows + WSL2).

---

## AirSim Settings

Use `settings_sample.json` as your `~/Documents/AirSim/settings.json`.
Key parameters:

```json
"ClockSpeed": 10.0    // Speed up simulation for faster training
"ImageType": 0        // Scene (RGB) — used for observations
"ImageType": 2        // DepthPerspective — used for proximity penalty
"Width": 256          // Native render resolution (downsampled to 84x84 in code)
"Height": 256
"FOV_Degrees": 120    // Wide FOV for better peripheral obstacle detection
```

---

## Step Timing

Per-step timing is printed every 25 steps to help diagnose bottlenecks:

```
[TIMING ms] pos1=4.2 move=1.1 images=612.3 collision=3.8 pos2=4.0 total=625.4 (~1.6 Hz)
[IMG-API ms] simGetImages=609.7
```

With the ROS2 bridge active, `images=` drops from ~600 ms to < 1 ms and
`depth_ros2=` replaces `depth_api=`.
