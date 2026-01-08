# Training Speed Optimization Guide

## Current Speed vs Optimized

| Configuration | Episodes/Hour | Time for 500k Steps |
|---------------|---------------|---------------------|
| **Default (with graphics)** | ~10-20 | 12-24 hours |
| **ClockSpeed=5 only** | ~50-100 | 6-10 hours |
| **With fast config** | ~100-200 | 2-4 hours |

---

## 🚀 Speed-Up Techniques

### 1. **Headless Mode** (No Graphics) ⚠️ **HAS ISSUES**

**What it does:** Runs simulation without rendering graphics

**Attempted:**
```bash
Blocks.exe -RenderOffScreen -NoVSync -SILENT
```

**Problem:** ❌ Shader library errors
```
Error: game files required to initialize the global shader library are missing
```

**Current Workaround:** Run with graphics window (can minimize it)
```bash
# launch_headless.bat now just launches normally
Blocks.exe
```

**Speed gain:** None (doesn't work reliably) - **Use ClockSpeed instead!**

---

### 2. **Clock Speed** - 5-10x Faster ⚡⚡⚡

**What it does:** Speeds up simulation physics

**Already configured in settings.json:**
```json
"ClockSpeed": 5.0  // 5x faster than real-time
```

**You can go even faster:**
```json
"ClockSpeed": 10.0  // 10x faster (may reduce physics accuracy)
"ClockSpeed": 20.0  // 20x faster (experimental, may be unstable)
```

**Trade-off:**
- Higher = faster training
- Too high = physics becomes unrealistic
- Recommended: 5-10 for good balance

---

### 3. **Reduce Image Resolution** - 2-3x Faster ⚡⚡

**Current:** 84x84 images

**Reduce to:**
```python
# In airsim_env_v3.py
env = AirSimDroneEnv(
    img_height=64,  # Instead of 84
    img_width=64,   # Instead of 84
    ...
)
```

**Even smaller for faster training:**
```python
img_height=48,  # Very fast
img_width=48,
```

**Trade-off:**
- Smaller = faster training, less detail
- Larger = slower training, more detail
- 64x64 is good balance

---

### 4. **Parallel Environments** (Advanced) - 4-8x Faster ⚡⚡⚡

**Run multiple simulations at once!**

Currently you have:
```python
env = DummyVecEnv([make_env])  # 1 environment
```

With multiple:
```python
from stable_baselines3.common.vec_env import SubprocVecEnv

# 4 parallel environments
env = SubprocVecEnv([make_env for _ in range(4)])
```

**Requirements:**
- Need to run 4 AirSim instances on different ports
- More complex setup
- Requires more RAM/CPU

**Configuration for multiple instances:**
```python
def make_env(port_offset):
    def _init():
        env = AirSimDroneEnv(...)
        # Connect to different AirSim instance
        env.client = airsim.MultirotorClient(port=41451 + port_offset)
        return env
    return _init

# 4 parallel environments on ports 41451, 41452, 41453, 41454
env = SubprocVecEnv([make_env(i) for i in range(4)])
```

**Note:** This is advanced - start with other optimizations first!

---

### 5. **Disable Recording** - Small Gain ⚡

**Already configured in settings.json:**
```json
"Recording": {
  "RecordOnMove": false,
  "RecordInterval": 0
}
```

---

### 6. **Use GPU for Training** - 2-3x Faster ⚡⚡

**Check if you have GPU:**
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

**Already configured in training scripts:**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
```

**If you have GPU, it's automatically used!**

---

### 7. **Reduce n_steps** - Faster Updates ⚡

**Current:**
```python
model = PPO(
    ...
    n_steps=2048,  # Collect 2048 steps before update
    ...
)
```

**Reduce for faster updates:**
```python
n_steps=1024,  # Update twice as often
# or
n_steps=512,   # Update 4x as often
```

**Trade-off:**
- Lower = more frequent updates, potentially less stable
- Higher = fewer updates, more stable
- 1024 is good balance

---

### 8. **Smaller Batch Size** - Faster Gradient Steps ⚡

**Current:**
```python
batch_size=64,
```

**Reduce:**
```python
batch_size=32,  # Faster
```

**Trade-off:**
- Smaller = faster, noisier gradients
- Larger = slower, more stable
- 32-64 is typical range

---

### 9. **Reduce Network Size** (Not Recommended) ⚠️

**Current:**
```python
net_arch=dict(pi=[256, 256], vf=[256, 256])
```

**Smaller (faster but less capacity):**
```python
net_arch=dict(pi=[128, 128], vf=[128, 128])
```

**Not recommended unless you need speed desperately**

---

## 🎯 Recommended Configuration for Fast Training

Create `train_ppo_v3_fast.py`:

```python
"""
Fast training configuration - optimized for speed
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_v3 import AirSimDroneEnv


def make_env():
    env = AirSimDroneEnv(
        randomize_goals=True,
        randomize_height_bounds=True,
        goal_range_x=(5, 25),
        goal_range_y=(5, 25),
        height_bound_ranges={
            'max_height': (-1.0, -1.8),
            'min_height': (-2.5, -4.0)
        },
        img_height=64,        # REDUCED from 84
        img_width=64,         # REDUCED from 84
        max_steps=300         # REDUCED from 500 (faster episodes)
    )
    return Monitor(env)


def train():
    os.makedirs("models_v3_fast", exist_ok=True)

    print("FAST TRAINING MODE")
    print("Optimizations: Headless + ClockSpeed=5 + Small images + Reduced steps")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    env = DummyVecEnv([make_env])

    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=1024,         # REDUCED from 2048
        batch_size=32,        # REDUCED from 64
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        verbose=1,
        tensorboard_log="./logs_v3_fast/",
        device=device
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,      # Save less often
        save_path="./models_v3_fast/",
        name_prefix="ppo_fast"
    )

    print("Training 300k steps (faster than 500k)")
    model.learn(
        total_timesteps=300_000,  # REDUCED from 500k
        callback=checkpoint_callback,
        progress_bar=True
    )

    model.save("./models_v3_fast/final")
    env.close()


if __name__ == "__main__":
    train()
```

---

## 📋 Quick Start for Fast Training

### Step 1: Update settings.json
```json
{
  "ClockSpeed": 5.0,
  "ViewMode": "",
  "Recording": {"RecordOnMove": false}
}
```
✅ Already done!

### Step 2: Launch headless
```bash
launch_headless.bat
```

### Step 3: Train with optimized settings
```bash
python train_ppo_v3_fast.py
```

---

## ⚡ Speed Comparison

### Configuration Levels:

#### Level 1: Default (Slowest)
```
- Normal launch (with graphics)
- ClockSpeed = 1
- Image: 84x84
- n_steps = 2048

Speed: ~10-20 episodes/hour
Time: 12-24 hours for 500k steps
```

#### Level 2: Moderate
```
- Headless mode
- ClockSpeed = 5
- Image: 84x84
- n_steps = 2048

Speed: ~50-100 episodes/hour
Time: 4-6 hours for 500k steps
```

#### Level 3: Fast (Recommended) ⭐
```
- Headless mode
- ClockSpeed = 5
- Image: 64x64
- n_steps = 1024
- Reduced to 300k steps

Speed: ~100-200 episodes/hour
Time: 1.5-3 hours for 300k steps
```

#### Level 4: Ultra Fast (Experimental)
```
- Headless mode
- ClockSpeed = 10
- Image: 48x48
- n_steps = 512
- Reduced to 200k steps

Speed: ~200-400 episodes/hour
Time: 30 mins - 1 hour for 200k steps
WARNING: May reduce training quality!
```

---

## 🎮 Monitoring Without Graphics

Since you can't see the simulation, use TensorBoard:

```bash
tensorboard --logdir=./logs_v3_fast/
```

Open: http://localhost:6006

**Watch these metrics:**
- `rollout/ep_rew_mean` - Average reward (should increase)
- `rollout/ep_len_mean` - Episode length (should stabilize)
- `train/policy_loss` - Policy loss (should decrease initially)

---

## ⚠️ Important Notes

### Physics Accuracy vs Speed

Higher ClockSpeed = Less accurate physics

**Recommended:**
- Training: ClockSpeed = 5-10 (fast, good enough)
- Final testing: ClockSpeed = 1 (accurate)
- Deployment: Real world (perfect physics!)

The model learns policies that work across speeds.

### When to Use What

**Use slow (ClockSpeed=1, graphics on):**
- Initial testing
- Debugging issues
- Final evaluation
- Making videos

**Use fast (ClockSpeed=5-10, headless):**
- Main training
- Hyperparameter search
- Long training runs

---

## 💾 Disk Space

**Faster training = Less disk usage:**
- Save checkpoints less frequently
- Don't record episodes
- TensorBoard logs stay small

---

## Summary: Best Practice (UPDATED)

```bash
# 1. Configure settings.json (already done!)
#    - ClockSpeed: 5.0
#    - Recording: disabled

# 2. Launch AirSim normally (headless mode has shader issues)
#    Just double-click: Blocks.exe
#    Or use: launch_headless.bat (now launches normally)

# 3. Minimize the window to keep it out of the way

# 4. Train with quality settings
python train_ppo_v3_quality.py

# Expected: 6-10 hours for 500k steps
# vs 50-60 hours with default settings

# 5. Monitor progress
tensorboard --logdir=./logs_v3_quality/tensorboard/
```

**Speed gain: 5-8x faster (ClockSpeed only)** 🚀

**Note:** True headless mode (-RenderOffScreen) causes shader library errors.
The working optimization is ClockSpeed=5 with graphics window (can be minimized).
