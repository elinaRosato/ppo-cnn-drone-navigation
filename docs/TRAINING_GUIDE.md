# PPO + CNN Training Guide for AirSim Drone Navigation

This guide explains how to train a drone to navigate from point A to point B while avoiding obstacles using **PPO (Proximal Policy Optimization)** with **CNN (Convolutional Neural Network)** policy, powered by **Stable-Baselines3** and **PyTorch**.

## Overview

### Task
Train a drone to fly from a start position to a goal position while:
- **Avoiding collisions** with blocks/obstacles
- **Flying between blocks** (not over them) by constraining height
- **Learning from camera images** (first-person view)

### Technology Stack
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Policy Network**: CNN for processing camera images
- **Framework**: Stable-Baselines3 (SB3)
- **Backend**: PyTorch
- **Simulator**: Microsoft AirSim

---

## Installation

### 1. Install RL Dependencies

Make sure your virtual environment is activated:
```bash
cd C:\Users\Dator\Sites\uav-simulation-training
venv\Scripts\activate
```

Install the new requirements:
```bash
pip install torch>=2.0.0
pip install stable-baselines3[extra]>=2.0.0
pip install gymnasium>=0.29.0
pip install tensorboard>=2.14.0
```

Or install all at once:
```bash
pip install -r requirements.txt
```

---

## Project Structure

```
uav-simulation-training/
├── airsim_env.py           # Custom Gym environment for AirSim
├── train_ppo.py            # Training script with PPO + CNN
├── test_ppo.py             # Testing/evaluation script
├── main.py                 # Basic AirSim control script
├── requirements.txt        # Python dependencies
├── README.md               # General project documentation
├── TRAINING_GUIDE.md       # This file
│
├── models/                 # Saved models
│   ├── checkpoints/        # Training checkpoints
│   ├── best_model/         # Best model based on evaluation
│   └── ppo_airsim_drone_final.zip
│
└── logs/                   # Training logs
    ├── tensorboard/        # TensorBoard logs
    └── eval/               # Evaluation logs
```

---

## Environment Configuration

### Height Constraints
The environment is configured to force the drone to fly **between blocks**:

```python
max_height = -1.5  # Maximum height (ceiling)
min_height = -3.0  # Minimum height (floor)
```

**Note**: In AirSim, Z-axis is negative upward:
- `-1.5` is **higher** (closer to ceiling)
- `-3.0` is **lower** (closer to floor)

### Reward Function

The reward function encourages good behavior:

1. **Progress Reward** (+): Getting closer to goal (+10 × distance_reduced)
2. **Collision Penalty** (-): Hitting obstacles (-100)
3. **Height Penalty** (-): Going outside height bounds (-50 × deviation)
4. **Height Bonus** (+): Staying within bounds (+1 per step)
5. **Goal Reward** (+): Reaching the goal (+200)
6. **Step Penalty** (-): Encourage efficiency (-0.1 per step)

### Observation Space
- **Type**: RGB camera image from front camera
- **Shape**: (84, 84, 3) - Height × Width × Channels
- **Stacking**: 4 consecutive frames stacked for temporal information

### Action Space
- **Type**: Continuous velocities
- **Shape**: (3,) - [vx, vy, vz]
- **Range**: [-1, 1] (scaled to actual velocities in environment)

---

## Training

### 1. Start AirSim Environment

**First, launch the Blocks environment:**
```bash
"C:\Users\Dator\Documents\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" -windowed
```

Wait for the environment to fully load.

### 2. Start Training

Activate your venv and run:
```bash
cd C:\Users\Dator\Sites\uav-simulation-training
venv\Scripts\activate
python train_ppo.py
```

### 3. Monitor Training

**TensorBoard** (real-time monitoring):
```bash
# In a separate terminal
cd C:\Users\Dator\Sites\uav-simulation-training
venv\Scripts\activate
tensorboard --logdir=./logs/tensorboard/
```

Then open your browser to: `http://localhost:6006`

**Metrics to watch:**
- `rollout/ep_rew_mean` - Average episode reward
- `rollout/ep_len_mean` - Average episode length
- `train/policy_loss` - Policy network loss
- `train/value_loss` - Value network loss

### Training Parameters

Current configuration in `train_ppo.py`:

```python
total_timesteps = 500,000      # Total training steps
learning_rate = 3e-4           # Learning rate
n_steps = 2048                 # Steps before update
batch_size = 64                # Mini-batch size
n_epochs = 10                  # Optimization epochs
gamma = 0.99                   # Discount factor
```

**Expected training time**:
- With GPU: ~4-8 hours
- With CPU: ~12-24 hours

### Checkpoints

Models are automatically saved:
- Every **10,000 steps**: `./models/checkpoints/`
- **Best model** (based on eval): `./models/best_model/`
- **Final model**: `./models/ppo_airsim_drone_final.zip`

---

## Testing

### Test Trained Model

```bash
python test_ppo.py --model ./models/best_model/best_model.zip --episodes 10
```

**Arguments:**
- `--model`: Path to trained model (default: best_model)
- `--episodes`: Number of test episodes (default: 5)

### Test Other Checkpoints

```bash
# Test a specific checkpoint
python test_ppo.py --model ./models/checkpoints/ppo_airsim_drone_100000_steps.zip

# Test final model
python test_ppo.py --model ./models/ppo_airsim_drone_final.zip
```

---

## Hyperparameter Tuning

### Modify Training Parameters

Edit `train_ppo.py` to adjust:

```python
model = PPO(
    "CnnPolicy",
    env,
    learning_rate=3e-4,        # Try: 1e-4, 5e-4
    n_steps=2048,              # Try: 1024, 4096
    batch_size=64,             # Try: 32, 128
    n_epochs=10,               # Try: 5, 15
    gamma=0.99,                # Discount factor
    gae_lambda=0.95,           # GAE lambda
    clip_range=0.2,            # PPO clip range
    ent_coef=0.01,             # Entropy coefficient (exploration)
    ...
)
```

### Modify Environment

Edit `airsim_env.py` to adjust:

```python
# Height bounds
max_height = -1.5  # Ceiling
min_height = -3.0  # Floor

# Episode length
max_steps = 500    # Maximum steps per episode

# Goal position
goal_pos = (15, 15, -2)  # (x, y, z)

# Image size
img_height = 84
img_width = 84
```

---

## Troubleshooting

### Issue: Training is slow
**Solution**:
- Use GPU if available (check with `torch.cuda.is_available()`)
- Reduce `n_steps` or `batch_size`
- Reduce image resolution

### Issue: Drone always collides
**Solution**:
- Increase collision penalty in reward function
- Reduce action velocity scaling
- Train for more timesteps
- Adjust `ent_coef` for more exploration

### Issue: Drone flies over blocks
**Solution**:
- Decrease `max_height` (more negative)
- Increase height penalty multiplier
- Add bonus for staying within bounds

### Issue: "Connection refused" error
**Solution**:
- Make sure AirSim Blocks environment is running
- Check that AirSim is fully loaded before starting training
- Restart AirSim if it becomes unresponsive

### Issue: Training crashes
**Solution**:
- Models are saved at checkpoints - resume from last checkpoint
- Reduce batch size if out of memory
- Check AirSim simulator hasn't crashed

---

## Advanced Usage

### Resume Training

```python
# Load checkpoint and continue training
model = PPO.load("./models/checkpoints/ppo_airsim_drone_50000_steps.zip", env=env)
model.learn(total_timesteps=500000)  # Continue training
```

### Transfer Learning

```python
# Start from pre-trained model
model = PPO.load("pretrained_model.zip", env=env)
model.learn(total_timesteps=100000)  # Fine-tune
```

### Custom CNN Architecture

Edit `train_ppo.py`:

```python
policy_kwargs=dict(
    features_extractor_kwargs=dict(features_dim=512),  # Larger feature dim
    net_arch=[dict(pi=[512, 512], vf=[512, 512])]     # Larger networks
)
```

---

## Next Steps

1. **Start with default parameters** - Train for 500k timesteps
2. **Monitor TensorBoard** - Check if learning is happening
3. **Test periodically** - Use evaluation callback results
4. **Tune hyperparameters** - Based on performance
5. **Adjust rewards** - Fine-tune reward function for desired behavior
6. **Experiment with architecture** - Try different CNN sizes

---

## References

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [AirSim Documentation](https://microsoft.github.io/AirSim/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## Tips for Success

✅ **Start small**: Train for 100k steps first to verify everything works
✅ **Monitor closely**: Use TensorBoard to catch issues early
✅ **Save often**: Checkpoints prevent losing progress
✅ **Test frequently**: Evaluate model every 10k-20k steps
✅ **Tune rewards**: Reward engineering is key to good behavior
✅ **Be patient**: RL training can take time to converge

Good luck with your training! 🚁🤖
