# Project Files Overview

## 📚 Documentation Files

### Core Documentation
- **README.md** - Project overview, setup instructions, quick start guide
- **TRAINING_EVOLUTION.md** - Complete development history and problem-solving journey
- **VERSION_COMPARISON.md** - Detailed comparison of V1, V2, and V3 environments
- **GOAL_AWARE_EXPLANATION.md** - Why goal awareness is critical for navigation
- **SPEED_OPTIMIZATION.md** - Training speed optimization techniques
- **PROJECT_FILES.md** (this file) - Overview of all project files

---

## 🎮 Environment Files

### V1 - Deprecated (Don't Use)
- **airsim_env.py** - Initial prototype, only uses camera images
  - ❌ No goal awareness
  - ❌ Memorizes specific routes
  - ❌ Can't generalize

### V2 - Limited (Use only for fixed environments)
- **airsim_env_v2.py** - Goal-aware but height-unaware
  - ✅ Knows where goal is
  - ✅ Can navigate to any goal
  - ❌ Assumes fixed height constraints
  - ⚠️ Won't adapt to different forests

### V3 - Production Ready (RECOMMENDED)
- **airsim_env_v3.py** - Complete adaptive navigation
  - ✅ Goal-aware navigation
  - ✅ Height-aware adaptation
  - ✅ Altitude maintenance rewards
  - ✅ Randomized start/goal positions
  - ✅ Guaranteed long trajectories (20-40m)
  - ✅ Works in any environment

---

## 🚀 Training Scripts

### Main Training
- **train_ppo_v3_quality.py** - Full quality training (RECOMMENDED)
  - 500k timesteps
  - 84x84 images
  - 256x256 networks
  - Time: 6-10 hours
  - Best for production deployment

- **train_ppo_v3_fast.py** - Fast training (reduced quality)
  - 300k timesteps
  - 64x64 images
  - Smaller batch sizes
  - Time: 1.5-3 hours
  - Good for quick iterations

### Resume Training
- **resume_training_v3.py** - Continue from checkpoints
  - Automatically finds latest checkpoint
  - Preserves all learned behavior
  - Can add more training beyond 500k
  - Useful after interruptions or for applying new reward functions

### Testing
- **test_ppo_v3.py** - Test trained models
  - Load any saved model
  - Test with random or fixed goals
  - Visualize drone behavior
  - Evaluate performance

---

## 🛠️ Utility Scripts

### Basic Control
- **main.py** - Manual drone control demo
  - Basic movement patterns
  - Camera capture examples
  - API usage demonstration
  - Useful for testing/debugging

### Launch Scripts
- **launch_headless.bat** - Launch AirSim
  - Originally tried headless mode (had shader issues)
  - Now launches normally (can minimize window)
  - Works with ClockSpeed optimization

---

## ⚙️ Configuration Files

### Python Environment
- **requirements.txt** - Python package dependencies
  - airsim
  - stable-baselines3
  - torch
  - tensorboard
  - gymnasium
  - opencv-python
  - numpy

### AirSim Settings
- **C:\Users\Dator\Documents\AirSim\settings.json** - AirSim configuration
  - SimMode: Multirotor
  - ClockSpeed: 5.0 (5x faster!)
  - Recording: Disabled

---

## 📁 Generated Directories

### Models
- **models_v3_quality/** - Quality training output
  - `checkpoints/` - Every 10k steps
  - `best_model/` - Best performing model (updated every 5k steps)
  - `ppo_quality_final.zip` - Final trained model
  - `ppo_quality_interrupted.zip` - Auto-saved if training stopped

- **models_v3_fast/** - Fast training output
  - Same structure as quality models

### Logs
- **logs_v3_quality/** - Quality training logs
  - `tensorboard/` - TensorBoard metrics
  - `eval/` - Evaluation results

- **logs_v3_fast/** - Fast training logs
  - Same structure as quality logs

---

## 📊 What Gets Created During Training

### Checkpoints (Every 10,000 steps)
```
models_v3_quality/checkpoints/
  ppo_quality_10000_steps.zip
  ppo_quality_20000_steps.zip
  ppo_quality_30000_steps.zip
  ...
```

### Best Model (Evaluated every 5,000 steps)
```
models_v3_quality/best_model/
  best_model.zip         # Best so far
  evaluations.npz        # Performance metrics
```

### TensorBoard Logs
```
logs_v3_quality/tensorboard/PPO_1/
  events.out.tfevents...  # Training metrics
```

### Evaluation Logs
```
logs_v3_quality/eval/
  evaluations.npz         # Eval results
  monitor.csv             # Episode data
```

---

## 🎯 Quick Reference

### Start New Training
```bash
python train_ppo_v3_quality.py
```

### Resume Training
```bash
python resume_training_v3.py
```

### Test Model
```bash
python test_ppo_v3.py --model ./models_v3_quality/best_model/best_model.zip
```

### Monitor Training
```bash
tensorboard --logdir=./logs_v3_quality/tensorboard/
```

---

## 🗺️ File Dependencies

```
train_ppo_v3_quality.py
  └── airsim_env_v3.py
      └── AirSim API

resume_training_v3.py
  └── airsim_env_v3.py
  └── models_v3_quality/checkpoints/*

test_ppo_v3.py
  └── airsim_env_v3.py
  └── models_v3_quality/best_model/best_model.zip
```

---

## 📖 Recommended Reading Order

1. **README.md** - Start here for project overview
2. **TRAINING_EVOLUTION.md** - Understand the development journey
3. **VERSION_COMPARISON.md** - See why V3 is necessary
4. **SPEED_OPTIMIZATION.md** - Optimize training time
5. **GOAL_AWARE_EXPLANATION.md** - Deep dive into goal awareness

---

## 🚀 For New Users

**Just want to train a model?**
1. Read README.md (Quick Start section)
2. Launch AirSim: `Blocks.exe`
3. Run: `python train_ppo_v3_quality.py`
4. Wait 6-10 hours
5. Test: `python test_ppo_v3.py --model ./models_v3_quality/best_model/best_model.zip`

**Want to understand why things work this way?**
1. Read TRAINING_EVOLUTION.md
2. See how each problem was discovered and solved
3. Understand the design decisions

**Want to deploy in real world?**
1. Read VERSION_COMPARISON.md (Real-World Deployment section)
2. See example deployment code in README.md
3. Understand what sensor data you need

---

## 💡 Key Insights

### Why V3?
- **V1**: Memorizes routes (useless)
- **V2**: Can't adapt to different height constraints
- **V3**: Fully adaptive, production-ready

### Why Altitude Maintenance?
- Without it: Drone takes diagonal shortcuts
- With it: Efficient level flight patterns

### Why Long Trajectories?
- Short paths (5-15m): Often clear, learns shortcuts
- Long paths (20-40m): MUST navigate obstacles, learns properly

### Why ClockSpeed=5?
- 5x faster training with no quality loss
- Physics still accurate
- Alternative to problematic headless mode

---

## 🔧 Troubleshooting

### Can't find checkpoint
- Look in: `models_v3_quality/checkpoints/`
- resume_training_v3.py finds latest automatically

### TensorBoard shows no data
- Wait for first 2048 steps (n_steps parameter)
- Refresh browser
- Check logs exist: `logs_v3_quality/tensorboard/PPO_1/`

### Training very slow
- Check ClockSpeed in settings.json (should be 5.0)
- Check GPU usage (should use CUDA if available)
- Minimize AirSim window

### Shader library errors
- Don't use -RenderOffScreen flag
- Launch AirSim normally
- Use ClockSpeed optimization instead

---

## 📝 Version History

- **V1**: Initial prototype (camera only)
- **V2**: Added goal awareness
- **V3 Initial**: Added height awareness
- **V3 + Altitude**: Added cruising altitude rewards
- **V3 Final**: Added long trajectory requirements

Current version: **V3 Final** (Production Ready)
