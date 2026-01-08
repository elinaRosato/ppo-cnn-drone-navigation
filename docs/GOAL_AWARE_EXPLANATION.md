# Goal-Aware Navigation: Why V2 is Better

## The Problem with V1

### What the drone could see:
```python
observation = camera_image  # Only (84, 84, 3) RGB image
```

### What the drone COULDN'T see:
- Where is the goal?
- Which direction should I go?
- How far is the goal?
- Am I getting closer?

### Result:
The drone would only learn to navigate to the **specific fixed position (15, 15, -2)** in the Blocks environment. It couldn't generalize to:
- Different goal positions
- Different environments (forests, etc.)
- Dynamic navigation tasks

---

## The Solution: V2 with Goal Awareness

### Multi-Input Observation Space

```python
observation = {
    'image': camera_image,      # (84, 84, 3) - For obstacle detection
    'vector': [                 # (7,) - For navigation
        rel_x,                  # Relative X to goal
        rel_y,                  # Relative Y to goal
        rel_z,                  # Relative Z to goal
        distance,               # Euclidean distance to goal
        yaw_to_goal,            # Direction angle to goal
        current_height,         # Current Z position
        goal_height             # Target Z position
    ]
}
```

### What the drone NOW knows:
- **Exact direction** to the goal (yaw_to_goal)
- **Distance** to the goal
- **3D relative position** (where the goal is relative to me)
- **Height information** (current and target)
- **Visual information** (camera for obstacle avoidance)

---

## How It Works

### 1. **CNN Branch** (Image Processing)
```
Camera Image → Conv Layers → Features
                                ↓
                    "There's an obstacle ahead!"
```

### 2. **MLP Branch** (Goal Processing)
```
Goal Vector → Dense Layers → Features
                               ↓
                   "Goal is 10m ahead, slightly right"
```

### 3. **Combined Decision**
```
Image Features + Goal Features → Policy Network → Action
                                                    ↓
                            "Go forward-right, avoid obstacle!"
```

---

## Key Improvements in V2

### 1. **Randomized Goals Each Episode**
```python
# Every episode, new random goal:
goal_x = random(5, 25)
goal_y = random(5, 25)
goal_z = random(-3, -1.5)
```

**Benefits:**
- Drone learns to navigate to ANY position
- Doesn't memorize specific routes
- Generalizes to new environments
- Works in forests, cities, etc.

### 2. **Goal-Oriented Reward Function**
```python
# Progress toward goal
reward += (previous_distance - current_distance) * 20

# Distance penalty
reward -= distance * 0.1

# Goal reached bonus
if distance < 1.0:
    reward += 200
```

**Benefits:**
- Always knows if it's getting closer to goal
- Works for ANY goal position
- Encourages efficient paths

### 3. **MultiInputPolicy**
```python
PPO("MultiInputPolicy", ...)  # Instead of "CnnPolicy"
```

**Benefits:**
- Specialized CNN for visual processing
- Specialized MLP for goal processing
- Better performance than concatenating everything

---

## Training Workflow

### V1 (Old - Bad):
```
Episode 1: Try to reach (15, 15, -2) in Blocks
Episode 2: Try to reach (15, 15, -2) in Blocks
Episode 3: Try to reach (15, 15, -2) in Blocks
...

Result: Only works for that specific goal in that specific environment
```

### V2 (New - Good):
```
Episode 1: Try to reach (8, 12, -2.1) in Blocks
Episode 2: Try to reach (18, 7, -2.5) in Blocks
Episode 3: Try to reach (22, 15, -1.8) in Blocks
...

Result: Learns general navigation skills!
```

---

## Forest Deployment

### Design Goal:
Enable navigation where points A and B change dynamically. The drone should adapt to different environments and navigate to any endpoint.

### V2 Solution:

1. **During Training (Blocks environment):**
   - Learn with 1000+ different random goals
   - Learn obstacle avoidance from camera
   - Learn to follow goal direction

2. **During Deployment (Forest environment):**
   - Same observation format (image + goal vector)
   - Camera shows different obstacles (trees instead of blocks)
   - Goal vector tells where to go
   - **Drone uses learned skills to navigate!**

### Transfer to Forest:

```python
# 1. Train in Blocks with random goals
python train_ppo_v2.py  # 500k steps

# 2. Test in Blocks with new random goals
python test_ppo_v2.py   # Should work!

# 3. Transfer to Forest (optional fine-tuning)
# Launch forest environment instead of Blocks
# Model already knows:
#   - How to read goal vector
#   - How to avoid obstacles from camera
#   - How to navigate efficiently

# 4. (Optional) Fine-tune in forest
model = PPO.load("ppo_goal_aware_final.zip")
model.learn(100_000)  # Quick adaptation to forest
```

---

## Comparison Table

| Feature | V1 (Old) | V2 (New) |
|---------|----------|----------|
| **Knows goal location?** | No | Yes |
| **Randomized goals?** | No | Yes |
| **Generalizes to new environments?** | No | Yes |
| **Works in forest?** | No | Yes |
| **Observation** | Image only | Image + Goal vector |
| **Policy** | CnnPolicy | MultiInputPolicy |
| **Can navigate to ANY point?** | No | Yes |

---

## Quick Start with V2

### Training:
```bash
# Start AirSim Blocks
"C:\Users\Dator\Documents\AirSim\Unreal\Environments\Blocks\Binaries\Win64\Blocks.exe" -windowed

# Train with randomized goals
python train_ppo_v2.py
```

### Testing:
```bash
# Test with random goals
python test_ppo_v2.py --model ./models_v2/best_model/best_model.zip --episodes 10

# Test with fixed goal
python test_ppo_v2.py --fixed-goal --episodes 5
```

### Monitor:
```bash
tensorboard --logdir=./logs_v2/tensorboard/
```

---

## Key Takeaway

**V1**: Drone blindly moves based on camera → **Only works for specific memorized routes**

**V2**: Drone sees obstacles (camera) + knows where to go (goal vector) → **True navigation ability**

The drone will now:
1. Know where the goal is
2. Avoid obstacles using camera
3. Work with ANY goal position
4. Transfer to forests/other environments
5. Navigate intelligently, not memorize
