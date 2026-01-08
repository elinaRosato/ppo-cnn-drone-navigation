# Training Evolution: From Prototype to Production-Ready

## Complete History of Model Development

This document chronicles the evolution of our UAV navigation training system, documenting each problem discovered and solution implemented.

---

## Phase 1: Initial Prototype (V1) - The Problem

### Goal:
Train a drone to navigate from point A to point B using PPO with CNN.

### Initial Implementation:
```python
# V1: airsim_env.py
observation = camera_image  # Shape: (84, 84, 3)
goal_position = (15, 15, -2)  # FIXED
height_bounds = [-3.0, -1.5]  # FIXED
```

### Training Configuration:
- Policy: CnnPolicy (camera only)
- Goal: Always (15, 15, -2)
- Height constraints: Fixed [-3.0, -1.5]
- Environment: AirSim Blocks

### **Critical Discovery #1: Lack of Goal Awareness**

**Problem Identified:**
> "Does the drone know which direction it needs to go?"
> "If model gets reward for reaching endpoint in this environment, it will only learn that specific location"

**What Went Wrong:**
- Drone only saw camera images
- No information about WHERE the goal is
- No information about WHICH direction to fly
- Model memorized: "If I see these blocks, turn left, then go straight"
- Result: **Only works for one specific goal location!**

**Real-World Impact:**
```
Train: Goal at (15, 15, -2)
Deploy: Goal at (20, 10, -2.5)
Result: ❌ Drone flies to old memorized location (15, 15, -2)
```

### Lesson Learned:
**A navigation model MUST know where it's trying to go!**

---

## Phase 2: Goal-Aware Navigation (V2)

### Solution:
Add goal information to observations.

### Implementation:
```python
# V2: airsim_env_v2.py
observation = {
    'image': camera_image,      # (84, 84, 3)
    'vector': [
        goal_x, goal_y, goal_z, # Goal position
        distance_to_goal,       # Scalar distance
        yaw_to_goal            # Direction to face
    ]
}
```

### Changes:
- Policy: MultiInputPolicy (handles Dict observations)
- Goals: Randomized each episode
- Observation space: Combined image + vector

### Training Results:
✅ Model learned to navigate to different goal positions!
✅ Generalization improved significantly

### **Critical Discovery #2: Fixed Environment Constraints**

**Problem Identified:**
> "Can this transfer to different environments with different heights?"
> "If training only has one height constraint, model won't adapt"

**What Went Wrong:**
- Height bounds fixed at [-3.0, -1.5]
- Model learned: "Always fly at -2.0m altitude"
- In real forests: Tree canopy height varies!
- Result: **Won't work in different environments**

**Real-World Impact:**
```
Train: Fixed ceiling at -1.5m
Deploy: Forest with ceiling at -0.5m
Result: ❌ Drone flies into tree canopy (still flying at -2.0m)
```

---

## Phase 3: Height-Aware Navigation (V3 Initial)

### Solution:
Add height constraint information to observations and randomize them.

### Implementation:
```python
# V3: airsim_env_v3.py
observation = {
    'image': camera_image,
    'vector': [
        # Goal info
        goal_x, goal_y, goal_z,
        distance, yaw,
        # NEW: Height awareness
        current_height,
        max_safe_height,  # Ceiling (varies per episode!)
        min_safe_height,  # Floor (varies per episode!)
        distance_to_ceiling,
        distance_to_floor
    ]
}

# Randomized per episode
max_height = random.uniform(-1.0, -1.8)
min_height = random.uniform(-2.5, -4.0)
```

### Benefits:
✅ Model sees current height bounds in every observation
✅ Different bounds each episode
✅ Learns to adapt altitude to environment
✅ Can deploy to different forest/environment heights

### Training Configuration:
- 500k timesteps
- Goals randomized 15-30m from start
- Height bounds randomized each reset

---

## Phase 4: Reward Shaping for Altitude

### Reward Structure:
```python
# Distance progress
reward += (prev_dist - curr_dist) * 20.0

# Goal reached
if distance < 1.0:
    reward += 200.0

# Collision
if collision:
    reward -= 100.0
```

### Training Results (First 60k steps):
- Model learned basic navigation
- Successfully reached goals sometimes
- But behavior seemed suboptimal...

---

## Phase 5: Training Results Analysis (127k steps)

### Observations from Testing:
✅ **Successes:**
- Drone reaches goals
- Avoids some obstacles
- Adapts to randomized heights

❌ **Problems Discovered:**
1. **Diagonal flight pattern**: Goes up → forward → down
2. **Short, direct paths**: Minimal obstacle avoidance
3. **Inefficient**: Could be smoother/more direct

### Analysis:
The reward structure had no incentive for:
- Maintaining consistent altitude
- Level flight
- Efficient cruising behavior

**Real drones fly level at cruising altitude for efficiency!**

---

## Phase 6: Altitude Maintenance Reward (V3 Updated)

### Solution:
Add reward for flying at optimal cruising altitude.

### New Reward Components:
```python
# Optimal altitude = center of allowed range
target_altitude = (max_height + min_height) / 2.0

# Distance from optimal
deviation = abs(current_height - target_altitude)
normalized = deviation / (height_range / 2.0)

# Quadratic reward (peaked at center)
altitude_bonus = 5.0 * (1.0 - normalized**2)
reward += altitude_bonus
```

### Effect:
```
At ceiling (-1.5):  +0.0
At center (-2.5):   +5.0  ← Maximum!
At floor (-3.5):    +0.0
```

### Expected Behavior:
- Drone should prefer flying at center altitude
- Smoother, more level flight
- More efficient trajectories

---

## Phase 7: Challenging Trajectories (V3 Final)

### Problem:
Goals too close (5-20m) → drone takes shortest path, doesn't need advanced navigation

### Solution:
```python
# OLD
goal_range_x = (5, 20)
goal_range_y = (5, 20)
min_distance = 10  # Goals can be just 10m away

# NEW
goal_range_x = (15, 30)
goal_range_y = (15, 30)
min_distance = 20  # Force longer trajectories
```

### Rationale:
- Longer distances → more obstacles in path
- Forces drone to navigate AROUND obstacles
- Better tests obstacle avoidance
- More realistic scenarios

### Updated Training Target:
- 500k timesteps
- Goals 20-40m away
- Randomized heights
- Altitude maintenance reward

---

## Phase 8: The Camera Vision Problem (Critical Discovery at 160k steps)

### **The Shocking Discovery**

**Observation:**
User reported: "Even after 30k additional training (total 160k steps), drone still shows diagonal flight behavior - goes up, stays briefly, goes down."

### Investigation Results:

**The Fundamental Problem:**
The drone was **NOT using camera vision at all**. Instead, it discovered an exploit:

#### The Height Exploit Strategy:
```
1. Goal appears (anywhere in 3D space)
2. Drone: "Can I fly over it?"
   - Check: current_height vs max_height
   - If room above → FLY UP (above obstacles)
3. Drone flies toward goal at ceiling level
4. Drone: "Can I drop down to goal?"
   - Check: goal_height vs min_height
   - If room below → DROP DOWN
5. Reach goal ✓
```

**Result:** Camera input completely ignored!

### Why This Strategy Worked:

**Old environment allowed it:**
```python
# Height bounds were loose
max_height = random.uniform(-1.0, -1.8)  # 0.8m variation
min_height = random.uniform(-2.5, -4.0)  # 1.5m variation
# Total vertical space: 1.8m to 3.0m

# Only 0.3m goal padding
goal_z = random.uniform(min_height + 0.3, max_height - 0.3)
# Goals had 1.2m to 2.4m of vertical freedom
```

### Evidence in Logs:
```
Episode 1: Height bounds [-3.2, -1.2] (2.0m range)
  → Drone flies at -1.3m (near ceiling)
  → Drops to -2.8m near goal
  → SUCCESS without seeing obstacles!

Episode 2: Height bounds [-2.8, -1.5] (1.3m range)
  → Drone flies at -1.6m (ceiling)
  → Drops to -2.5m
  → SUCCESS without camera!
```

### The Real Problem:

**Camera vision was optional, not required!**

The model learned:
- ✅ Use height information (in observation vector)
- ✅ Fly at ceiling to avoid obstacles
- ✅ Drop to goal altitude when close
- ❌ Never look at camera (unnecessary!)

### Impact on Training:
- First 50k steps: Random exploration, some camera use
- 50k-100k: Discovered height exploit works better
- 100k-160k: Reinforced height exploitation, camera usage decreased
- Result: **Pure height-based navigation, zero camera reliance**

### Critical Insight:
> "When a model ignores an input (camera), it's not a training bug - it's that the environment allows a simpler strategy. You must make the complex strategy (camera vision) the ONLY viable option."

---

## Phase 9: Forcing Camera-Based Navigation (V3 Final Fix)

### The Solution: Make Camera Vision Mandatory

#### 1. Tight Height Constraints
```python
# BEFORE (allowed exploitation)
max_height = random.uniform(-1.0, -1.8)  # 0.8m variation
min_height = random.uniform(-2.5, -4.0)  # 1.5m variation
# Result: 1.8-3.0m vertical space

# AFTER (forces horizontal navigation)
max_height = random.uniform(-1.5, -1.7)  # 0.2m variation
min_height = random.uniform(-2.5, -2.7)  # 0.2m variation
# Result: ~1.0m vertical space (barely fits drone)
```

#### 2. Exponential Out-of-Bounds Penalty
```python
# Calculate violation magnitude
if z_pos > max_height:
    violation = z_pos - max_height
    normalized_violation = violation / (height_range / 2.0)

    # Cap to prevent overflow
    capped = min(normalized_violation, 3.0)

    # EXPONENTIAL penalty (gets MUCH worse)
    penalty = 50.0 * (exp(2.0 * capped) - 1.0)
    reward -= penalty

# Example penalties:
# 0.1m over: -10 penalty
# 0.3m over: -45 penalty
# 0.5m over: -130 penalty (episode ruined!)
# 1.0m over: -500+ penalty (worse than collision!)
```

#### 3. Vertical Movement Penalty
```python
# Penalize going up/down (encourages level flight)
vertical_change = abs(z_pos - previous_z)
reward -= vertical_change * 5.0

# Diagonal flight example:
# Going up 0.5m: -2.5 reward
# Going down 0.5m: -2.5 reward
# Total per up-down cycle: -5.0 reward loss
# Over 100 steps: -500 reward (like a collision!)
```

#### 4. Quadratic Altitude Bonus
```python
# Reward staying at CENTER of allowed range
target_height = (max_height + min_height) / 2.0
deviation = abs(z_pos - target_height)
normalized = deviation / (height_range / 2.0)

# Smooth quadratic function
altitude_bonus = 5.0 * (1.0 - normalized**2)

# Rewards by position:
# At ceiling: 0.0 (no bonus)
# At center:  +5.0 (maximum!)
# At floor:   0.0 (no bonus)
```

### Combined Effect:

**OLD Strategy (exploitation):**
```
1. Fly to ceiling (-1.5m)     → Gets past obstacles
2. Maintain ceiling altitude   → Safe from collisions
3. Drop to goal when close     → Success!
Total reward: ~+400 (200 goal + 200 progress - minimal penalties)
```

**NEW Strategy (required behavior):**
```
1. Try to fly to ceiling       → EXPONENTIAL PENALTY (-130)
2. Try to drop down           → VERTICAL MOVEMENT PENALTY (-15)
3. Violate bounds repeatedly  → MASSIVE PENALTIES (-500+)
Total reward: -645 (worse than crashing!)

ONLY VIABLE STRATEGY:
1. Stay at center altitude    → +5 bonus per step
2. Navigate horizontally      → Use camera to avoid obstacles
3. Minimal vertical movement  → Minimize penalties
Total reward: +500+ (goal) + 500 (progress) + 250 (altitude bonus)
```

### Updated Configuration:
```python
# Tight height constraints
height_bound_ranges = {
    'max_height': (-1.5, -1.7),  # Only 0.2m variation
    'min_height': (-2.5, -2.7)   # Only 0.2m variation
}
# Total: 1.0m vertical space (just enough for drone body)

# Goal placement within bounds
goal_z = random.uniform(min_height + 0.3, max_height - 0.3)
# Goals have only 0.4m vertical range

# Reward structure
rewards = {
    'progress': +20.0 per meter,
    'altitude_bonus': +0 to +5.0 (quadratic),
    'vertical_movement': -5.0 per meter,
    'goal_reached': +200.0,
    'collision': -100.0,
    'out_of_bounds': -50 * (e^(2x) - 1)  # Exponential!
}
```

### Expected Results:

**The drone MUST:**
- ✅ Use camera to detect obstacles
- ✅ Navigate horizontally around them
- ✅ Maintain center altitude (can't fly over/under)
- ✅ Make smooth, level flight adjustments

**The drone CANNOT:**
- ❌ Fly above obstacles (too close to ceiling)
- ❌ Fly below obstacles (too close to floor)
- ❌ Use diagonal flight (too expensive vertically)
- ❌ Ignore camera (only horizontal path works)

### Training Implications:

**First 10k steps:**
- Drone explores randomly
- Discovers: "Going up/down = bad"
- Learns: "Stay at center altitude = good"

**10k-50k steps:**
- Tries various horizontal paths
- Crashes into obstacles → learns to avoid via camera
- Discovers: "Camera shows obstacles before I hit them!"

**50k-200k steps:**
- Refines camera-based obstacle detection
- Learns optimal paths around obstacles
- Improves horizontal navigation efficiency

**Result:** True camera-based navigation system!

---

## Phase 10: Baby Steps Curriculum & Training Optimizations

### **Critical Discovery: The Random Initialization Problem**

**Observation at Training Start:**
- Drone takes off, hovers briefly
- Immediately lands/crashes at starting position
- Episodes last only 10-20 steps
- No learning progress after thousands of steps

### Root Cause Analysis:

#### Problem 1: Goals Too Far for Untrained Model
```python
# Previous curriculum
curriculum_start_distance = 10.0m  # Still too hard!

# Untrained random network:
# - Never reaches 10m goals
# - Never gets +500 goal reward
# - Never learns "move toward goal = good"
# Result: Learned helplessness
```

#### Problem 2: Instant Crashing from Random Actions
```python
# Random network outputs action[2] = +0.8
vz = action[2] * 1.5  # = +1.2 m/s downward
# Result: Hits ground in 2 seconds, crashes before learning
```

#### Problem 3: Low Exploration
```python
ent_coef = 0.01  # Default
# Result: Conservative actions, doesn't try diverse behaviors
```

#### Problem 4: Impossible Ground Goals
```python
# 50% of goals spawned at Z=0.0 (ground level)
# Allowed flight range: -2.6m to -1.1m
# Result: Goals 1.1m ABOVE ceiling! Impossible to reach!
```

### Solutions Implemented:

#### 1. Ultra-Easy Curriculum (Baby Steps)
```python
# train_ppo_v3_baby_steps.py
curriculum_start_distance = 3.0m   # ULTRA EASY (was 10m)
curriculum_end_distance = 35.0m    # Full difficulty
curriculum_timesteps = 300000      # Slower progression (was 200k)
goal_radius = 3.0m                 # Larger success zone (was 2.0m)

# Progression:
# Step 0-50k:   3-8m goals (can reach with random movement!)
# Step 50-100k: 8-13m goals
# Step 100-150k: 13-18m goals
# Step 150-200k: 18-23m goals
# Step 200-250k: 23-28m goals
# Step 250-300k: 28-33m goals
# Step 300k+:   33-35m goals
```

#### 2. Reduced Vertical Velocity
```python
# BEFORE
vz = action[2] * 1.5  # ±1.5 m/s - too fast, instant crashes

# AFTER
vz = action[2] * 0.3  # ±0.3 m/s - 5x slower
# Gives drone ~7 seconds to learn before hitting ground
# Horizontal still fast: ±3.0 m/s for efficient navigation
```

#### 3. 10x Increased Exploration
```python
# BEFORE
ent_coef = 0.01  # Low exploration, conservative actions

# AFTER
ent_coef = 0.1   # 10x higher - much more diverse behavior
# Drone tries many different actions → discovers what works
```

#### 4. Optimal Starting Altitude
```python
# Drone starts at CENTER of allowed range
target_altitude = (max_height + min_height) / 2.0
# e.g., (-1.1 + -2.6) / 2 = -1.85m

# Benefits:
# - Immediately receives +5.0 altitude bonus
# - Learns "this height = good" from step 1
# - Equals distance from ceiling/floor boundaries
```

#### 5. Forced Hover Stabilization
```python
# First 10 steps of EVERY episode:
if current_step <= 10:
    vz = 0.0  # Force hover, no vertical movement

# Benefits:
# - Drone stabilizes after takeoff
# - Can't crash in first 10 steps (guaranteed survival)
# - Has time to start moving toward 3m goal
# - Learns hovering behavior is rewarded
```

#### 6. Survival Rewards
```python
# Reward just for staying airborne
if min_height <= z_pos <= max_height:
    reward += 0.5  # Per step in bounds

# Reduced step penalty
reward -= 0.1  # (was -0.5)

# Net effect: +0.4 per step alive (was -0.5)
# Staying alive > crashing quickly
```

#### 7. Fixed Goal Placement (No Ground Goals!)
```python
# BEFORE (BUG!)
if random() < 0.5:
    goal_z = 0.0  # Ground level (OUTSIDE flight zone!)
else:
    goal_z = uniform(min_height + 0.3, max_height - 0.3)

# AFTER (FIXED!)
goal_z = uniform(min_height + 0.2, max_height - 0.2)
# ALL goals within reachable flight zone (-2.4m to -0.9m)
# Increased from 1.0m to 1.5m total altitude range
# Goals have 1.1m vertical space (was 0.4m)
```

#### 8. Increased Altitude Range
```python
# BEFORE (too tight, caused wiggle violations)
max_height = (-1.5, -1.7)  # Range: 1.0m
min_height = (-2.5, -2.7)

# AFTER (allows natural drone movement)
max_height = (-1.0, -1.2)  # Range: 1.3-1.7m
min_height = (-2.5, -2.7)  # Average: ~1.5m

# Benefits:
# - Natural drone wobbling doesn't trigger penalties
# - Still tight enough to prevent height exploitation
# - Extreme ceiling at -4.1m (terminates if exceeded)
```

#### 9. Fixed Extreme Altitude Check (NED Coordinates)
```python
# BEFORE (WRONG for NED!)
extreme_max = max_height + height_range
if z_pos > extreme_max:  # Only catches underground!

# AFTER (CORRECT for NED)
# In NED: more negative Z = higher altitude
extreme_ceiling = min_height - height_range  # e.g., -4.1m
if z_pos < extreme_ceiling:  # Catches flying too high
    terminate_episode()
    reward -= 300.0
```

#### 10. Visual Flight Corridor
```python
# Three guide lines from origin to goal XY:
# 1. Green line (center altitude) - optimal path
# 2. Cyan line (ceiling) - upper boundary
# 3. Orange line (floor) - lower boundary

# Plus: Red sphere (goal) + Red circle (3m radius)

# Benefits:
# - Visual feedback on drone's altitude compliance
# - Easy to see if drone is using height exploit
# - Clear target corridor for navigation
```

#### 11. Forward Velocity Rewards
```python
# Calculate movement direction
movement_vector = current_pos - previous_pos
goal_direction = normalize(goal_pos - current_pos)
forward_velocity = dot(movement_vector, goal_direction)

# Reward moving toward goal (where camera sees)
if forward_velocity > 0:
    reward += forward_velocity * 5.0  # Moving forward
else:
    reward += forward_velocity * 10.0  # 2x penalty moving backward

# Benefits:
# - Encourages drone to face movement direction
# - Camera becomes essential (points forward)
# - Penalizes flying backward blind
# - More natural flight behavior
```

#### 12. Camera Verification System
```python
# On startup, verify camera orientation
camera_info = client.simGetCameraInfo("front_center")
qw, qx, qy, qz = camera_info.pose.orientation

if abs(qy) > 0.5:
    print("[WARNING] Camera tilted downward!")
    print("Camera should face FORWARD for navigation")

# Detects common misconfiguration (downward camera)
# Ensures camera is useful for obstacle avoidance
```

### Combined Training Improvements:

**Episode Start Sequence:**
```
1. Drone spawns at origin (0, 0)
2. Takes off to optimal altitude (-1.85m)
3. Hovers and stabilizes for 0.5s
4. Visual corridor appears (green/cyan/orange lines)
5. Red goal sphere appears (3-8m away)
6. First 10 steps: Forced hover (vz=0)
   - Receives +0.5 survival reward
   - Receives +5.0 altitude bonus (at optimal height)
   - Starts moving horizontally with exploration
7. Step 11+: Normal control
   - Vertical: ±0.3 m/s (safe)
   - Horizontal: ±3.0 m/s (efficient)
```

**Expected Learning Curve:**
```
Steps 0-1k:
- Random exploration, occasional 3m goal reached
- Learns: staying alive = good (+0.4/step)
- Learns: reaching red sphere = GREAT (+500)

Steps 1k-10k:
- Discovers: moving toward red = reward
- Learns: center altitude = bonus
- Starts using horizontal movement

Steps 10k-50k:
- Consistently reaches 3-8m goals
- Curriculum gradually increases to 8-13m
- Learns basic obstacle avoidance with camera

Steps 50k-150k:
- Goals 8-18m, obstacles in path
- Camera vision becomes essential
- Learns to navigate around obstacles

Steps 150k-300k:
- Goals 18-35m, full difficulty
- Expert camera-based navigation
- Smooth, efficient flight paths
```

### Rescue Training Script:
```python
# rescue_training.py
# For models that started training but got stuck

# 1. Loads existing checkpoint (keeps learned weights)
# 2. Resets curriculum to 0 (starts with 3m goals)
# 3. Gives model chance to learn basics

# Use when:
# - Drone never reached goals in previous training
# - Want to keep partial learning, add easy curriculum
```

### Updated Training Recommendations:

**Fresh Training (Recommended):**
```bash
python train_ppo_v3_baby_steps.py
```

**Rescue Failed Training:**
```bash
python rescue_training.py
# Finds latest checkpoint automatically
# Restarts curriculum from 3m goals
```

### Key Metrics to Watch:

**Console Output:**
```
[RESET] Drone positioned at optimal altitude: -1.85m
[Episode] Goal: (4.2, 2.1, -1.92) [OPTIMAL]
[Episode] Distance from origin: 4.7m
[WARM-UP] Forcing hover for first 10 steps. Original action[2]=0.234
[SUCCESS] Goal reached! Distance: 2.1m < 3.0m | Steps: 45
```

**Warning Signs:**
```
[COLLISION] Episode ended - Crashed at step 12
[EXTREME ALTITUDE] Drone flew WAY too high: Z=-4.5m
[DEBUG] Short episode at step 14: goal=False, collision=True
[WARNING] Camera tilted downward! (Fix settings.json)
```

### Performance Expectations:

**First Hour (0-10k steps):**
- 20-30% success rate on 3-5m goals
- Episodes lasting 50-200 steps
- Occasional crashes, learning stabilization

**Hours 2-4 (10k-50k steps):**
- 60-70% success rate
- Goals increasing to 8-13m
- Clear improvement in navigation

**Hours 4-8 (50k-200k steps):**
- 70-80% success rate
- Goals 13-25m with obstacles
- Camera-based avoidance visible

**Hours 8-10 (200k-300k+ steps):**
- 80-90% success rate
- Full 35m navigation
- Expert-level performance

### Critical Success Factors:

1. ✅ **3m starting goals** - achievable with random exploration
2. ✅ **Slow vertical velocity** - prevents instant crashes
3. ✅ **10x exploration** - discovers working behaviors
4. ✅ **Optimal starting altitude** - immediate positive feedback
5. ✅ **Forced stabilization** - 10 steps guaranteed survival
6. ✅ **Survival rewards** - staying alive > crashing
7. ✅ **Reachable goals** - no impossible ground goals
8. ✅ **1.5m altitude range** - allows natural movement
9. ✅ **Forward velocity bonus** - encourages camera use
10. ✅ **Visual feedback** - corridor shows compliance

---

## Summary: Complete Evolution

This project evolved through critical discoveries:
1. Memorization problem (V1) - needed goal awareness
2. Fixed environment problem (V2) - needed height awareness
3. Diagonal flight pattern (Phase 6) - needed altitude maintenance
4. Short trajectories (Phase 7) - needed longer obstacle-filled paths
5. **Camera not being used (Phase 8)** - the fundamental flaw
6. **Exploiting height bounds (Phase 9)** - needed vertical penalties
7. **Training failure (Phase 10)** - needed baby steps curriculum
8. **Instant crashing (Phase 10)** - needed survival mechanisms
9. **Impossible goals (Phase 10)** - needed proper goal placement
10. **Camera orientation (Phase 10)** - needed verification system

**The key insights:**
- Reward shaping alone isn't enough - environment must force desired behavior
- When model ignores input, environment allows simpler strategy
- Untrained models need achievable early wins to bootstrap learning
- Vertical and horizontal navigation need different velocity scaling
- Camera vision must be REQUIRED, not optional

**The final configuration:**
- 1.5m altitude range (natural movement, not exploitable)
- Exponential out-of-bounds penalties (strict boundaries)
- 3m starting goals with 300k step curriculum (gradual difficulty)
- ±0.3 m/s vertical, ±3.0 m/s horizontal (safe + efficient)
- 10x exploration coefficient (discovers working strategies)
- Forced 10-step hover (guaranteed stabilization)
- Survival rewards (staying alive > crashing)
- Forward velocity rewards (camera-facing movement)
- Visual corridor feedback (green/cyan/orange guides)

**The model learns:**
- True obstacle avoidance using camera vision
- Level flight navigation at optimal cruising altitude
- Horizontal path planning around obstacles
- Goal-directed navigation with height adaptation
- Forward-facing movement for camera utilization

**Critical lesson:** When model doesn't learn expected behavior, check:
1. Is the desired behavior actually REQUIRED by the environment?
2. Can the model succeed with a simpler strategy?
3. Are initial tasks achievable for random/untrained agent?
4. Are there systematic failures preventing any learning?
5. Does the agent receive early positive feedback?
