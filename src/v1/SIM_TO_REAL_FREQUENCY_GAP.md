# Sim-to-Real Transfer: Camera Frequency Gap

## The Problem

A key challenge in transferring a simulation-trained obstacle avoidance model to a real drone is the mismatch between the **inference frequency in simulation** and the **target inference frequency in real life**.

### Real-World Target

In deployment, the goal is to run the full perception-to-action loop at **30 Hz**:

```
Depth Camera → Model Inference → Velocity Command → Pixhawk
     ↑___________________ 33ms ______________________|
```

At a cruise speed of **3 m/s**, the drone moves **10 cm per decision step**. This gives the model fine-grained control — it can detect an obstacle at 5m and have ~50 decision steps to react before impact.

### Simulation Reality

The AirSim Python API is not designed for high-frequency image capture. Each call to `simGetImages()` involves a round-trip over TCP from Python to the C++ simulation engine, including rendering. This overhead is typically **100–2000ms per frame** depending on hardware and image resolution.

| AirSim Capture Resolution | Approx. Capture Time | Effective Hz |
|---------------------------|----------------------|--------------|
| 256×256 (RGB + Depth)     | ~1000–2000ms         | <1 Hz        |
| 84×84 (RGB + Depth)       | ~150–250ms           | ~4–5 Hz      |

At 5 Hz with a drone flying at 3 m/s, the drone moves **60 cm per decision step** — 6× coarser than real deployment. At <1 Hz with 256×256 images, it moves **3–6 meters per step** — 30–60× coarser.

### Why This Matters

The model receives images as input. What it "sees" between consecutive frames — the **optical flow**, or how quickly obstacles grow and shift in the image — is determined by:

```
Scene change per step = drone_speed × time_between_frames
```

If the training scene changes at a very different rate than deployment, the model's learned timing and magnitude of corrections may not transfer.

| Setup | Speed | Hz | Scene change/step |
|-------|-------|----|-------------------|
| Real deployment | 3 m/s | 30 Hz | 10 cm |
| Sim at 84×84 | 3 m/s | 5 Hz | 60 cm |
| Sim at 256×256 | 5 m/s | <1 Hz | >300 cm |
| **Sim at 84×84 (matched)** | **0.5 m/s** | **5 Hz** | **~10 cm** |

Setting `base_speed = 0.5 m/s` with 84×84 captures at ~5 Hz produces the same scene-change-per-step as real 30 Hz / 3 m/s flight — but requires the simulated drone to fly unrealistically slowly.

---

## The Solution: Speed Domain Randomization

Instead of fixing a single speed that either matches the sim camera rate or the real deployment speed, we **randomize the base speed each episode** over a range that spans both concerns:

```python
# In ObstacleAvoidanceEnv.reset():
self.base_speed = np.random.uniform(1.0, 3.0)
```

### Why This Works

1. **At 1 m/s episodes**: the model sees slow optical flow (~20 cm/step at 5 Hz), closer to real 30 Hz deployment conditions.
2. **At 3 m/s episodes**: the model sees faster optical flow (~60 cm/step), learning to react under more time pressure.
3. **Across all speeds**: the model learns that the correct action depends on what it sees in the image, not the absolute speed — because the speed itself is not given as an observation.

This is a form of **domain randomization**, a standard technique in sim-to-real transfer. By training across a range of conditions, the model becomes robust to the exact conditions it will face at deployment.

### Expected Behavior at Deployment

When deployed at 3 m/s / 30 Hz (10 cm/step), the visual input changes more slowly than anything in training (minimum 20 cm/step at 1 m/s). The model will see a "slower" world than the fastest training episodes but still within the learned distribution. Avoidance corrections should generalize correctly since the model has learned direction of avoidance from visual cues, not speed-specific patterns.

---

## Reducing Simulation Latency

Beyond speed randomization, the simulation capture frequency itself can be improved by reducing the number of images requested per step.

### Removing Depth Capture

In the current training setup, each step requests **two images** in a single `simGetImages()` call: an RGB scene image (converted to grayscale for the model) and a depth image (used only for the soft proximity penalty reward). Removing the depth request reduces rendering work and API round-trip time:

| Images per step | Approx. capture time | Effective Hz |
|-----------------|----------------------|--------------|
| RGB + Depth, 256×256 | ~1000–2000ms | <1 Hz |
| RGB only, 256×256 | ~600–1200ms | ~1 Hz |
| RGB + Depth, 84×84 | ~150–250ms | ~4–5 Hz |
| **RGB only, 84×84** | **~100–150ms** | **~6–10 Hz** |

The depth image is **never fed to the model** — it is only used to compute a soft proximity penalty that penalises the drone for getting within 5m of an obstacle. This penalty provides a smoother reward gradient than collision alone, but is not essential: the hard collision penalty (`-10`) remains and provides the core avoidance signal.

Removing depth capture is a worthwhile latency reduction, especially combined with lower resolution:

```python
# _get_images() — RGB only, no depth
responses = self.client.simGetImages([
    airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
])
```

The proximity penalty section in `step()` is removed accordingly, and the reward simplifies to:

```python
reward = 0.0
if goal_reached:
    reward += 10.0
if collision:
    reward -= 10.0
reward -= float(np.linalg.norm(action)) * 0.05  # action penalty
```

### Combined Effect

Applying both recommendations — **RGB-only capture at 84×84** — gives the best achievable frequency with the AirSim Python API:

| Setup | Hz | Scene change at 1 m/s | Scene change at 3 m/s |
|-------|----|-----------------------|-----------------------|
| 256×256 RGB+Depth | <1 Hz | >1000 cm | >3000 cm |
| 84×84 RGB+Depth | ~5 Hz | ~20 cm | ~60 cm |
| **84×84 RGB only** | **~8 Hz** | **~12 cm** | **~37 cm** |
| Real (30 Hz, 3 m/s) | 30 Hz | — | 10 cm |

At 84×84 RGB-only and 1 m/s, the scene changes ~12 cm per step — very close to the real 10 cm target.

---

## Remaining Gap

Even with speed randomization, a residual sim-to-real gap remains:

- **AirSim renders synthetic environments** — lighting, textures, and depth noise differ from a real RealSense D435
- **Simulation physics** are idealized — no wind, perfect altitude hold, instantaneous yaw
- **Frame stacking** (4 frames) partially compensates for low frequency by encoding recent motion history

Future mitigations could include:
- Domain randomization of lighting and texture
- Adding synthetic depth noise matching the real camera's noise profile
- Training at higher frequency using AirSim's ROS2 bridge instead of the Python API
