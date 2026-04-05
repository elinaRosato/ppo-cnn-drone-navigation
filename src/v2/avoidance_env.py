"""
Simplified Obstacle Avoidance Environment

Navigation is handled by a simple controller (go toward goal).
The RL model ONLY learns obstacle avoidance from the camera image.

Controller: flies forward toward goal at episode_speed (randomised each reset)
RL Model:   sees 4 stacked grayscale frames, outputs lateral correction only
Combined:   velocity = forward_toward_goal + lateral_perpendicular
Altitude:   held by a P-controller (not learned) — vertical output removed from action space
Yaw:        faces actual movement direction (camera sees what's ahead)

Observation: (4, 128, 128) uint8 — 4 stacked grayscale frames, channels-first (CHW)
             No VecTransposeImage needed; SB3 CnnPolicy detects CHW automatically.
Depth image: requested alongside RGB in the same API call, used only for the
             soft proximity penalty reward (not fed to the model).

ROS2 bridge: if a ROS2CameraBridge instance is passed as ros2_bridge, RGB frames
             are consumed from the subscriber at ~30 Hz instead of via simGetImages.
             Depth is still fetched from the AirSim Python API (one call per step).
"""

import math
import os
import random
import time
import gymnasium as gym
from gymnasium import spaces
import cosysairsim as airsim
import numpy as np
import cv2


# Substring matched against simListSceneObjects() to identify tree actors.
# All tree actors must have Movable mobility in Unreal (Details → Transform → ⚡).
TREE_ACTOR_FILTER = 'StaticMeshActor_UAID_'


class ObstacleAvoidanceEnv(gym.Env):
    """RL model learns to avoid obstacles while a simple controller navigates."""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 goal_distance_range=(50, 50),
                 cruising_altitude=-1.3,
                 speed_range=(0.5, 1.5),
                 lateral_scale=1.0,
                 goal_radius=2.0,
                 max_steps=2000,
                 step_hz=30.0,
                 show_visual_marker=False,
                 ros2_bridge=None):
        super().__init__()

        self.goal_distance_range = goal_distance_range
        self.cruising_altitude = cruising_altitude
        self.speed_range = speed_range
        self.lateral_scale = lateral_scale
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.step_hz = step_hz
        self.show_visual_marker = show_visual_marker
        self.ros2_bridge = ros2_bridge

        self.img_height = 128
        self.img_width = 128
        self.stack_frames = 4

        # Soft proximity penalty threshold (metres)
        self.prox_threshold = 3.5

        # Connect to AirSim.
        # When running from WSL2, set AIRSIM_HOST to the Windows host IP
        # (e.g. export AIRSIM_HOST=172.x.x.x).  Defaults to localhost.
        airsim_host = os.environ.get('AIRSIM_HOST', '')
        self.client = airsim.MultirotorClient(ip=airsim_host) if airsim_host else airsim.MultirotorClient()
        self.client.confirmConnection()
        host_str = airsim_host if airsim_host else 'localhost'
        print(f"Connected to AirSim at {host_str}!")
        # Brief pause to let AirSim finish registering all physics actors,
        # especially important right after a fresh Unreal Editor session.
        time.sleep(2.0)

        self.camera_name = "front_center"
        self.goal_pos = None
        self.episode_speed = float(np.mean(speed_range))  # placeholder until first reset

        # Discover tree actors by FName from simListSceneObjects.
        # Requires all tree actors to have Movable mobility in Unreal.
        all_objects = self.client.simListSceneObjects()
        self.tree_names = [o for o in all_objects if TREE_ACTOR_FILTER in o]
        print(f"Found {len(self.tree_names)} tree actors in scene.")
        if len(self.tree_names) == 0:
            print("  WARNING: no trees found. Check TREE_ACTOR_FILTER matches your actor names.")

        # Sample ground z from the first tree's current position.
        # The ground in this level is not at z=0 in world coordinates.
        self.ground_z = 0.0
        if self.tree_names:
            sample_pose = self.client.simGetObjectPose(self.tree_names[0])
            self.ground_z = sample_pose.position.z_val
            print(f"Ground z sampled from tree: {self.ground_z:.2f}")

        # Observation: 4 stacked grayscale frames in CHW format.
        # SB3 CnnPolicy expects (C, H, W); first dim (4) < spatial dims (84)
        # so SB3 detects this as already channels-first and skips transposition.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.stack_frames, self.img_height, self.img_width),
            dtype=np.uint8
        )

        # Action: lateral correction only.
        # Vertical is handled by the altitude hold P-controller (not learned).
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Forest density curriculum (advanced by ValidationCallback).
        # 0 = sparse only, 1 = sparse + medium, 2 = all densities
        self.density_stage = 0

        self.frame_stack = np.zeros(
            (self.stack_frames, self.img_height, self.img_width), dtype=np.uint8
        )
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0

    def _place_trees(self, goal_x, goal_y):
        """Reposition ALL trees each episode in a density-scaled ellipse.

        All trees are always placed — no parking needed, which avoids Unreal's
        world-partition streaming problem (trees parked far away/underground
        become inaccessible via simSetObjectPose in subsequent episodes).

        The ellipse semi-axes scale so that all len(tree_names) trees fit at
        the chosen min_dist spacing:
          Stage 0: sparse only  (15 m → large ellipse)
          Stage 1: sparse + medium (15 m or 10 m)
          Stage 2: sparse + medium + dense (6 m → compact ellipse)

        Dense forests are compact; sparse forests cover a wider area.
        """
        ALL_DENSITY_CONFIGS = [
            ('sparse', 8.0),
            ('medium', 6.5),
            ('dense',   5.0),
        ]
        active_configs = ALL_DENSITY_CONFIGS[:self.density_stage + 1]
        label, min_dist = random.choice(active_configs)

        n_trees = len(self.tree_names)
        DRONE_CLEARANCE = 4.0   # min distance from drone spawn at (0, 0)
        GOAL_CLEARANCE  = 4.0   # min distance from goal position

        # Base ellipse with drone start and goal as focal points.
        goal_distance = math.sqrt(goal_x ** 2 + goal_y ** 2)
        extension = 10.0
        c = goal_distance / 2
        base_a = c + extension
        base_b = math.sqrt(max(base_a ** 2 - c ** 2, 1.0))

        # Scale semi-axes so all n_trees fit at min_dist spacing.
        # Random packing efficiency ≈ 0.55 (conservative to keep rejection rate low).
        required_area = n_trees * (min_dist ** 2) / 0.55
        base_area = math.pi * base_a * base_b
        scale = max(1.0, math.sqrt(required_area / base_area))
        a = base_a * scale
        b = base_b * scale

        goal_angle = math.atan2(goal_y, goal_x)
        cos_a, sin_a = math.cos(goal_angle), math.sin(goal_angle)
        cx, cy = goal_x / 2.0, goal_y / 2.0

        # Rejection sampling — place ALL trees within the scaled ellipse
        placed = []
        max_attempts = n_trees * 300
        attempts = 0
        while len(placed) < n_trees and attempts < max_attempts:
            attempts += 1
            r = math.sqrt(random.random())
            theta = random.uniform(0, 2 * math.pi)
            lx = r * a * math.cos(theta)
            ly = r * b * math.sin(theta)
            wx = cx + cos_a * lx - sin_a * ly
            wy = cy + sin_a * lx + cos_a * ly
            if math.sqrt(wx * wx + wy * wy) < DRONE_CLEARANCE:
                continue
            if math.sqrt((wx - goal_x) ** 2 + (wy - goal_y) ** 2) < GOAL_CLEARANCE:
                continue
            if all(math.sqrt((wx - px) ** 2 + (wy - py) ** 2) >= min_dist
                   for px, py in placed):
                placed.append((wx, wy))

        print(f"  Forest: {label} ({len(placed)}/{n_trees} trees, "
              f"min spacing {min_dist}m, ellipse {a:.0f}×{b:.0f}m)")

        ok = 0
        fail = 0
        for i, name in enumerate(self.tree_names):
            if i < len(placed):
                x, y = placed[i]
            else:
                # Rejection sampling exhausted — overlap remaining trees with last placed
                # (extremely rare with properly scaled ellipse)
                x, y = placed[-1] if placed else (cx, cy)

            # Random yaw rotation around vertical axis
            yaw = random.uniform(0, 2 * math.pi)
            pose = airsim.Pose(
                airsim.Vector3r(x, y, self.ground_z),
                airsim.Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            )

            # Random scale: xy between 0.7–1.3, z between 1.0–1.5
            scale_xy = random.uniform(0.7, 1.3)
            scale_z = random.uniform(1.0, 1.5)
            self.client.simSetObjectScale(name, airsim.Vector3r(scale_xy, scale_xy, scale_z))

            if self.client.simSetObjectPose(name, pose, teleport=True):
                ok += 1
            else:
                fail += 1
        print(f"  Tree placement: {ok} moved, {fail} failed"
              + (f" (first failed: {self.tree_names[ok]})" if fail > 0 else ""))

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise forward speed for this episode
        self.episode_speed = float(np.random.uniform(*self.speed_range))

        # Random goal position
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(*self.goal_distance_range)
        goal_x = dist * np.cos(angle)
        goal_y = dist * np.sin(angle)
        goal_z = self.cruising_altitude
        self.goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

        print(f"\nGoal: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f}) | "
              f"Speed: {self.episode_speed:.1f} m/s")

        try:
            with open("goal_position.txt", "w") as f:
                f.write(f"{goal_x} {goal_y} {goal_z}")
        except Exception:
            pass

        # Reset drone first so AirSim is in a clean state before we reposition trees.
        # Placing trees before reset() causes failures on episodes > 0 because AirSim
        # is still in an end-of-episode (collision/active) state when simSetObjectPose
        # is called. After reset(), AirSim is fully initialised and all actors accept
        # teleport commands reliably.
        self.client.reset()

        # Random sun position via time of day (must be after reset()).
        # move_sun=True is required — without it AirSim updates the clock but
        # does not reposition the sun in the scene.
        hour = random.randint(6, 19)
        minute = random.randint(0, 59)
        self.client.simSetTimeOfDay(
            True,
            start_datetime=f"2025-06-15 {hour:02d}:{minute:02d}:00",
            is_start_datetime_dst=True,
            celestial_clock_speed=1,
            update_interval_secs=0,
            move_sun=True
        )
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        # Give AirSim a moment to finish reinitialising actor physics after reset.
        # Without this, simSetObjectPose returns False for ~194 of the tree actors.
        time.sleep(1.0)

        # Reposition trees after the AirSim reset so all actors are in a clean state.
        if self.tree_names:
            self._place_trees(goal_x, goal_y)

        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self.cruising_altitude, 2.0).join()

        # Rotate to face the goal before capturing the first frame.
        # Without this, the frame stack is initialised with whatever direction
        # AirSim resets to (typically X+), so the first episode steps see a
        # camera view that doesn't match the actual direction of travel.
        goal_yaw = math.degrees(math.atan2(goal_y, goal_x))
        self.client.rotateToYawAsync(goal_yaw, timeout_sec=5.0).join()
        time.sleep(0.2)  # let angular velocity settle after rotation

        # AirSim sets has_collided=True when the drone spawns on the ground
        # after reset(). The drone is now airborne, so read collision once to
        # flush the stale flag before the first step() call checks it.
        self.client.simGetCollisionInfo()

        if self.show_visual_marker:
            self.client.simFlushPersistentMarkers()
            self.client.simPlotPoints(
                points=[airsim.Vector3r(float(goal_x), float(goal_y), float(goal_z))],
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                size=40,
                duration=-1,
                is_persistent=True
            )

        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0
        self.lateral_history = []
        self._ep_proximity_reward = 0.0
        self._ep_straight_bonus = 0.0
        self._ep_action_norm_penalty = 0.0

        # Fill the frame stack with the first captured frame (no motion yet,
        # but already facing the goal after rotateToYaw above)
        initial_gray, _ = self._get_images()
        for i in range(self.stack_frames):
            self.frame_stack[i] = initial_gray

        return self.frame_stack.copy(), {}

    def step(self, action):
        t_step_start = time.perf_counter()
        self.current_step += 1

        action = np.array(action, dtype=np.float32)

        t0 = time.perf_counter()
        position = self._get_position()
        t_get_pos1 = time.perf_counter() - t0

        # Controller: unit vector toward goal and perpendicular (left)
        dx = self.goal_pos[0] - position[0]
        dy = self.goal_pos[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)
        ux, uy = dx / dist_xy, dy / dist_xy
        px, py = -uy, ux

        # RL correction mapped to world frame (lateral only)
        lateral = float(action[0]) * self.lateral_scale
        self.lateral_history.append(float(action[0]))

        # Combined velocity: forward toward goal + lateral perpendicular
        vel_x = self.episode_speed * ux + lateral * px
        vel_y = self.episode_speed * uy + lateral * py
        speed_xy = math.sqrt(vel_x * vel_x + vel_y * vel_y)

        # Yaw faces actual movement direction so camera sees what's ahead
        yaw_deg = math.degrees(math.atan2(vel_y, vel_x))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        # Body frame: all horizontal speed is forward, no lateral slip
        body_vx = speed_xy
        body_vy = 0.0

        # Altitude hold P-controller (not learned)
        altitude_error = self.cruising_altitude - position[2]
        body_vz = float(np.clip(altitude_error * 0.2, -1.0, 1.0))

        t0 = time.perf_counter()
        self.client.moveByVelocityBodyFrameAsync(
            float(body_vx), float(body_vy), float(body_vz), 5.0,
            yaw_mode=yaw_mode
        )
        t_move_cmd = time.perf_counter() - t0

        # Single API call returns both RGB (for obs) and depth (for penalty)
        t0 = time.perf_counter()
        gray, depth = self._get_images()
        t_images = time.perf_counter() - t0

        self._update_frame_stack(gray)

        # Check collision
        t0 = time.perf_counter()
        collision = self.client.simGetCollisionInfo().has_collided
        t_collision = time.perf_counter() - t0

        if collision:
            self.collision_count += 1

        # Check goal
        t0 = time.perf_counter()
        new_position = self._get_position()
        t_get_pos2 = time.perf_counter() - t0

        new_distance = np.linalg.norm(new_position[:2] - self.goal_pos[:2])
        goal_reached = new_distance < self.goal_radius

        t_step_total = time.perf_counter() - t_step_start

        # Rate-limit to step_hz when using the ROS2 bridge so the training loop
        # matches real deployment frequency (default 30 Hz) and the frame stack
        # captures genuinely different frames rather than repeated cache reads.
        if self.ros2_bridge is not None and self.step_hz is not None:
            target = 1.0 / self.step_hz
            remaining = target - t_step_total
            if remaining > 0:
                time.sleep(remaining)
        t_step_wall = time.perf_counter() - t_step_start

        if self.current_step % 25 == 1:
            print(f"  Step {self.current_step}: dist={dist_xy:.1f}m "
                  f"speed={self.episode_speed:.1f}m/s "
                  f"lateral={action[0]:.2f}")
            print(f"  [TIMING ms] pos1={t_get_pos1*1000:.1f} "
                  f"move={t_move_cmd*1000:.1f} "
                  f"images={t_images*1000:.1f} "
                  f"collision={t_collision*1000:.1f} "
                  f"pos2={t_get_pos2*1000:.1f} "
                  f"compute={t_step_total*1000:.1f} "
                  f"(~{1/t_step_wall:.1f} Hz)")

        if goal_reached:
            print(f"  Goal reached! Steps: {self.current_step}")

        # --- Reward ---
        reward = 0.0

        if goal_reached:
            reward += 100.0

        if collision:
            reward -= 100.0

        # Soft proximity penalty: centre 50% of depth image.
        # Clamp to [0, prox_threshold] first — anything beyond the threshold
        # is background we don't care about, and this also guards against
        # NaN/inf that AirSim can return for open sky.
        center_depth = depth[
            self.img_height // 4: 3 * self.img_height // 4,
            self.img_width  // 4: 3 * self.img_width  // 4
        ]
        center_depth = np.clip(center_depth, 0.0, self.prox_threshold)
        min_depth_center = float(np.min(center_depth))
        if min_depth_center < self.prox_threshold:
            prox_r = -((self.prox_threshold - min_depth_center) / self.prox_threshold * 2.0)
            reward += prox_r
            self._ep_proximity_reward += prox_r
        else:
            straight_r = (1.0 - abs(float(action[0]))) * 0.05
            reward += straight_r
            self._ep_straight_bonus += straight_r

        action_norm_r = -(float(np.linalg.norm(action)) * 0.1)
        reward += action_norm_r
        self._ep_action_norm_penalty += action_norm_r

        self.episode_reward += reward

        # Termination
        terminated = goal_reached or collision
        truncated = self.current_step >= self.max_steps

        if terminated or truncated:
            reason = "GOAL" if goal_reached else ("COLLISION" if collision else "MAX STEPS")
            lat = np.array(self.lateral_history)
            avg_lat = float(np.mean(lat)) if len(lat) > 0 else 0.0
            avg_abs_lat = float(np.mean(np.abs(lat))) if len(lat) > 0 else 0.0
            print(f"Episode ended ({reason}) | Reward: {self.episode_reward:.2f} | "
                  f"Steps: {self.current_step} | Distance: {new_distance:.1f}m | "
                  f"Collisions: {self.collision_count} | "
                  f"Lateral avg={avg_lat:+.3f} abs={avg_abs_lat:.3f} | "
                  f"prox={self._ep_proximity_reward:.1f} "
                  f"straight={self._ep_straight_bonus:.1f} "
                  f"norm={self._ep_action_norm_penalty:.1f}")

        info = {
            'goal_reached': goal_reached,
            'collision': collision,
            'distance': new_distance,
            'collision_count': self.collision_count
        }
        if terminated or truncated:
            lat = np.array(self.lateral_history)
            info['avg_lateral'] = float(np.mean(lat)) if len(lat) > 0 else 0.0
            info['avg_abs_lateral'] = float(np.mean(np.abs(lat))) if len(lat) > 0 else 0.0
            info['ep_proximity_reward'] = self._ep_proximity_reward
            info['ep_straight_bonus'] = self._ep_straight_bonus
            info['ep_action_norm_penalty'] = self._ep_action_norm_penalty

        return self.frame_stack.copy(), reward, terminated, truncated, info

    def _get_images(self):
        """Get grayscale frame (observation) and depth (penalty).

        When ros2_bridge is set: both RGB and depth come from ROS2 topic
        caches (~30 Hz, no blocking API calls). Depth falls back to the
        AirSim Python API only if the bridge has not yet received a depth
        frame (e.g. depth_topic=None or bridge just started).

        Without bridge: both fetched in a single simGetImages call.
        """
        if self.ros2_bridge is not None:
            # RGB — reads from the cached frame (no network call)
            t0 = time.perf_counter()
            gray = self.ros2_bridge.get_latest_frame()
            t_rgb = time.perf_counter() - t0
            if gray is None:
                print("    [ROS2] WARNING: no RGB frame yet, using blank")
                gray = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

            # Depth — prefer ROS2 cache; fall back to AirSim API if not available
            t0 = time.perf_counter()
            depth = self.ros2_bridge.get_latest_depth()
            t_depth_ros2 = time.perf_counter() - t0
            t_depth_api = 0.0

            if depth is None:
                # Bridge has no depth yet (or depth_topic=None) — use API
                t0 = time.perf_counter()
                depth_resp = self.client.simGetImages([
                    airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False),
                ])
                t_depth_api = time.perf_counter() - t0

                if depth_resp and depth_resp[0].width > 0:
                    r = depth_resp[0]
                    depth = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
                    depth = cv2.resize(depth, (self.img_width, self.img_height))
                else:
                    depth = np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)

            if self.current_step % 25 == 1:
                src = "api" if t_depth_api > 0 else "ros2"
                t_depth = t_depth_api if t_depth_api > 0 else t_depth_ros2
                print(f"    [IMG-ROS2 ms] rgb_cache={t_rgb*1000:.1f} "
                      f"depth_{src}={t_depth*1000:.1f}")

            return gray, depth

        # No bridge: single API call for both
        t0 = time.perf_counter()
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False),
        ])
        t_api = time.perf_counter() - t0

        # --- RGB → grayscale ---
        if responses and responses[0].width > 0:
            r = responses[0]
            raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            # AirSim returns BGRA (4 ch) or BGR (3 ch) depending on platform/version
            n_ch = len(raw) // (r.width * r.height)
            img = raw.reshape(r.height, r.width, n_ch)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if n_ch == 4 else cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (self.img_width, self.img_height))

            if self.current_step % 100 == 1:
                print(f"    [IMG] gray shape={gray.shape} "
                      f"min={gray.min()} max={gray.max()} mean={gray.mean():.1f}")
        else:
            print("    [IMG] WARNING: empty RGB image!")
            gray = np.zeros((self.img_height, self.img_width), dtype=np.uint8)

        # --- Depth (raw metres, penalty only) ---
        if len(responses) > 1 and responses[1].width > 0:
            r = responses[1]
            depth = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
            depth = cv2.resize(depth, (self.img_width, self.img_height))
        else:
            depth = np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)

        if self.current_step % 25 == 1:
            print(f"    [IMG-API ms] simGetImages={t_api*1000:.1f}")

        return gray, depth

    def _update_frame_stack(self, new_frame):
        """Shift stack and insert the newest frame at the end (index -1)."""
        self.frame_stack = np.roll(self.frame_stack, shift=-1, axis=0)
        self.frame_stack[-1] = new_frame

    def _get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def close(self):
        self.client.simFlushPersistentMarkers()
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
