"""
Obstacle Avoidance Environment — v3 (monocular depth estimation)

Navigation is handled by a simple controller (fly toward goal).
The RL model learns obstacle avoidance from an estimated depth stack
produced by Depth Anything V2 Small instead of the raw grayscale feed.

Key differences from v2:
  - Actor sees 3 stacked ESTIMATED DEPTH frames (float32 [0,1])
    instead of 4 stacked grayscale frames (uint8).  Higher = closer.
  - Depth Anything V2 Small is applied to each RGB frame captured from
    the scene camera before storing it in the frame stack.
  - RGB augmentation (noise, brightness, blur) is applied to the input
    frame BEFORE the depth estimator, so perturbations propagate
    naturally through the model rather than corrupting depth values.
  - CLAHE is removed — contrast normalisation is handled internally by
    Depth Anything V2.
  - Privileged critic observation: GT metric depth from the AirSim depth
    camera (same 15m-range scalar as v2), unchanged.
  - Deployed policy only requires an RGB camera and the depth estimator
    — no dedicated depth sensor needed.

Observation: Dict {
    "image":      (3, 192, 192) float32  — 3 stacked estimated depth frames [0,1]
    "state":      (2,) float32           — [speed_norm, lateral_offset_norm]
    "privileged": (1,) float32           — [min_depth_norm] — critic only
}

ROS2 bridge: if a ROS2CameraBridge instance is passed, RGB frames are
             consumed from the subscriber at ~30 Hz (no blocking API calls).
             GT depth falls back to the AirSim Python API if not available
             from the bridge.
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
                 ros2_bridge=None,
                 depth_estimator=None,
                 bg_tree_fraction=0.30):
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
        self.depth_estimator = depth_estimator
        self.bg_tree_fraction = bg_tree_fraction

        self.img_height = 192
        self.img_width  = 192
        self.stack_frames = 3   # 3 depth frames — temporal velocity cues

        # Soft proximity penalty threshold (metres) — reward signal
        self.prox_threshold = 3.5

        # Depth range for the privileged critic (metres).
        # Longer than prox_threshold so the critic can anticipate obstacles
        # before they trigger the penalty.
        self.critic_depth_range = 15.0

        # training_mode=True enables RGB augmentation and depth scale jitter.
        # Set to False during deterministic validation.
        self.training_mode = True

        # Path geometry — updated each reset
        self.spawn_pos    = np.zeros(2, dtype=np.float32)
        self.path_dir     = np.array([1.0, 0.0], dtype=np.float32)
        self.goal_distance = float(np.mean(goal_distance_range))

        airsim_host = os.environ.get('AIRSIM_HOST', '')
        self.client = (
            airsim.MultirotorClient(ip=airsim_host) if airsim_host
            else airsim.MultirotorClient()
        )
        self.client.confirmConnection()
        host_str = airsim_host if airsim_host else 'localhost'
        print(f"Connected to AirSim at {host_str}!")
        time.sleep(2.0)

        self.camera_name = "front_center"
        self.goal_pos = None
        self.episode_speed = float(np.mean(speed_range))

        all_objects = self.client.simListSceneObjects()
        self.tree_names = [o for o in all_objects if TREE_ACTOR_FILTER in o]
        print(f"Found {len(self.tree_names)} tree actors in scene.")
        if not self.tree_names:
            print("  WARNING: no trees found. Check TREE_ACTOR_FILTER.")

        self.ground_z = 0.0
        if self.tree_names:
            sample_pose = self.client.simGetObjectPose(self.tree_names[0])
            self.ground_z = sample_pose.position.z_val
            print(f"Ground z sampled from tree: {self.ground_z:.2f}")

        n_obstacle = max(1, round((1 - self.bg_tree_fraction) * len(self.tree_names)))
        n_bg = len(self.tree_names) - n_obstacle
        print(f"Tree split: {n_obstacle} obstacle, {n_bg} background "
              f"(bg_fraction={self.bg_tree_fraction:.0%})")

        # Observation space — image is float32 [0,1] depth frames (not uint8 grayscale)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.stack_frames, self.img_height, self.img_width),
                dtype=np.float32,
            ),
            "state": spaces.Box(
                low=-2.0, high=2.0,
                shape=(2,),
                dtype=np.float32,
            ),
            "privileged": spaces.Box(
                low=0.0, high=1.0,
                shape=(1,),
                dtype=np.float32,
            ),
        })

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Forest density curriculum (advanced by ValidationCallback).
        self.density_stage = 0

        self.frame_stack = np.zeros(
            (self.stack_frames, self.img_height, self.img_width), dtype=np.float32
        )
        self.current_step     = 0
        self.episode_reward   = 0.0
        self.collision_count  = 0

    # ── Tree placement ────────────────────────────────────────────────────────
    # (identical to v2 — no vision changes here)

    def _place_trees(self, goal_x, goal_y, n_trees):
        ALL_DENSITY_CONFIGS = [
            ('sparse', 8.0),
            ('medium', 6.5),
            ('dense',  5.0),
        ]
        active_configs = ALL_DENSITY_CONFIGS[:self.density_stage + 1]
        label, min_dist = random.choice(active_configs)

        DRONE_CLEARANCE = 4.0
        GOAL_CLEARANCE  = 4.0

        goal_distance = math.sqrt(goal_x ** 2 + goal_y ** 2)
        extension = 10.0
        c = goal_distance / 2
        base_a = c + extension
        base_b = math.sqrt(max(base_a ** 2 - c ** 2, 1.0))

        required_area = n_trees * (min_dist ** 2) / 0.55
        base_area = math.pi * base_a * base_b
        scale = max(1.0, math.sqrt(required_area / base_area))
        a = base_a * scale
        b = base_b * scale

        goal_angle = math.atan2(goal_y, goal_x)
        cos_a, sin_a = math.cos(goal_angle), math.sin(goal_angle)
        cx, cy = goal_x / 2.0, goal_y / 2.0

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

        ok = fail = 0
        for i in range(n_trees):
            name = self.tree_names[i]
            if i < len(placed):
                x, y = placed[i]
            else:
                x, y = placed[-1] if placed else (cx, cy)
            yaw = random.uniform(0, 2 * math.pi)
            pose = airsim.Pose(
                airsim.Vector3r(x, y, self.ground_z),
                airsim.Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            )
            scale_xy = random.uniform(0.7, 1.3)
            scale_z  = random.uniform(1.0, 1.5)
            self.client.simSetObjectScale(name, airsim.Vector3r(scale_xy, scale_xy, scale_z))
            if self.client.simSetObjectPose(name, pose, teleport=True):
                ok += 1
            else:
                fail += 1

        print(f"  Tree placement: {ok} moved, {fail} failed"
              + (f" (first failed: {self.tree_names[ok]})" if fail > 0 else ""))
        return placed

    def _place_background_trees(self, goal_x, goal_y, start_idx, obstacle_positions):
        bg_names = self.tree_names[start_idx:]
        if not bg_names:
            return

        goal_dist = math.sqrt(goal_x ** 2 + goal_y ** 2)
        fwd  = self.path_dir
        perp = np.array([-fwd[1], fwd[0]], dtype=np.float32)

        CORRIDOR_HALF = 5.0
        MAX_RADIUS    = goal_dist + 20.0
        MIN_GAP       = 5.0
        SPAWN_CLEAR   = 3.0

        all_blocked = list(obstacle_positions)
        placed_bg   = []
        placed = 0

        for name in bg_names:
            success = False
            for _ in range(60):
                angle = random.uniform(0, 2 * math.pi)
                r     = random.uniform(0.0, MAX_RADIUS)
                tx    = r * math.cos(angle)
                ty    = r * math.sin(angle)
                if math.sqrt(tx ** 2 + ty ** 2) < SPAWN_CLEAR:
                    continue
                fwd_proj = tx * fwd[0]  + ty * fwd[1]
                lat_dist = abs(tx * perp[0] + ty * perp[1])
                if lat_dist < CORRIDOR_HALF and 0.0 < fwd_proj < goal_dist:
                    continue
                if any(math.sqrt((tx - px) ** 2 + (ty - py) ** 2) < MIN_GAP
                       for px, py in all_blocked + placed_bg):
                    continue
                placed_bg.append((tx, ty))
                success = True
                break

            if success:
                x, y = placed_bg[-1]
            else:
                x, y = 9000.0 + placed * 10.0, 9000.0
                placed_bg.append((x, y))

            yaw = random.uniform(0, 2 * math.pi)
            pose = airsim.Pose(
                airsim.Vector3r(x, y, self.ground_z),
                airsim.Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            )
            sc = random.uniform(0.6, 1.4)
            self.client.simSetObjectScale(name, airsim.Vector3r(sc, sc, random.uniform(1.0, 1.5)))
            self.client.simSetObjectPose(name, pose, teleport=True)
            placed += 1

        print(f"  Background trees: {placed} placed outside corridor")

    # ── Vision ────────────────────────────────────────────────────────────────

    def _get_images(self):
        """Fetch RGB frame and GT depth from AirSim (or ROS2 bridge).

        Returns
        -------
        rgb : (H, W, 3) uint8  — RGB frame for DepthEstimator input
        gt_depth : (H, W) float32  — metric depth (metres) for privileged critic
        """
        if self.ros2_bridge is not None:
            t0 = time.perf_counter()
            rgb = self.ros2_bridge.get_latest_frame()
            t_rgb = time.perf_counter() - t0

            if rgb is None:
                print("    [ROS2] WARNING: no RGB frame yet, using blank")
                rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

            t0 = time.perf_counter()
            gt_depth = self.ros2_bridge.get_latest_depth()
            t_depth_ros2 = time.perf_counter() - t0
            t_depth_api  = 0.0

            if gt_depth is None:
                t0 = time.perf_counter()
                depth_resp = self.client.simGetImages([
                    airsim.ImageRequest(
                        self.camera_name, airsim.ImageType.DepthPerspective, True, False
                    ),
                ])
                t_depth_api = time.perf_counter() - t0
                if depth_resp and depth_resp[0].width > 0:
                    r = depth_resp[0]
                    gt_depth = airsim.list_to_2d_float_array(
                        r.image_data_float, r.width, r.height
                    )
                    gt_depth = cv2.resize(gt_depth, (self.img_width, self.img_height))
                else:
                    gt_depth = np.full(
                        (self.img_height, self.img_width), 100.0, dtype=np.float32
                    )

            if self.current_step % 25 == 1:
                src = "api" if t_depth_api > 0 else "ros2"
                t_d = t_depth_api if t_depth_api > 0 else t_depth_ros2
                print(f"    [IMG-ROS2 ms] rgb_cache={t_rgb*1000:.1f} "
                      f"depth_{src}={t_d*1000:.1f}")

        else:
            t0 = time.perf_counter()
            responses = self.client.simGetImages([
                airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
                airsim.ImageRequest(
                    self.camera_name, airsim.ImageType.DepthPerspective, True, False
                ),
            ])
            t_api = time.perf_counter() - t0

            if responses and responses[0].width > 0:
                r = responses[0]
                raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
                n_ch = len(raw) // (r.width * r.height)
                img = raw.reshape(r.height, r.width, n_ch)
                # Convert to RGB for DepthEstimator (which expects RGB channel order)
                if n_ch == 4:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                else:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb = cv2.resize(rgb, (self.img_width, self.img_height))
            else:
                print("    [IMG] WARNING: empty RGB image!")
                rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

            if len(responses) > 1 and responses[1].width > 0:
                r = responses[1]
                gt_depth = airsim.list_to_2d_float_array(
                    r.image_data_float, r.width, r.height
                )
                gt_depth = cv2.resize(gt_depth, (self.img_width, self.img_height))
            else:
                gt_depth = np.full(
                    (self.img_height, self.img_width), 100.0, dtype=np.float32
                )

            if self.current_step % 25 == 1:
                print(f"    [IMG-API ms] simGetImages={t_api*1000:.1f}")

        return rgb, gt_depth

    def _estimate_depth(self, rgb: np.ndarray) -> np.ndarray:
        """Run the depth estimator (or fall back to a zero frame if not set)."""
        if self.depth_estimator is not None:
            return self.depth_estimator.estimate(rgb, training=self.training_mode)
        # Fallback: no estimator — fill with zeros (should not happen in practice)
        return np.zeros((self.img_height, self.img_width), dtype=np.float32)

    def _update_frame_stack(self, new_frame: np.ndarray):
        """Shift stack and insert newest depth frame at index -1."""
        self.frame_stack = np.roll(self.frame_stack, shift=-1, axis=0)
        self.frame_stack[-1] = new_frame

    def _compute_state(self, position):
        vec  = position[:2] - self.spawn_pos
        perp = np.array([-self.path_dir[1], self.path_dir[0]], dtype=np.float32)
        lateral_offset      = float(np.dot(vec, perp))
        lateral_offset_norm = float(np.clip(
            lateral_offset / max(self.goal_distance, 1.0), -2.0, 2.0
        ))
        speed_norm = float(self.episode_speed / self.speed_range[1])
        return np.array([speed_norm, lateral_offset_norm], dtype=np.float32)

    def _get_position(self):
        state = self.client.getMultirotorState()
        pos   = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode_speed = float(np.random.uniform(*self.speed_range))

        angle  = np.random.uniform(0, 2 * np.pi)
        dist   = np.random.uniform(*self.goal_distance_range)
        goal_x = dist * np.cos(angle)
        goal_y = dist * np.sin(angle)
        goal_z = self.cruising_altitude
        self.goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

        self.goal_distance = dist
        goal_len = math.sqrt(goal_x ** 2 + goal_y ** 2)
        self.path_dir = np.array(
            [goal_x / max(goal_len, 1e-6), goal_y / max(goal_len, 1e-6)],
            dtype=np.float32
        )

        print(f"\nGoal: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f}) | "
              f"Speed: {self.episode_speed:.1f} m/s")

        try:
            with open("goal_position.txt", "w") as f:
                f.write(f"{goal_x} {goal_y} {goal_z}")
        except Exception:
            pass

        self.client.reset()

        hour   = random.randint(6, 19)
        minute = random.randint(0, 59)
        self.client.simSetTimeOfDay(
            True,
            start_datetime=f"2025-06-15 {hour:02d}:{minute:02d}:00",
            is_start_datetime_dst=True,
            celestial_clock_speed=1,
            update_interval_secs=0,
            move_sun=True,
        )

        self.client.simEnableWeather(True)
        self.client.simSetWeatherParameter(
            airsim.WeatherParameter.Fog, random.uniform(0.0, 0.05)
        )

        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        time.sleep(1.0)

        if self.tree_names:
            n_total    = len(self.tree_names)
            n_obstacle = max(1, round((1 - self.bg_tree_fraction) * n_total))
            obstacle_pos = self._place_trees(goal_x, goal_y, n_obstacle)
            self._place_background_trees(goal_x, goal_y, n_obstacle, obstacle_pos)

        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self.cruising_altitude, 2.0).join()

        goal_yaw = math.degrees(math.atan2(goal_y, goal_x))
        self.client.rotateToYawAsync(goal_yaw, timeout_sec=5.0).join()
        time.sleep(0.2)

        self.client.simGetCollisionInfo()

        if self.show_visual_marker:
            self.client.simFlushPersistentMarkers()
            self.client.simPlotPoints(
                points=[airsim.Vector3r(float(goal_x), float(goal_y), float(goal_z))],
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                size=40, duration=-1, is_persistent=True,
            )

        self.current_step           = 0
        self.episode_reward         = 0.0
        self.collision_count        = 0
        self.lateral_history        = []
        self._ep_proximity_reward   = 0.0
        self._ep_straight_bonus     = 0.0
        self._ep_action_norm_penalty = 0.0
        self._ep_drift_penalty      = 0.0

        spawn_state = self.client.getMultirotorState()
        sp = spawn_state.kinematics_estimated.position
        self.spawn_pos = np.array([sp.x_val, sp.y_val], dtype=np.float32)

        # Fill frame stack with first estimated depth frame
        initial_rgb, initial_gt_depth = self._get_images()
        initial_depth = self._estimate_depth(initial_rgb)
        for i in range(self.stack_frames):
            self.frame_stack[i] = initial_depth

        state_vec = np.array(
            [float(self.episode_speed / self.speed_range[1]), 0.0],
            dtype=np.float32,
        )

        # Privileged: GT depth clipped to critic_depth_range
        center_gt = initial_gt_depth[
            self.img_height // 4: 3 * self.img_height // 4,
            self.img_width  // 4: 3 * self.img_width  // 4,
        ]
        init_min = float(np.min(np.clip(center_gt, 0.0, self.critic_depth_range)))
        privileged_vec = np.array(
            [float(np.clip(init_min / self.critic_depth_range, 0.0, 1.0))],
            dtype=np.float32,
        )

        return {
            "image":      self.frame_stack.copy(),
            "state":      state_vec,
            "privileged": privileged_vec,
        }, {}

    def step(self, action):
        t_step_start = time.perf_counter()
        self.current_step += 1

        action = np.array(action, dtype=np.float32)

        t0 = time.perf_counter()
        position = self._get_position()
        t_get_pos1 = time.perf_counter() - t0

        dx = self.goal_pos[0] - position[0]
        dy = self.goal_pos[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)
        ux, uy = dx / dist_xy, dy / dist_xy
        px, py = -uy, ux

        lateral = float(action[0]) * self.lateral_scale
        self.lateral_history.append(float(action[0]))

        vel_x = self.episode_speed * ux + lateral * px
        vel_y = self.episode_speed * uy + lateral * py
        speed_xy = math.sqrt(vel_x * vel_x + vel_y * vel_y)

        yaw_deg  = math.degrees(math.atan2(vel_y, vel_x))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        body_vx = speed_xy
        body_vy = 0.0

        altitude_error = self.cruising_altitude - position[2]
        body_vz = float(np.clip(altitude_error * 0.2, -1.0, 1.0))

        t0 = time.perf_counter()
        self.client.moveByVelocityBodyFrameAsync(
            float(body_vx), float(body_vy), float(body_vz), 5.0,
            yaw_mode=yaw_mode,
        )
        t_move_cmd = time.perf_counter() - t0

        t0 = time.perf_counter()
        rgb, gt_depth = self._get_images()
        t_fetch = time.perf_counter() - t0

        t0 = time.perf_counter()
        est_depth = self._estimate_depth(rgb)
        t_estimate = time.perf_counter() - t0

        self._update_frame_stack(est_depth)

        t0 = time.perf_counter()
        collision = self.client.simGetCollisionInfo().has_collided
        t_collision = time.perf_counter() - t0

        if collision:
            self.collision_count += 1

        t0 = time.perf_counter()
        new_position = self._get_position()
        t_get_pos2 = time.perf_counter() - t0

        new_distance = np.linalg.norm(new_position[:2] - self.goal_pos[:2])
        goal_reached = new_distance < self.goal_radius

        t_step_total = time.perf_counter() - t_step_start

        if self.ros2_bridge is not None and self.step_hz is not None:
            target    = 1.0 / self.step_hz
            remaining = target - t_step_total
            if remaining > 0:
                time.sleep(remaining)
        t_step_wall = time.perf_counter() - t_step_start

        if self.current_step % 25 == 1:
            print(f"  Step {self.current_step}: dist={dist_xy:.1f}m "
                  f"speed={self.episode_speed:.1f}m/s lateral={action[0]:.2f}")
            print(f"  [TIMING ms] pos1={t_get_pos1*1000:.1f} "
                  f"move={t_move_cmd*1000:.1f} "
                  f"fetch={t_fetch*1000:.1f} "
                  f"estimate={t_estimate*1000:.1f} "
                  f"collision={t_collision*1000:.1f} "
                  f"pos2={t_get_pos2*1000:.1f} "
                  f"total={t_step_total*1000:.1f} "
                  f"(~{1/t_step_wall:.1f} Hz)")

        if goal_reached:
            print(f"  Goal reached! Steps: {self.current_step}")

        # --- Reward ---
        reward = 0.0

        if goal_reached:
            reward += 100.0
        if collision:
            reward -= 100.0

        center_gt = gt_depth[
            self.img_height // 4: 3 * self.img_height // 4,
            self.img_width  // 4: 3 * self.img_width  // 4,
        ]
        center_gt_clipped = np.clip(center_gt, 0.0, self.prox_threshold)
        min_depth_center  = float(np.min(center_gt_clipped))

        if min_depth_center < self.prox_threshold:
            prox_r = -((self.prox_threshold - min_depth_center) / self.prox_threshold * 0.5)
            reward += prox_r
            self._ep_proximity_reward += prox_r
        else:
            straight_r = (1.0 - abs(float(action[0]))) * 0.05
            reward += straight_r
            self._ep_straight_bonus += straight_r

        action_norm_r = -(float(np.linalg.norm(action)) * 0.1)
        reward += action_norm_r
        self._ep_action_norm_penalty += action_norm_r

        vec = new_position[:2] - self.spawn_pos
        perp = np.array([-self.path_dir[1], self.path_dir[0]], dtype=np.float32)
        lateral_offset = float(np.dot(vec, perp))
        if abs(lateral_offset) > 1.0:
            drift_r = -(abs(lateral_offset) - 1.0) * 0.005
            reward += drift_r
            self._ep_drift_penalty += drift_r

        self.episode_reward += reward

        terminated = goal_reached or collision
        truncated  = self.current_step >= self.max_steps

        if terminated or truncated:
            reason = "GOAL" if goal_reached else ("COLLISION" if collision else "MAX STEPS")
            lat = np.array(self.lateral_history)
            avg_lat     = float(np.mean(lat))     if len(lat) > 0 else 0.0
            avg_abs_lat = float(np.mean(np.abs(lat))) if len(lat) > 0 else 0.0
            print(f"Episode ended ({reason}) | Reward: {self.episode_reward:.2f} | "
                  f"Steps: {self.current_step} | Distance: {new_distance:.1f}m | "
                  f"Collisions: {self.collision_count} | "
                  f"Lateral avg={avg_lat:+.3f} abs={avg_abs_lat:.3f} | "
                  f"prox={self._ep_proximity_reward:.1f} "
                  f"straight={self._ep_straight_bonus:.1f} "
                  f"norm={self._ep_action_norm_penalty:.1f} "
                  f"drift={self._ep_drift_penalty:.1f}")

        state_vec    = self._compute_state(new_position)
        min_depth_long = float(np.min(np.clip(center_gt, 0.0, self.critic_depth_range)))
        depth_norm     = float(np.clip(min_depth_long / self.critic_depth_range, 0.0, 1.0))
        privileged_vec = np.array([depth_norm], dtype=np.float32)

        info = {
            'goal_reached':    goal_reached,
            'collision':       collision,
            'distance':        new_distance,
            'collision_count': self.collision_count,
        }
        if terminated or truncated:
            lat = np.array(self.lateral_history)
            info['avg_lateral']           = float(np.mean(lat)) if len(lat) > 0 else 0.0
            info['avg_abs_lateral']       = float(np.mean(np.abs(lat))) if len(lat) > 0 else 0.0
            info['ep_proximity_reward']   = self._ep_proximity_reward
            info['ep_straight_bonus']     = self._ep_straight_bonus
            info['ep_action_norm_penalty'] = self._ep_action_norm_penalty
            info['ep_drift_penalty']      = self._ep_drift_penalty

        return (
            {
                "image":      self.frame_stack.copy(),
                "state":      state_vec,
                "privileged": privileged_vec,
            },
            reward, terminated, truncated, info,
        )
