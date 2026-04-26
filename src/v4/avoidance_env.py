"""
GPS-Free Obstacle Avoidance Environment — v4

The drone flies forward along X+ at a coupled speed.  The RL policy controls
only lateral (Y) movement to dodge obstacles and recover to the centre line.
No GPS or goal position appears in any actor observation.

Key differences from v3:
  - No goal position — yaw locked to 0° (X+ always forward)
  - No state vector in actor obs (removed dist-to-goal, speed_norm)
  - Speed coupled to lateral action: fwd = fixed_speed × (1 − |action|)
  - 1-step action delay + 0.3 momentum smoothing
  - Single velocity command per gym step (v3 pattern, no frame-skip loop)
  - Ellipse tree placement scaled to fit all trees at the chosen spacing
  - Privileged critic vector: lat_offset + velocities + forward tree positions/scales
  - Curriculum: 50-episode batch win rate (win = cross CORRIDOR_LENGTH without collision)
  - Reward: depth proximity penalty + quadratic lateral centering penalty + survival bonus

Observation (actor): Dict {
    "image":      (STACK_FRAMES, H, W) float32  — stacked estimated depth frames [0,1]
    "privileged": (PRIVILEGED_DIM,) float32  — critic only (not seen by actor at deploy)
}

Action: (1,) float32 in [−1, 1]
    Lateral correction; forward speed coupled: fixed_speed × (1 − |action|)
"""

import math
import os
import random
import time

import cosysairsim as airsim
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces


TREE_ACTOR_FILTER   = 'StaticMeshActor_UAID_'

CORRIDOR_LENGTH      = 30.0   # metres — win distance (drone must cross this without collision)
CORRIDOR_WIDTH       = 3.5    # metres either side — lateral penalty zone / background exclusion
ACTION_MOMENTUM      = 0.3    # lateral smoothing coefficient

MAX_TREES_PRIVILEGED = 20
GT_DEPTH_RANGE       = 20.0   # metres — critic sees GT depth up to this range
PRIVILEGED_DIM       = 4 + MAX_TREES_PRIVILEGED * 2 + 1   # 45  (+1 = GT min-depth)


class DroneAvoidanceEnv(gym.Env):
    """
    GPS-free obstacle avoidance.
    Actor sees only the 3-frame estimated depth stack.
    Critic additionally sees the privileged vector (tree positions + drone state).
    Win condition: cross CORRIDOR_LENGTH (30 m) forward without collision.
    max_steps is a safety timeout — episodes that time out count as losses.
    """

    metadata = {'render_modes': ['rgb_array']}

    # ── Class-level curriculum / batch state (shared across instances) ─────────
    _batch_wins               = 0
    _batch_episodes           = 0
    _batch_win_rate           = None
    _batch_complete           = False
    _current_curriculum_stage = 0

    def __init__(
        self,
        fixed_speed=1.5,
        max_lateral=1.5,
        cruising_altitude=-1.3,
        altitude_kp=0.2,
        max_vz=1.0,
        max_steps=300,
        prox_threshold=3.5,
        lat_penalty_weight=0.01,
        fwd_reward_weight=2.0,
        action_reward_weight=0.1,
        collision_penalty=50.0,
        show_visual_marker=False,
        ros2_bridge=None,
        depth_estimator=None,
    ):
        super().__init__()

        self.fixed_speed       = fixed_speed
        self.max_lateral       = max_lateral
        self.cruising_altitude = cruising_altitude
        self.altitude_kp       = altitude_kp
        self.max_vz            = max_vz
        self.max_steps         = max_steps
        self.prox_threshold    = prox_threshold
        self.lat_penalty_weight    = lat_penalty_weight
        self.fwd_reward_weight     = fwd_reward_weight
        self.action_reward_weight  = action_reward_weight
        self.collision_penalty     = collision_penalty
        self.show_visual_marker = show_visual_marker
        self.ros2_bridge       = ros2_bridge
        self.depth_estimator   = depth_estimator

        self.img_height   = 144    # 16:9 at ~192×192 pixel count
        self.img_width    = 256
        self.stack_frames = 3
        self.training_mode = True

        # Episode state
        self.spawn_pos            = np.zeros(3, dtype=np.float32)
        self._pos                 = np.zeros(3, dtype=np.float32)
        self._vel                 = np.zeros(3, dtype=np.float32)
        self.current_step         = 0
        self.episode_reward       = 0.0
        self.episode_count        = 0
        self.last_survived        = False
        self._episode_stage       = 0
        self.last_episode_stage   = 0
        self._last_collision_time = 0

        # Per-episode reward breakdown tracking
        self._ep_depth_penalty    = 0.0
        self._ep_lateral_penalty  = 0.0
        self._ep_fwd_reward       = 0.0
        self._ep_action_reward    = 0.0
        self._action_history      = []   # smoothed lateral actions for stats
        self._ep_density_label    = ''   # set by _spawn_corridor_trees()
        self._prev_x              = 0.0  # for computing per-step delta_x
        self._step_times          = []   # wall-clock duration of each step (seconds)

        # Action smoothing
        self.prev_action     = np.zeros(1, dtype=np.float32)
        self._delayed_action = np.zeros(1, dtype=np.float32)

        # Frame stack + latest camera data
        self.frame_stack   = np.zeros(
            (self.stack_frames, self.img_height, self.img_width), dtype=np.float32
        )
        self._latest_rgb   = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        self._latest_depth = np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)

        # Trees
        self.tree_positions = []   # [(x, y), ...]
        self.tree_scales    = []   # [scale, ...]

        # Privileged vector (built in _build_privileged_vector)
        self._privileged_vector = np.zeros(PRIVILEGED_DIM, dtype=np.float32)

        # AirSim client
        airsim_host = os.environ.get('AIRSIM_HOST', '')
        self.client = (
            airsim.MultirotorClient(ip=airsim_host) if airsim_host
            else airsim.MultirotorClient()
        )
        self.client.confirmConnection()
        print(f"Connected to AirSim at {airsim_host or 'localhost'}")

        all_objects = self.client.simListSceneObjects()
        self.tree_names = [o for o in all_objects if TREE_ACTOR_FILTER in o]
        n_bg = max(1, round(0.30 * len(self.tree_names)))
        self._n_obstacle  = len(self.tree_names) - n_bg
        print(
            f"Found {len(self.tree_names)} tree actors "
            f"({self._n_obstacle} obstacle, {n_bg} background)"
        )

        self.ground_z = 0.0
        if self.tree_names:
            pose = self.client.simGetObjectPose(self.tree_names[0])
            self.ground_z = pose.position.z_val
            print(f"Ground z: {self.ground_z:.2f}")

        # Observation / action spaces
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(self.stack_frames, self.img_height, self.img_width),
                dtype=np.float32,
            ),
            "privileged": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(PRIVILEGED_DIM,),
                dtype=np.float32,
            ),
        })
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # ── Tree placement ────────────────────────────────────────────────────────

    def _hide_all_trees(self):
        hide = airsim.Pose(airsim.Vector3r(0, 0, 500), airsim.Quaternionr(0, 0, 0, 1))
        for name in self.tree_names:
            try:
                self.client.simSetObjectPose(name, hide, teleport=True)
            except Exception:
                pass

    def _spawn_corridor_trees(self):
        """Place all obstacle trees in a scaled ellipse along the X+ axis.

        Identical to v3 logic: the ellipse semi-axes scale so that all
        n_obstacle trees fit at the chosen min_dist spacing.  Trees that
        cannot be placed fall back to the last successfully placed position
        so every actor is still moved.
        """
        ALL_DENSITY_CONFIGS = [
            ('sparse', 8.0),
            ('medium', 6.5),
            ('dense',  5.0),
        ]
        active_configs          = ALL_DENSITY_CONFIGS[:self._episode_stage + 1]
        label, min_dist         = random.choice(active_configs)
        self._ep_density_label  = label

        n_trees = self._n_obstacle
        sx, sy  = float(self.spawn_pos[0]), float(self.spawn_pos[1])

        DRONE_CLEARANCE = 4.0

        # Build a base ellipse along X+, centred 50 m ahead of spawn.
        # This mirrors v3's ellipse geometry (spawn→goal replaced by a fixed
        # 100 m reference so the shape is always elongated forward).
        ref_dist = 100.0
        c        = ref_dist / 2.0          # half the reference distance
        base_a   = c + 10.0                # forward semi-axis
        base_b   = math.sqrt(max(base_a ** 2 - c ** 2, 1.0))  # lateral semi-axis

        # Scale semi-axes so all n_trees fit at min_dist spacing (v3 formula)
        required_area = n_trees * (min_dist ** 2) / 0.55
        base_area     = math.pi * base_a * base_b
        scale         = max(1.0, math.sqrt(required_area / base_area))
        a             = base_a * scale   # scaled forward semi-axis
        b             = base_b * scale   # scaled lateral semi-axis

        cx, cy = sx + ref_dist / 2.0, sy  # ellipse centre

        # Rejection sampling — uniform area sampling within the ellipse
        placed   = []
        attempts = 0
        while len(placed) < n_trees and attempts < n_trees * 300:
            attempts += 1
            r     = math.sqrt(random.random())
            theta = random.uniform(0, 2 * math.pi)
            tx    = cx + r * a * math.cos(theta)
            ty    = cy + r * b * math.sin(theta)
            if tx < sx + DRONE_CLEARANCE:   # keep trees in front of drone
                continue
            if all(math.sqrt((tx - px) ** 2 + (ty - py) ** 2) >= min_dist
                   for px, py in placed):
                placed.append((tx, ty))

        print(
            f"  Forest: {label} ({len(placed)}/{n_trees} trees, "
            f"min spacing {min_dist}m, ellipse {a:.0f}×{b:.0f}m)"
        )

        # Move every obstacle actor — overflow trees fall back to last placed position
        fallback = placed[-1] if placed else (cx, cy)
        placed_positions = []
        placed_scales    = []
        ok = fail = 0
        for i in range(n_trees):
            tx, ty = placed[i] if i < len(placed) else fallback
            yaw    = random.uniform(0, 2 * math.pi)
            sc_xy  = random.uniform(0.7, 1.3)
            sc_z   = random.uniform(1.0, 1.5)
            pose   = airsim.Pose(
                airsim.Vector3r(tx, ty, self.ground_z),
                airsim.Quaternionr(0, 0, math.sin(yaw / 2), math.cos(yaw / 2))
            )
            self.client.simSetObjectScale(
                self.tree_names[i], airsim.Vector3r(sc_xy, sc_xy, sc_z)
            )
            if self.client.simSetObjectPose(self.tree_names[i], pose, teleport=True):
                ok += 1
            else:
                fail += 1
            placed_positions.append((tx, ty))
            placed_scales.append(sc_xy)

        print(f"  Tree placement: {ok} moved, {fail} failed")

        # Only uniquely placed trees count as real obstacles for privileged vector
        self.tree_positions = placed[:len(placed)]
        self.tree_scales    = placed_scales[:len(placed)]

        return placed_positions

    def _place_background_trees(self, obstacle_positions):
        """Scatter all background tree actors outside the corridor for visual realism."""
        bg_names = self.tree_names[self._n_obstacle:]
        if not bg_names:
            return

        sx, sy  = float(self.spawn_pos[0]), float(self.spawn_pos[1])
        MAX_R   = 180.0   # covers the full obstacle ellipse (~100 m ref + scaled semi-axis)
        MIN_GAP = 5.0

        all_blocked = list(obstacle_positions)
        placed_bg   = []
        placed = 0

        for name in bg_names:
            success = False
            for _ in range(60):
                angle = random.uniform(0, 2 * math.pi)
                r     = random.uniform(0.0, MAX_R)
                tx    = sx + r * math.cos(angle)
                ty    = sy + r * math.sin(angle)

                if math.sqrt((tx - sx) ** 2 + (ty - sy) ** 2) < 2.0:
                    continue
                fwd_proj = tx - sx
                lat_dist = abs(ty - sy)
                if lat_dist < CORRIDOR_WIDTH and 0.0 < fwd_proj < CORRIDOR_LENGTH:
                    continue
                if any(
                    math.sqrt((tx - px) ** 2 + (ty - py) ** 2) < MIN_GAP
                    for px, py in all_blocked + placed_bg
                ):
                    continue
                placed_bg.append((tx, ty))
                success = True
                break

            if success:
                tx, ty = placed_bg[-1]
            else:
                tx, ty = sx + 9000.0 + placed * 10.0, sy + 9000.0
                placed_bg.append((tx, ty))

            sc = random.uniform(0.5, 1.4)
            try:
                pose = airsim.Pose(
                    airsim.Vector3r(tx, ty, self.ground_z),
                    airsim.Quaternionr(0, 0, 0, 1)
                )
                self.client.simSetObjectScale(name, airsim.Vector3r(sc, sc, random.uniform(1.0, 1.5)))
                self.client.simSetObjectPose(name, pose, teleport=True)
                placed += 1
            except Exception:
                pass

        print(f"  Background trees: {placed} placed outside corridor")

    # ── Vision ────────────────────────────────────────────────────────────────

    def _fetch_rgb_and_depth(self):
        """Get latest RGB (for depth estimation) and GT depth (for reward penalty)."""
        if self.ros2_bridge is not None:
            rgb = self.ros2_bridge.get_latest_frame()
            if rgb is None:
                rgb = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
            depth = self.ros2_bridge.get_latest_depth()
            if depth is None:
                depth = self._api_depth()
        else:
            rgb, depth = self._api_rgb_and_depth()

        self._latest_rgb   = rgb
        self._latest_depth = depth

    def _api_rgb_and_depth(self):
        """Fetch both RGB and depth from AirSim API in a single call."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.Scene,           False, False),
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True,  False),
            ])
            rgb   = self._parse_rgb(responses[0])   if responses and len(responses) > 0 else None
            depth = self._parse_depth(responses[1]) if responses and len(responses) > 1 else None
        except Exception:
            rgb, depth = None, None

        if rgb   is None:
            rgb   = np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)
        if depth is None:
            depth = np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)
        return rgb, depth

    def _api_depth(self):
        """Fetch GT depth only from AirSim API (fallback when ROS2 bridge has no depth)."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False),
            ])
            if responses and responses[0].width > 0:
                return self._parse_depth(responses[0])
        except Exception:
            pass
        return np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)

    def _parse_rgb(self, r):
        if r is None or r.width == 0:
            return None
        raw  = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        n_ch = len(raw) // (r.width * r.height)
        img  = raw.reshape(r.height, r.width, n_ch)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB if n_ch == 4 else cv2.COLOR_BGR2RGB)
        return cv2.resize(rgb, (self.img_width, self.img_height))

    def _parse_depth(self, r):
        if r is None or r.width == 0:
            return None
        d = airsim.list_to_2d_float_array(r.image_data_float, r.width, r.height)
        return cv2.resize(d, (self.img_width, self.img_height))

    def _run_depth_estimation(self):
        """Estimate depth from latest RGB and push one frame onto the stack."""
        if self.depth_estimator is not None:
            est = self.depth_estimator.estimate(self._latest_rgb, training=self.training_mode)
        else:
            est = np.zeros((self.img_height, self.img_width), dtype=np.float32)
        self.frame_stack = np.roll(self.frame_stack, shift=-1, axis=0)
        self.frame_stack[-1] = est

    # ── State & privileged vector ─────────────────────────────────────────────

    def _update_state(self):
        s  = self.client.getMultirotorState()
        ap = s.kinematics_estimated.position
        av = s.kinematics_estimated.linear_velocity
        self._pos = np.array([ap.x_val, ap.y_val, ap.z_val], dtype=np.float32)
        self._vel = np.array([av.x_val, av.y_val, av.z_val], dtype=np.float32)
        self._build_privileged_vector()

    def _build_privileged_vector(self):
        pv   = np.zeros(PRIVILEGED_DIM, dtype=np.float32)
        norm = max(CORRIDOR_LENGTH, 1.0)

        # Lateral offset from spawn Y (positive = right of corridor centreline)
        lat_off  = self._pos[1] - self.spawn_pos[1]
        # Forward progress from spawn X
        fwd_prog = self._pos[0] - self.spawn_pos[0]

        pv[0] = lat_off  / norm
        pv[1] = fwd_prog / norm
        pv[2] = self._vel[0] / max(self.fixed_speed, 0.1)
        pv[3] = self._vel[1] / max(self.max_lateral,  0.1)

        # Forward trees only (those still ahead of drone along X+)
        forward = [
            (tx, ty)
            for (tx, ty) in self.tree_positions
            if tx > float(self._pos[0])
        ]
        forward.sort(key=lambda t: (t[0] - self._pos[0]) ** 2 + (t[1] - self._pos[1]) ** 2)
        for i, (tx, ty) in enumerate(forward[:MAX_TREES_PRIVILEGED]):
            pv[4 + i * 2]     = (tx - float(self._pos[0])) / norm
            pv[4 + i * 2 + 1] = (ty - float(self._pos[1])) / norm

        # GT min-depth from centre ROI — gives the critic reliable real-time
        # obstacle proximity, independent of tree-position accuracy.
        if self._latest_depth is not None:
            qh    = self.img_height // 4
            qw    = self.img_width  // 4
            roi   = self._latest_depth[qh: 3 * qh, qw: 3 * qw]
            min_d = float(np.min(np.clip(roi, 0.0, GT_DEPTH_RANGE)))
            pv[PRIVILEGED_DIM - 1] = min_d / GT_DEPTH_RANGE  # index 44
        # else stays 0 (all clear) until first depth frame arrives

        self._privileged_vector = pv

    # ── Reward components ─────────────────────────────────────────────────────

    def _depth_penalty(self):
        """Centre-ROI proximity penalty using GT depth (same formula as v3)."""
        qh  = self.img_height // 4
        qw  = self.img_width  // 4
        roi = self._latest_depth[qh: 3 * qh, qw: 3 * qw]
        min_d = float(np.min(np.clip(roi, 0.0, self.prox_threshold)))
        if min_d < self.prox_threshold:
            return -((self.prox_threshold - min_d) / self.prox_threshold) * 0.5
        return 0.0

    def _lateral_penalty(self):
        """Quadratic penalty proportional to lateral offset from corridor centreline."""
        lat_off = float(self._pos[1] - self.spawn_pos[1])
        return -self.lat_penalty_weight * lat_off ** 2

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step        = 0
        self.episode_reward      = 0.0
        self.last_survived       = False
        self.prev_action         = np.zeros(1, dtype=np.float32)
        self._delayed_action     = np.zeros(1, dtype=np.float32)
        self._ep_depth_penalty   = 0.0
        self._ep_lateral_penalty = 0.0
        self._ep_fwd_reward      = 0.0
        self._ep_action_reward   = 0.0
        self._action_history     = []
        self._step_times         = []

        self._episode_stage = DroneAvoidanceEnv._current_curriculum_stage

        self.client.reset()
        time.sleep(0.3)
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self.cruising_altitude, 2.0).join()
        # Lock yaw to 0° (X+ forward)
        self.client.rotateToYawAsync(0.0, timeout_sec=3.0).join()
        time.sleep(0.2)

        state = self.client.getMultirotorState()
        sp = state.kinematics_estimated.position
        self.spawn_pos = np.array([sp.x_val, sp.y_val, sp.z_val], dtype=np.float32)
        self._pos      = self.spawn_pos.copy()

        # Randomise lighting and light fog
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
            airsim.WeatherParameter.Fog, random.uniform(0.0, 0.08)
        )

        obstacle_pos = self._spawn_corridor_trees()
        self._place_background_trees(obstacle_pos)
        time.sleep(0.5)

        if self.show_visual_marker:
            # Draw at vis_z = -20m (20m above scene) — visible from UE viewport,
            # outside drone camera FOV during normal forward flight.
            vis_z = -20.0
            self.client.simFlushPersistentMarkers()
            sx, sy = float(self.spawn_pos[0]), float(self.spawn_pos[1])
            ex = sx + CORRIDOR_LENGTH
            # Corridor centreline: spawn → end
            self.client.simPlotLineStrip(
                [airsim.Vector3r(sx, sy, vis_z), airsim.Vector3r(ex, sy, vis_z)],
                color_rgba=[0.0, 1.0, 0.2, 1.0],
                thickness=8.0, duration=3600.0, is_persistent=False,
            )
            # Corridor side walls
            w = CORRIDOR_WIDTH
            for lat in [-w, w]:
                self.client.simPlotLineStrip(
                    [airsim.Vector3r(sx, sy + lat, vis_z),
                     airsim.Vector3r(ex, sy + lat, vis_z)],
                    color_rgba=[1.0, 0.5, 0.0, 1.0],
                    thickness=4.0, duration=3600.0, is_persistent=False,
                )
            # End marker
            self.client.simPlotPoints(
                [airsim.Vector3r(ex, sy, vis_z)],
                color_rgba=[1.0, 1.0, 0.0, 1.0],
                size=40, duration=3600.0, is_persistent=False,
            )

        # Record collision timestamp so sub-step checks don't fire on stale data
        try:
            self._last_collision_time = self.client.simGetCollisionInfo().time_stamp
        except Exception:
            self._last_collision_time = 0

        # Initialise frame stack with first depth estimate
        self._fetch_rgb_and_depth()
        self._update_state()
        self._prev_x = float(self._pos[0])
        for _ in range(self.stack_frames):
            self._run_depth_estimation()

        stage_names = ['sparse', 'sparse+medium', 'all']
        print(
            f"\n[Ep {self.episode_count}] "
            f"Stage {self._episode_stage + 1} ({stage_names[self._episode_stage]}) | "
            f"density={self._ep_density_label} | "
            f"spawn=({self.spawn_pos[0]:.1f}, {self.spawn_pos[1]:.1f}) | "
            f"fwd={self.fixed_speed}m/s  max_lat={self.max_lateral}m/s | "
            f"win={CORRIDOR_LENGTH:.0f}m  timeout={self.max_steps}steps"
        )

        return {
            "image":      self.frame_stack.copy(),
            "privileged": self._privileged_vector.copy(),
        }, {}

    def step(self, action):
        self.current_step += 1
        _t_step_start = time.perf_counter()

        # 1-step pipeline delay + momentum smoothing
        action_to_exec       = self._delayed_action.copy()
        self._delayed_action = np.array(action, dtype=np.float32)
        smoothed = (
            (1 - ACTION_MOMENTUM) * action_to_exec
            + ACTION_MOMENTUM     * self.prev_action
        )
        self.prev_action = smoothed.copy()

        terminated = False
        truncated  = False

        # ── Send velocity command (v3 pattern — no frame-skip loop) ──────────
        # duration=5.0 means AirSim keeps executing until the next call
        # overrides it.  The natural step rate is set by depth estimation
        # overhead (~200-300 ms), so no explicit sleep is needed.
        action_abs = abs(float(smoothed[0]))
        fwd_speed  = self.fixed_speed * (1.0 - action_abs)
        lat_speed  = self.max_lateral * float(smoothed[0])
        alt_error  = float(self.cruising_altitude) - float(self._pos[2])
        vz         = float(np.clip(alt_error * self.altitude_kp, -self.max_vz, self.max_vz))

        self.client.moveByVelocityAsync(
            fwd_speed, lat_speed, vz, 5.0,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0.0),
        )

        # Collision check
        if self.current_step > 1:
            try:
                col = self.client.simGetCollisionInfo()
                if col.has_collided and col.time_stamp > self._last_collision_time:
                    self._last_collision_time = col.time_stamp
                    terminated = True
            except Exception:
                pass

        # ── Fetch latest frames and update state ─────────────────────────────
        self._fetch_rgb_and_depth()
        self._run_depth_estimation()
        self._update_state()

        # ── Win / timeout check ───────────────────────────────────────────────
        fwd_progress = float(self._pos[0] - self.spawn_pos[0])
        if not terminated:
            if fwd_progress >= CORRIDOR_LENGTH or self.current_step >= self.max_steps:
                self.last_survived = True
                truncated = True

        # ── Reward ────────────────────────────────────────────────────────────
        delta_x       = float(self._pos[0]) - self._prev_x          # actual metres forward
        self._prev_x  = float(self._pos[0])
        reward_fwd    = self.fwd_reward_weight * max(delta_x, 0.0)    # only reward forward motion
        reward_action = self.action_reward_weight * (1.0 - abs(float(smoothed[0])))

        dp = self._depth_penalty()
        lp = self._lateral_penalty()

        self._ep_fwd_reward      += reward_fwd
        self._ep_action_reward   += reward_action
        self._ep_depth_penalty   += dp
        self._ep_lateral_penalty += lp
        self._action_history.append(float(smoothed[0]))

        reward = reward_fwd + reward_action + dp + lp
        if terminated:
            reward -= self.collision_penalty

        self.episode_reward += reward

        # ── Episode end bookkeeping ───────────────────────────────────────────
        if terminated or truncated:
            if self.last_survived:
                reason = "WIN"
            elif terminated:
                reason = "COLLISION"
            else:
                reason = "TIMEOUT"

            lat_off  = float(self._pos[1] - self.spawn_pos[1])
            acts    = np.array(self._action_history) if self._action_history else np.zeros(1)
            avg_lat = float(np.mean(acts))
            abs_lat = float(np.mean(np.abs(acts)))
            col_str = f"-{self.collision_penalty:.0f}" if terminated else "0"

            print(
                f"  → {reason} | "
                f"steps={self.current_step} | "
                f"fwd={fwd_progress:.1f}m | lat={lat_off:+.2f}m | "
                f"reward={self.episode_reward:.1f}  "
                f"[fwd=+{self._ep_fwd_reward:.1f}  "
                f"action=+{self._ep_action_reward:.1f}  "
                f"depth={self._ep_depth_penalty:.1f}  "
                f"lat={self._ep_lateral_penalty:.1f}  "
                f"collision={col_str}]",
                flush=True,
            )
            if self._step_times:
                avg_ms = float(np.mean(self._step_times)) * 1000
                avg_hz = 1000.0 / avg_ms if avg_ms > 0 else 0.0
                print(
                    f"     Step timing:   avg={avg_ms:.1f}ms ({avg_hz:.1f} Hz)  "
                    f"min={min(self._step_times)*1000:.1f}ms  "
                    f"max={max(self._step_times)*1000:.1f}ms",
                    flush=True,
                )
            print(
                f"     Lateral action: avg={avg_lat:+.3f}  abs_avg={abs_lat:.3f}",
                flush=True,
            )

            self.last_episode_stage = self._episode_stage
            self.episode_count += 1

            is_win = self.last_survived
            DroneAvoidanceEnv._batch_wins     += 1 if is_win else 0
            DroneAvoidanceEnv._batch_episodes += 1

            print(
                f"  [BATCH {DroneAvoidanceEnv._batch_episodes}/50] "
                f"wins={DroneAvoidanceEnv._batch_wins} "
                f"({DroneAvoidanceEnv._batch_wins / DroneAvoidanceEnv._batch_episodes:.0%}) "
                f"| Stage {self._episode_stage + 1}",
                flush=True,
            )

            if DroneAvoidanceEnv._batch_episodes >= 50:
                win_rate = DroneAvoidanceEnv._batch_wins / 50
                DroneAvoidanceEnv._batch_win_rate = win_rate
                DroneAvoidanceEnv._batch_complete = True
                stage_names = ['sparse', 'sparse+medium', 'all']
                cur = DroneAvoidanceEnv._current_curriculum_stage
                advance = cur < 2 and win_rate >= 0.80
                print(
                    f"\n{'='*60}\n"
                    f"[BATCH COMPLETE] 50 episodes | "
                    f"win_rate={win_rate:.0%} | "
                    f"Stage {cur + 1} ({stage_names[cur]})"
                    + (f" → advancing to Stage {cur + 2}!" if advance else " | no advance")
                    + f"\n{'='*60}\n",
                    flush=True,
                )
                DroneAvoidanceEnv._batch_wins     = 0
                DroneAvoidanceEnv._batch_episodes = 0
                if advance:
                    DroneAvoidanceEnv._current_curriculum_stage += 1

        # Rate-limit to 10 Hz — sleep the remaining time in this step
        _elapsed = time.perf_counter() - _t_step_start
        _remaining = (1.0 / 10.0) - _elapsed
        if _remaining > 0:
            time.sleep(_remaining)
        self._step_times.append(time.perf_counter() - _t_step_start)

        # Episode-level breakdown keys are always present so Monitor/callback
        # can read them at episode end (when done=True).  Mid-episode values
        # are partial accumulations and should be ignored by the callback.
        info = {
            "survived":           self.last_survived,
            "fwd_progress":       float(self._pos[0] - self.spawn_pos[0]),
            "lat_offset":         float(self._pos[1] - self.spawn_pos[1]),
            "collided":           terminated,
            "ep_fwd_reward":      self._ep_fwd_reward,
            "ep_action_reward":   self._ep_action_reward,
            "ep_depth_penalty":   self._ep_depth_penalty,
            "ep_lateral_penalty": self._ep_lateral_penalty,
            "lat_avg_signed":     float(np.mean(self._action_history))
                                  if self._action_history else 0.0,
        }

        return (
            {
                "image":      self.frame_stack.copy(),
                "privileged": self._privileged_vector.copy(),
            },
            float(reward),
            terminated,
            truncated,
            info,
        )

    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except Exception:
            pass
