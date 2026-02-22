"""
Simplified Obstacle Avoidance Environment

Navigation is handled by a simple controller (go toward goal).
The RL model ONLY learns obstacle avoidance from the camera image.

Controller: flies forward toward goal at base_speed
RL Model:   sees 4 stacked grayscale frames, outputs lateral + vertical correction
Combined:   velocity = forward_toward_goal + lateral_perpendicular + altitude_hold
Yaw:        faces actual movement direction (camera sees what's ahead)

Observation: (4, 84, 84) uint8 — 4 stacked grayscale frames, channels-first (CHW)
             No VecTransposeImage needed; SB3 CnnPolicy detects CHW automatically.
Depth image: requested alongside RGB in the same API call, used only for the
             soft proximity penalty reward (not fed to the model).
"""

import math
import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2


class ObstacleAvoidanceEnv(gym.Env):
    """RL model learns to avoid obstacles while a simple controller navigates."""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 goal_distance_range=(50, 50),
                 cruising_altitude=-5.0,
                 base_speed=1.0,
                 lateral_scale=1.0,
                 vertical_scale=0.5,
                 goal_radius=1.5,
                 max_steps=500,
                 show_visual_marker=False):
        super().__init__()

        self.goal_distance_range = goal_distance_range
        self.cruising_altitude = cruising_altitude
        self.base_speed = base_speed
        self.lateral_scale = lateral_scale
        self.vertical_scale = vertical_scale
        self.goal_radius = goal_radius
        self.max_steps = max_steps
        self.show_visual_marker = show_visual_marker

        self.img_height = 84
        self.img_width = 84
        self.stack_frames = 4

        # Soft proximity penalty: penalise if anything in the centre of the
        # depth image is closer than this threshold (metres).
        self.prox_threshold = 5.0

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected to AirSim!")

        self.camera_name = "front_center"
        self.goal_pos = None

        # Observation: 4 stacked grayscale frames in CHW format.
        # SB3 CnnPolicy expects (C, H, W); first dim (4) < spatial dims (84)
        # so SB3 detects this as already channels-first and skips transposition.
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.stack_frames, self.img_height, self.img_width),
            dtype=np.uint8
        )

        # Action: lateral correction + vertical correction
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.frame_stack = np.zeros(
            (self.stack_frames, self.img_height, self.img_width), dtype=np.uint8
        )
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.current_step = 0
        self.episode_reward = 0.0
        self.collision_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random goal position
        angle = np.random.uniform(0, 2 * np.pi)
        dist = np.random.uniform(*self.goal_distance_range)
        goal_x = dist * np.cos(angle)
        goal_y = dist * np.sin(angle)
        goal_z = self.cruising_altitude
        self.goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

        print(f"\nGoal: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f})")

        try:
            with open("goal_position.txt", "w") as f:
                f.write(f"{goal_x} {goal_y} {goal_z}")
        except Exception:
            pass

        # Reset drone and take off
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(self.cruising_altitude, 2.0).join()

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
        self.prev_action = np.zeros(2, dtype=np.float32)

        # Fill the frame stack with the first captured frame (no motion yet)
        initial_gray, _ = self._get_images()
        for i in range(self.stack_frames):
            self.frame_stack[i] = initial_gray

        return self.frame_stack.copy(), {}

    def step(self, action):
        self.current_step += 1

        # Action smoothing: 50% blend with previous action.
        # Prevents abrupt velocity reversals and stabilises AirSim physics.
        action = 0.5 * np.array(action, dtype=np.float32) + 0.5 * self.prev_action
        self.prev_action = action.copy()

        position = self._get_position()

        # Controller: unit vector toward goal and perpendicular (left)
        dx = self.goal_pos[0] - position[0]
        dy = self.goal_pos[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)
        ux, uy = dx / dist_xy, dy / dist_xy
        px, py = -uy, ux

        # RL correction mapped to world frame
        lateral = float(action[0]) * self.lateral_scale
        vertical = float(action[1]) * self.vertical_scale

        # Combined velocity: forward toward goal + lateral perpendicular
        vel_x = self.base_speed * ux + lateral * px
        vel_y = self.base_speed * uy + lateral * py
        speed_xy = math.sqrt(vel_x * vel_x + vel_y * vel_y)

        # Yaw faces actual movement direction so camera sees what's ahead
        yaw_deg = math.degrees(math.atan2(vel_y, vel_x))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        # Body frame: all horizontal speed is forward, no lateral slip
        body_vx = speed_xy
        body_vy = 0.0

        # Altitude hold
        altitude_error = self.cruising_altitude - position[2]
        base_vz = float(np.clip(altitude_error * 0.5, -1.0, 1.0))
        body_vz = base_vz + vertical
        if position[2] > self.cruising_altitude + 1.0:
            body_vz = min(body_vz, base_vz)

        if self.current_step % 25 == 1:
            print(f"  Step {self.current_step}: dist={dist_xy:.1f}m "
                  f"vel=({vel_x:.2f},{vel_y:.2f}) body_vx={body_vx:.2f} "
                  f"action=({action[0]:.2f},{action[1]:.2f})")

        self.client.moveByVelocityBodyFrameAsync(
            float(body_vx), float(body_vy), float(body_vz), 5.0,
            yaw_mode=yaw_mode
        )

        # Single API call returns both RGB (for obs) and depth (for penalty)
        gray, depth = self._get_images()
        self._update_frame_stack(gray)

        # Check collision
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            self.collision_count += 1

        # Check goal
        new_position = self._get_position()
        new_distance = np.linalg.norm(new_position[:2] - self.goal_pos[:2])
        goal_reached = new_distance < self.goal_radius

        if goal_reached:
            print(f"  Goal reached! Steps: {self.current_step}")

        # --- Reward ---
        reward = 0.0

        if goal_reached:
            reward += 10.0

        if collision:
            reward -= 10.0

        # Soft proximity penalty: centre 50% of depth image, threshold self.prox_threshold.
        # Gives a smooth gradient pushing the drone to stay clear of obstacles,
        # rather than only penalising at the hard collision boundary.
        center_depth = depth[
            self.img_height // 4: 3 * self.img_height // 4,
            self.img_width  // 4: 3 * self.img_width  // 4
        ]
        min_depth_center = float(np.min(center_depth))
        if min_depth_center < self.prox_threshold:
            reward -= (self.prox_threshold - min_depth_center) / self.prox_threshold * 2.0

        # Action norm penalty on the smoothed action
        reward -= float(np.linalg.norm(action)) * 0.05

        self.episode_reward += reward

        # Termination
        terminated = goal_reached or collision
        truncated = self.current_step >= self.max_steps

        if terminated or truncated:
            reason = "GOAL" if goal_reached else ("COLLISION" if collision else "MAX STEPS")
            print(f"Episode ended ({reason}) | Reward: {self.episode_reward:.2f} | "
                  f"Steps: {self.current_step} | Distance: {new_distance:.1f}m | "
                  f"Collisions: {self.collision_count}")

        return self.frame_stack.copy(), reward, terminated, truncated, {
            'goal_reached': goal_reached,
            'collision': collision,
            'distance': new_distance,
            'collision_count': self.collision_count
        }

    def _get_images(self):
        """Single API call: RGB scene → grayscale (observation) + raw depth (penalty).

        Depth is returned in raw metres and is NOT normalised — it is used only
        for computing the proximity reward, never fed to the model.
        """
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False),
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False),
        ])

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
            # Fallback: treat everything as far away so no penalty is applied
            depth = np.full((self.img_height, self.img_width), 100.0, dtype=np.float32)

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
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
