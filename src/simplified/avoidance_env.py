"""
Simplified Obstacle Avoidance Environment

Navigation is handled by a simple controller (go toward goal).
The RL model ONLY learns obstacle avoidance from the camera image.

Controller: yaws toward goal, flies forward at base_speed
RL Model:   sees depth camera, outputs lateral + vertical correction (body frame)
Combined:   body_velocity = (forward, lateral, altitude_hold + vertical)
"""

import math
import time
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

        # Connect to AirSim
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected to AirSim!")

        self.camera_name = "front_center"
        self.goal_pos = None

        # Observation: Depth camera image (single channel)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.img_height, self.img_width, 1),
            dtype=np.uint8
        )

        # Action: lateral correction + vertical correction
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32
        )

        self.current_step = 0
        self.episode_reward = 0.0
        self.previous_distance = None
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

        # Write goal for monitor script
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

        # Visual marker
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

        position = self._get_position()
        self.previous_distance = np.linalg.norm(position[:2] - self.goal_pos[:2])

        obs = self._get_camera_image()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        position = self._get_position()

        # Controller: yaw toward goal
        dx = self.goal_pos[0] - position[0]
        dy = self.goal_pos[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)
        yaw_to_goal = math.degrees(math.atan2(dy, dx))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_to_goal)

        # RL model correction (body frame: lateral = left/right, vertical = up/down)
        lateral = float(action[0]) * self.lateral_scale
        vertical = float(action[1]) * self.vertical_scale

        # Body frame velocities: x=forward, y=right, z=down
        body_vx = self.base_speed
        body_vy = lateral

        # Altitude hold (don't let model push drone into the ground)
        altitude_error = self.cruising_altitude - position[2]
        base_vz = float(np.clip(altitude_error * 0.5, -1.0, 1.0))
        body_vz = base_vz + vertical
        if position[2] > self.cruising_altitude + 1.0:
            body_vz = min(body_vz, base_vz)

        # Debug output
        if self.current_step % 25 == 1:
            print(f"  Step {self.current_step}: dist={dist_xy:.1f}m "
                  f"body=({body_vx:.2f},{body_vy:.2f},{body_vz:.2f}) "
                  f"action=({action[0]:.2f},{action[1]:.2f})")

        # Execute movement in body frame (long duration so it persists during camera capture)
        self.client.moveByVelocityBodyFrameAsync(
            float(body_vx), float(body_vy), float(body_vz), 5.0,
            yaw_mode=yaw_mode
        )

        # Get observation (drone keeps flying at commanded velocity during capture)
        obs = self._get_camera_image()

        # Check collision
        collision = self.client.simGetCollisionInfo().has_collided
        if collision:
            self.collision_count += 1

        # Check goal
        new_position = self._get_position()
        new_distance = np.linalg.norm(new_position[:2] - self.goal_pos[:2])
        goal_reached = new_distance < self.goal_radius

        if goal_reached:
            print(f"Goal reached! Steps: {self.current_step}")

        # Reward (collision + action penalty only)
        reward = 0.0
        if collision:
            reward -= 25.0
        reward -= float(np.linalg.norm(action)) * 2.0

        self.episode_reward += reward

        # Termination
        terminated = goal_reached
        truncated = self.current_step >= self.max_steps

        if terminated or truncated:
            reason = "GOAL" if goal_reached else "MAX STEPS"
            print(f"Episode ended ({reason}) | Reward: {self.episode_reward:.2f} | "
                  f"Steps: {self.current_step} | Distance: {new_distance:.1f}m | "
                  f"Collisions: {self.collision_count}")

        self.previous_distance = new_distance

        return obs, reward, terminated, truncated, {
            'goal_reached': goal_reached,
            'collision': collision,
            'distance': new_distance,
            'collision_count': self.collision_count
        }

    def _get_camera_image(self):
        """Get depth camera image."""
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.DepthPerspective, True, False)
        ])

        if responses and responses[0].width > 0:
            # Depth comes as float array
            depth = airsim.list_to_2d_float_array(responses[0].image_data_float,
                                                  responses[0].width, responses[0].height)
            # Clamp and normalize to 0-255
            depth = np.clip(depth, 0, 100)
            depth = (depth / 100.0 * 255).astype(np.uint8)
            depth_resized = cv2.resize(depth, (self.img_width, self.img_height))
            return depth_resized[:, :, np.newaxis]
        else:
            return np.zeros((self.img_height, self.img_width, 1), dtype=np.uint8)

    def _get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
