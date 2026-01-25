"""
Stage 3 Environment: Goal Seeking + Altitude Constraints
Adds min/max altitude bounds with penalties
No obstacles yet
"""

import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2
import time


class AirSimStage3Env(gym.Env):
    """Stage 3: Learn to navigate while respecting altitude constraints"""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 goal_range_x=(5, 20),
                 goal_range_y=(5, 20),
                 goal_range_z=(-20, -10),  # Safe altitude range (10-20m above ground)
                 goal_radius=3.0,
                 min_altitude=-25.0,  # Floor at 25m altitude
                 max_altitude=-10.0,  # Ceiling at 10m altitude (above Blocks obstacles)
                 show_visual_marker=False):  # Disabled by default for realistic training
        super(AirSimStage3Env, self).__init__()

        self.goal_range_x = goal_range_x
        self.goal_range_y = goal_range_y
        self.goal_range_z = goal_range_z
        self.goal_radius = goal_radius
        self.min_altitude = min_altitude
        self.max_altitude = max_altitude
        self.show_visual_marker = show_visual_marker
        self.img_height = 84
        self.img_width = 84
        self.max_steps = 500

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected to AirSim!")

        self.camera_name = "front_center"
        self.goal_pos = None

        # Action space
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation space: camera + full state + goal info + altitude info
        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(self.img_height, self.img_width, 3),
                dtype=np.uint8
            ),
            'vector': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(21,),  # Full state (13) + goal info (7) + altitude info (1)
                dtype=np.float32
            )
        })

        self.current_step = 0
        self.episode_reward = 0.0
        self.previous_position = None
        self.previous_distance = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Random goal with varying altitude
        goal_x = np.random.uniform(self.goal_range_x[0], self.goal_range_x[1]) * np.random.choice([-1, 1])
        goal_y = np.random.uniform(self.goal_range_y[0], self.goal_range_y[1]) * np.random.choice([-1, 1])
        goal_z = np.random.uniform(self.goal_range_z[0], self.goal_range_z[1])
        self.goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)

        distance = np.linalg.norm(self.goal_pos)
        print(f"\n[STAGE 3] Goal: ({goal_x:.1f}, {goal_y:.1f}, {goal_z:.1f}) | Distance: {distance:.1f}m")
        print(f"[STAGE 3] Altitude bounds: {self.max_altitude:.1f}m to {self.min_altitude:.1f}m")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff to safe starting altitude
        start_altitude = (self.min_altitude + self.max_altitude) / 2.0
        self.client.takeoffAsync().join()
        self.client.moveToZAsync(start_altitude, 2.0).join()

        # Face goal
        import math
        yaw_to_goal = math.atan2(self.goal_pos[1], self.goal_pos[0])
        yaw_degrees = math.degrees(yaw_to_goal)
        self.client.rotateToYawAsync(yaw_degrees, timeout_sec=2.0).join()
        self.client.hoverAsync().join()
        time.sleep(0.5)

        # Visual marker for goal (only for debugging, disabled by default)
        self.client.simFlushPersistentMarkers()
        if self.show_visual_marker:
            self.client.simPlotPoints(
                points=[airsim.Vector3r(float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]))],
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                size=40,
                duration=-1,
                is_persistent=True
            )

        self.current_step = 0
        self.episode_reward = 0.0
        self.previous_position = None
        self.previous_distance = None

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Scale actions
        vx = float(action[0]) * 3.0
        vy = float(action[1]) * 3.0
        vz = float(action[2]) * 0.5  # Slightly more vertical control
        yaw_rate = float(action[3]) * 45.0

        # Move using BODY frame (velocities relative to drone's facing direction)
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.5, yaw_mode=yaw_mode).join()

        obs = self._get_obs()
        position = self._get_position()
        distance = np.linalg.norm(position - self.goal_pos)

        # Check altitude bounds
        extreme_altitude = position[2] > self.max_altitude or position[2] < self.min_altitude

        if extreme_altitude:
            print(f"[ALTITUDE VIOLATION] Alt: {position[2]:.2f}m | Bounds: [{self.min_altitude:.1f}, {self.max_altitude:.1f}]")
            # Auto-correct: Move to safe altitude
            if position[2] > self.max_altitude:
                safe_altitude = self.max_altitude - 0.5
            else:
                safe_altitude = self.min_altitude + 0.5
            self.client.moveToZAsync(safe_altitude, 1.0).join()
            position = self._get_position()  # Update position after correction

        # Check goal
        goal_reached = distance < self.goal_radius

        if goal_reached:
            print(f"[SUCCESS] Goal reached! Distance: {distance:.2f}m | Steps: {self.current_step}")

        # Calculate reward
        reward = self._calculate_reward(position, distance, goal_reached, extreme_altitude)
        self.episode_reward += reward

        terminated = goal_reached
        truncated = self.current_step >= self.max_steps

        if terminated or truncated:
            print(f"[STAGE 3] Episode ended | Reward: {self.episode_reward:.2f} | Steps: {self.current_step}")
            if truncated:
                print(f"  Distance to goal: {distance:.2f}m | Final altitude: {position[2]:.2f}m")

        self.previous_position = position.copy()
        self.previous_distance = distance

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        image = self._get_camera_image()
        state = self.client.getMultirotorState()

        # Position
        position = np.array([
            state.kinematics_estimated.position.x_val,
            state.kinematics_estimated.position.y_val,
            state.kinematics_estimated.position.z_val
        ], dtype=np.float32)

        # Linear velocity
        velocity = np.array([
            state.kinematics_estimated.linear_velocity.x_val,
            state.kinematics_estimated.linear_velocity.y_val,
            state.kinematics_estimated.linear_velocity.z_val
        ], dtype=np.float32)

        # Orientation (roll, pitch, yaw)
        orientation = state.kinematics_estimated.orientation
        import math
        roll = math.atan2(
            2.0 * (orientation.w_val * orientation.x_val + orientation.y_val * orientation.z_val),
            1.0 - 2.0 * (orientation.x_val * orientation.x_val + orientation.y_val * orientation.y_val)
        )
        pitch = math.asin(2.0 * (orientation.w_val * orientation.y_val - orientation.z_val * orientation.x_val))
        yaw = math.atan2(
            2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
            1.0 - 2.0 * (orientation.y_val * orientation.y_val + orientation.z_val * orientation.z_val)
        )

        # Angular velocity
        angular_vel = np.array([
            state.kinematics_estimated.angular_velocity.x_val,
            state.kinematics_estimated.angular_velocity.y_val,
            state.kinematics_estimated.angular_velocity.z_val
        ], dtype=np.float32)

        # Forward speed
        facing_direction = np.array([np.cos(yaw), np.sin(yaw)])
        forward_speed = np.dot(velocity[:2], facing_direction)

        # Goal information
        rel_pos = self.goal_pos - position
        distance = np.linalg.norm(rel_pos)
        yaw_to_goal = np.arctan2(rel_pos[1], rel_pos[0])

        # Relative yaw to goal
        relative_yaw = yaw_to_goal - yaw
        while relative_yaw > np.pi:
            relative_yaw -= 2 * np.pi
        while relative_yaw < -np.pi:
            relative_yaw += 2 * np.pi

        # Altitude info
        altitude_margin = min(abs(position[2] - self.max_altitude), abs(position[2] - self.min_altitude))

        # Full state vector: drone state (13) + goal info (7) + altitude info (1)
        state_vector = np.array([
            # Drone state
            position[0], position[1], position[2],           # Position (3)
            velocity[0], velocity[1], velocity[2],           # Linear velocity (3)
            roll, pitch, yaw,                                # Orientation (3)
            angular_vel[0], angular_vel[1], angular_vel[2],  # Angular velocity (3)
            forward_speed,                                   # Forward speed (1)
            # Goal information
            rel_pos[0], rel_pos[1], rel_pos[2],             # Relative position to goal (3)
            distance,                                        # Distance to goal (1)
            yaw_to_goal,                                    # Direction to goal (1)
            relative_yaw,                                    # Relative yaw (1)
            np.dot(velocity, rel_pos) / (distance + 1e-6),  # Velocity toward goal (1)
            # Altitude constraint info
            altitude_margin                                  # Distance from nearest altitude bound (1)
        ], dtype=np.float32)

        return {
            'image': image,
            'vector': state_vector
        }

    def _get_camera_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)
        ])

        if responses and len(responses[0].image_data_uint8) > 0:
            img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)
            img_resized = cv2.resize(img_rgb, (self.img_width, self.img_height))
            return img_resized
        else:
            return np.zeros((self.img_height, self.img_width, 3), dtype=np.uint8)

    def _get_position(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)

    def _calculate_reward(self, position, distance, goal_reached, extreme_altitude):
        """
        Stage 3 Reward: Goal progress + Altitude constraint penalties
        """
        reward = 0.0

        # Progress toward goal (dominant signal)
        if self.previous_distance is not None:
            progress = self.previous_distance - distance
            reward += progress * 200.0

        # Forward movement bonus
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        import math
        yaw = math.atan2(
            2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
            1.0 - 2.0 * (orientation.y_val * orientation.y_val + orientation.z_val * orientation.z_val)
        )

        if self.previous_position is not None:
            movement = position - self.previous_position
            movement_xy = movement[:2]
            facing_direction = np.array([np.cos(yaw), np.sin(yaw)])
            forward_velocity = np.dot(movement_xy, facing_direction)

            if forward_velocity > 0:
                reward += forward_velocity * 5.0
            else:
                reward += forward_velocity * 10.0

        # Altitude violation penalty (non-terminating)
        if extreme_altitude:
            reward -= 50.0

        # Small penalty for being near altitude bounds (soft constraint)
        altitude_margin = min(abs(position[2] - self.max_altitude), abs(position[2] - self.min_altitude))
        if altitude_margin < 1.0:
            reward -= (1.0 - altitude_margin) * 5.0  # Up to -5 when at boundary

        # Goal bonus
        if goal_reached:
            reward += 1000.0

        # Survival bonus
        reward += 0.5

        # Small step penalty
        reward -= 0.1

        return reward
