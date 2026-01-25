"""
Stage 1 Environment: Forward Movement Only
Simplest possible task - just learn to move forward

IMPORTANT: Use an EMPTY environment (no obstacles) for this stage!
Recommended: Flat terrain or open space in your Unreal project.
Obstacles are introduced in Stage 4.
"""

import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2
import time


class AirSimStage1Env(gym.Env):
    """Stage 1: Learn forward movement is rewarded"""

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self):
        super(AirSimStage1Env, self).__init__()

        self.img_height = 84
        self.img_width = 84
        self.max_steps = 500

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected to AirSim!")

        # Camera
        self.camera_name = "front_center"

        # Action space: vx, vy, vz, yaw_rate
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),
            dtype=np.float32
        )

        # Observation space: camera + full state
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
                shape=(21,),  # Full state (13) + goal info (6) + altitude info (2) for compatibility with all stages
                dtype=np.float32
            )
        })

        self.current_step = 0
        self.episode_reward = 0.0
        self.previous_position = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        print(f"\n[STAGE 1] Episode starting - Learn to move forward!")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Takeoff and fly to safe altitude (above obstacles in Blocks environment)
        self.client.takeoffAsync().join()
        # Move to -15m altitude (15m above ground in NED coordinates)
        self.client.moveToZAsync(-15, 5).join()  # altitude, velocity
        self.client.hoverAsync().join()
        time.sleep(0.5)

        self.current_step = 0
        self.episode_reward = 0.0
        self.previous_position = None

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.current_step += 1

        # Scale actions
        vx = float(action[0]) * 3.0  # Forward/backward
        vy = float(action[1]) * 3.0  # Left/right
        vz = float(action[2]) * 0.3  # Up/down
        yaw_rate = float(action[3]) * 45.0  # Rotation

        # Move using BODY frame (velocities relative to drone's facing direction)
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityBodyFrameAsync(vx, vy, vz, 0.5, yaw_mode=yaw_mode).join()

        obs = self._get_obs()
        position = self._get_position()

        # Calculate reward
        reward = self._calculate_reward(position, vx, vy)

        self.episode_reward += reward

        # Only end on max steps
        terminated = False
        truncated = self.current_step >= self.max_steps

        if truncated:
            print(f"[STAGE 1] Episode ended | Reward: {self.episode_reward:.2f} | Steps: {self.current_step}")

        self.previous_position = position.copy()

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

        # Forward speed (velocity in the direction the drone is facing)
        facing_direction = np.array([np.cos(yaw), np.sin(yaw)])
        forward_speed = np.dot(velocity[:2], facing_direction)

        # Full state vector (21 values for compatibility with all stages)
        state_vector = np.array([
            # Drone state (13 values)
            position[0], position[1], position[2],           # Position (3)
            velocity[0], velocity[1], velocity[2],           # Linear velocity (3)
            roll, pitch, yaw,                                # Orientation (3)
            angular_vel[0], angular_vel[1], angular_vel[2],  # Angular velocity (3)
            forward_speed,                                   # Forward speed (1)
            # Goal info placeholders (6 values) - zeros in Stage 1, used in Stage 2+
            0.0, 0.0,                                        # Relative position to goal (2)
            0.0,                                             # Distance to goal (1)
            0.0,                                             # Direction to goal (1)
            0.0,                                             # Relative yaw (1)
            0.0,                                             # Velocity toward goal (1)
            # Altitude info placeholders (2 values) - zeros in Stage 1-2, used in Stage 3+
            0.0,                                             # Altitude bounds info (1)
            0.0                                              # Distance to altitude bounds (1)
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

    def _calculate_reward(self, position, vx, vy):
        """
        Stage 1 Reward: ONLY reward forward movement
        - Positive reward for moving forward (vx > 0)
        - Small penalty for moving backward
        - Small penalty for lateral movement
        """
        reward = 0.0

        # Get drone's facing direction
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        import math
        yaw = math.atan2(
            2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
            1.0 - 2.0 * (orientation.y_val * orientation.y_val + orientation.z_val * orientation.z_val)
        )

        if self.previous_position is not None:
            # Calculate actual movement
            movement = position - self.previous_position
            movement_xy = movement[:2]

            # Drone's facing direction
            facing_direction = np.array([np.cos(yaw), np.sin(yaw)])

            # Forward velocity component
            forward_velocity = np.dot(movement_xy, facing_direction)

            # MAIN REWARD: Moving forward
            if forward_velocity > 0:
                reward += forward_velocity * 50.0  # Big reward for forward movement!
            else:
                reward += forward_velocity * 20.0  # Penalty for backward

            # Small bonus for just existing (don't crash)
            reward += 0.5

        return reward
