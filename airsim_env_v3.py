"""
AirSim Drone Navigation Environment with Height Awareness
Supports adaptive navigation across different environments with varying height constraints
"""

import gymnasium as gym
from gymnasium import spaces
import airsim
import numpy as np
import cv2
import time


class AirSimDroneEnv(gym.Env):
    """
    Gym environment for AirSim drone navigation with height-aware observations.
    Enables training models that adapt to different flight altitude constraints.
    """

    metadata = {'render_modes': ['rgb_array']}

    def __init__(self,
                 randomize_goals=True,
                 randomize_height_bounds=True,
                 goal_range_x=(5, 20),
                 goal_range_y=(5, 20),
                 height_bound_ranges={
                     'max_height': (-1.0, -1.8),
                     'min_height': (-2.5, -4.0)
                 },
                 default_max_height=-1.5,
                 default_min_height=-3.0,
                 img_height=84,
                 img_width=84,
                 max_steps=500,
                 goal_radius=2.0,
                 curriculum_learning=False,
                 curriculum_start_distance=10.0,
                 curriculum_end_distance=30.0,
                 curriculum_timesteps=200000):

        super(AirSimDroneEnv, self).__init__()

        self.randomize_goals = randomize_goals
        self.randomize_height_bounds = randomize_height_bounds
        self.goal_range_x = goal_range_x
        self.goal_range_y = goal_range_y
        self.height_bound_ranges = height_bound_ranges
        self.default_max_height = default_max_height
        self.default_min_height = default_min_height
        self.img_height = img_height
        self.img_width = img_width
        self.max_steps = max_steps
        self.goal_radius = goal_radius

        # Curriculum learning parameters
        self.curriculum_learning = curriculum_learning
        self.curriculum_start_distance = curriculum_start_distance
        self.curriculum_end_distance = curriculum_end_distance
        self.curriculum_timesteps = curriculum_timesteps
        self.total_timesteps = 0

        # Drone always starts at origin (0, 0) after takeoff
        # Curriculum controls goal distance from origin
        self.min_start_goal_distance = 20.0  # Will be overridden by curriculum if enabled

        self.goal_pos = None
        self.max_height = None
        self.min_height = None

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("Connected to AirSim!")

        # Verify camera orientation - should be forward-facing
        # In AirSim, "0" or "front_center" is typically the forward camera
        # Let's verify it exists and log its orientation
        self.camera_name = "front_center"
        try:
            # Try to get camera info to verify it exists
            camera_info = self.client.simGetCameraInfo(self.camera_name)
            print(f"[CAMERA] '{self.camera_name}' camera found - FOV: {camera_info.fov:.1f}°")
            print(f"[CAMERA] Position relative to drone: ({camera_info.pose.position.x_val:.2f}, {camera_info.pose.position.y_val:.2f}, {camera_info.pose.position.z_val:.2f})")

            # Orientation quaternion - (w, x, y, z)
            # Default forward-facing should be approximately (1, 0, 0, 0) or (0.7, 0, 0, 0.7) for 90° down tilt
            qw = camera_info.pose.orientation.w_val
            qx = camera_info.pose.orientation.x_val
            qy = camera_info.pose.orientation.y_val
            qz = camera_info.pose.orientation.z_val
            print(f"[CAMERA] Orientation quaternion: w={qw:.2f}, x={qx:.2f}, y={qy:.2f}, z={qz:.2f}")

            # Check if camera might be pointing down (common mistake)
            # Downward camera has pitch ~90° (qw≈0.7, qy≈0.7)
            if abs(qy) > 0.5:
                print("[WARNING] Camera appears to be tilted downward significantly!")
                print("[WARNING] For forward navigation, camera should face forward (horizontal)")

        except Exception as e:
            print(f"[WARNING] Could not verify '{self.camera_name}' camera: {e}")
            print("[WARNING] Will try to use it anyway - check your AirSim settings.json")
            self.camera_name = "0"  # Fallback to default camera

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(4,),  # vx, vy, vz, yaw_rate
            dtype=np.float32
        )

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0,
                high=255,
                shape=(img_height, img_width, 3),
                dtype=np.uint8
            ),
            'vector': spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(13,),  # Updated: now includes yaw and relative_yaw
                dtype=np.float32
            )
        })

        self.current_step = 0
        self.previous_distance = None
        self.previous_xy_distance = None
        self.previous_height = None
        self.collision_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Update curriculum if enabled
        if self.curriculum_learning:
            progress = min(1.0, self.total_timesteps / self.curriculum_timesteps)
            # Linear interpolation from start to end distance
            self.min_start_goal_distance = (
                self.curriculum_start_distance +
                progress * (self.curriculum_end_distance - self.curriculum_start_distance)
            )
            # Also adjust goal range to match
            current_max_range = 10 + progress * 20  # From 10m to 30m
            self.goal_range_x = (self.min_start_goal_distance * 0.7, current_max_range)
            self.goal_range_y = (self.min_start_goal_distance * 0.7, current_max_range)

        if self.randomize_height_bounds:
            self.max_height = np.random.uniform(
                self.height_bound_ranges['max_height'][0],
                self.height_bound_ranges['max_height'][1]
            )
            self.min_height = np.random.uniform(
                self.height_bound_ranges['min_height'][0],
                self.height_bound_ranges['min_height'][1]
            )
            print(f"[Episode] Height bounds: [{self.min_height:.2f}, {self.max_height:.2f}]")
        else:
            self.max_height = self.default_max_height
            self.min_height = self.default_min_height

        # Drone always starts at origin after takeoff
        # No need to track start position - drone just needs to reach goal from wherever it is

        if self.randomize_goals:
            # Generate goal at specified distance from origin (curriculum-controlled)
            max_attempts = 50
            for attempt in range(max_attempts):
                # Random angle
                angle = np.random.uniform(0, 2 * np.pi)
                # Distance based on curriculum
                distance = np.random.uniform(self.min_start_goal_distance, self.min_start_goal_distance + 5.0)

                goal_x = distance * np.cos(angle)
                goal_y = distance * np.sin(angle)

                # Goal altitude MUST be within allowed flight zone (not ground!)
                # Use 0.2m padding from ceiling/floor for safety
                # With 1.5m range and 0.4m total padding, goals have 1.1m of vertical space
                goal_z = np.random.uniform(self.min_height + 0.2, self.max_height - 0.2)

                self.goal_pos = np.array([goal_x, goal_y, goal_z], dtype=np.float32)
                break
            else:
                # Fallback - use center altitude (optimal)
                target_altitude = (self.max_height + self.min_height) / 2.0
                self.goal_pos = np.array([self.min_start_goal_distance, 0.0, target_altitude], dtype=np.float32)
        else:
            # Fixed goal when not randomizing
            target_altitude = (self.default_max_height + self.default_min_height) / 2.0
            self.goal_pos = np.array([15.0, 15.0, target_altitude], dtype=np.float32)

        # Calculate distance from origin
        distance = np.linalg.norm(self.goal_pos)

        # Determine if goal is near optimal altitude
        target_altitude = (self.max_height + self.min_height) / 2.0
        altitude_deviation = abs(self.goal_pos[2] - target_altitude)
        goal_type = "OPTIMAL" if altitude_deviation < 0.2 else "OFF-CENTER"

        print(f"[Episode] Goal: ({self.goal_pos[0]:.1f}, {self.goal_pos[1]:.1f}, {self.goal_pos[2]:.1f}) [{goal_type}]")
        print(f"[Episode] Distance from origin: {distance:.1f}m")

        if self.curriculum_learning:
            progress = min(100, (self.total_timesteps / self.curriculum_timesteps) * 100)
            print(f"[Episode] Curriculum: {progress:.0f}% | Min dist: {self.min_start_goal_distance:.1f}m")
        else:
            print(f"[Episode] Goal radius: {self.goal_radius}m")

        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Clear any previous markers
        self.client.simFlushPersistentMarkers()

        # Drone takes off at origin
        self.client.takeoffAsync().join()

        # CRITICAL: Move to optimal altitude (center of allowed range)
        # This immediately gives drone the best altitude reward and teaches it the target height
        target_altitude = (self.max_height + self.min_height) / 2.0
        self.client.moveToZAsync(target_altitude, 2.0).join()

        # CRITICAL: Rotate drone to face the goal
        # This makes the camera immediately useful and teaches camera-based navigation
        import math
        yaw_to_goal = math.atan2(self.goal_pos[1], self.goal_pos[0])
        yaw_degrees = math.degrees(yaw_to_goal)
        self.client.rotateToYawAsync(yaw_degrees, timeout_sec=2.0).join()

        # Explicitly command hover to stabilize
        self.client.hoverAsync().join()
        time.sleep(0.5)  # Extra stabilization time

        print(f"[RESET] Drone positioned at optimal altitude: {target_altitude:.2f}m (center of {self.min_height:.2f} to {self.max_height:.2f})")
        print(f"[RESET] Drone rotated to face goal: {yaw_degrees:.1f}° (camera pointing at goal)")

        # Add visual marker at GOAL position (Red)
        self.client.simPlotPoints(
            points=[airsim.Vector3r(float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.goal_pos[2]))],
            color_rgba=[1.0, 0.0, 0.0, 1.0],  # Red
            size=40,
            duration=-1,
            is_persistent=True
        )

        # Add circle around goal to show radius
        circle_points = []
        for theta in np.linspace(0, 2*np.pi, 20):
            circle_points.append(
                airsim.Vector3r(
                    float(self.goal_pos[0] + self.goal_radius * np.cos(theta)),
                    float(self.goal_pos[1] + self.goal_radius * np.sin(theta)),
                    float(self.goal_pos[2])
                )
            )

        self.client.simPlotLineStrip(
            points=circle_points,
            color_rgba=[1.0, 0.0, 0.0, 0.5],  # Semi-transparent red
            thickness=5,
            duration=-1,
            is_persistent=True
        )

        # Add flight corridor guide lines from origin to goal XY
        # These show the allowed altitude range (ceiling, center, floor)
        print(f"[VISUAL] Drawing corridor lines - Center: {target_altitude:.2f}m, Ceiling: {self.max_height:.2f}m, Floor: {self.min_height:.2f}m")

        # 1. Center line (optimal altitude) - Green
        center_line = [
            airsim.Vector3r(0.0, 0.0, float(target_altitude)),
            airsim.Vector3r(float(self.goal_pos[0]), float(self.goal_pos[1]), float(target_altitude))
        ]
        self.client.simPlotLineStrip(
            points=center_line,
            color_rgba=[0.0, 1.0, 0.0, 0.8],  # Brighter green
            thickness=5,
            duration=-1,
            is_persistent=True
        )

        # 2. Ceiling line (max_height) - Cyan
        ceiling_line = [
            airsim.Vector3r(0.0, 0.0, float(self.max_height)),
            airsim.Vector3r(float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.max_height))
        ]
        self.client.simPlotLineStrip(
            points=ceiling_line,
            color_rgba=[0.0, 1.0, 1.0, 0.8],  # Brighter cyan
            thickness=4,
            duration=-1,
            is_persistent=True
        )

        # 3. Floor line (min_height) - Orange
        floor_line = [
            airsim.Vector3r(0.0, 0.0, float(self.min_height)),
            airsim.Vector3r(float(self.goal_pos[0]), float(self.goal_pos[1]), float(self.min_height))
        ]
        self.client.simPlotLineStrip(
            points=floor_line,
            color_rgba=[1.0, 0.5, 0.0, 0.8],  # Brighter orange
            thickness=4,
            duration=-1,
            is_persistent=True
        )

        # Give Unreal Engine time to render the markers and lines
        time.sleep(1.0)

        self.current_step = 0
        self.collision_count = 0
        self.previous_distance = self._get_distance_to_goal()
        self.previous_height = self._get_position()[2]

        # Initialize XY distance tracking
        current_pos = self._get_position()
        self.previous_xy_distance = np.linalg.norm(current_pos[:2] - self.goal_pos[:2])

        # Track previous position for velocity calculation
        self.previous_position = current_pos.copy()

        # Initialize episode reward tracking
        self.episode_reward = 0.0

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        self.current_step += 1
        self.total_timesteps += 1  # Track for curriculum

        # CRITICAL FIX: Force hovering for first 10 steps of each episode
        # This gives the drone time to stabilize after takeoff
        # and prevents untrained network from immediately crashing
        if self.current_step <= 10:
            vx = float(action[0]) * 3.0
            vy = float(action[1]) * 3.0
            vz = 0.0  # FORCE HOVER - no vertical movement
            yaw_rate = float(action[3]) * 45.0  # Allow rotation during warmup

            if self.current_step == 1:
                print(f"[WARM-UP] Forcing hover for first 10 steps. Original action[2]={action[2]:.3f}, yaw_rate={yaw_rate:.1f}°/s")
        else:
            # Scale actions to velocities normally after warm-up
            # Horizontal: ±3.0 m/s (full speed for navigation)
            # Vertical: ±0.3 m/s (MUCH slower to prevent instant crashes)
            # Yaw: ±45°/s (moderate rotation speed)
            vx = float(action[0]) * 3.0
            vy = float(action[1]) * 3.0
            vz = float(action[2]) * 0.3
            yaw_rate = float(action[3]) * 45.0  # degrees per second

        # Move with velocity and yaw rate
        yaw_mode = airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate)
        self.client.moveByVelocityAsync(vx, vy, vz, 0.5, yaw_mode=yaw_mode).join()

        obs = self._get_obs()
        position = self._get_position()
        distance = self._get_distance_to_goal()
        collision = self._check_collision()
        extreme_altitude = self._check_extreme_altitude(position)

        goal_reached = self._check_goal_reached(distance)

        if goal_reached:
            print(f"[SUCCESS] Goal reached! Distance: {distance:.2f}m < {self.goal_radius}m | Steps: {self.current_step}")

        if collision:
            print(f"[COLLISION] Episode ended - Crashed at step {self.current_step} | Position: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")

        if extreme_altitude:
            z_pos = position[2]
            height_range = abs(self.max_height - self.min_height)
            extreme_ceiling = self.max_height - (5.0 * height_range)
            print(f"[EXTREME ALTITUDE] Episode ended - Drone flew WAY too high: Z={z_pos:.2f}m (extreme ceiling: {extreme_ceiling:.2f}m, normal ceiling: {self.max_height:.2f}m)")

        reward = self._calculate_reward(position, distance, collision, extreme_altitude)

        # Accumulate episode reward
        self.episode_reward += reward

        terminated = goal_reached or collision or extreme_altitude
        truncated = self.current_step >= self.max_steps

        # Print total episode reward when episode ends
        if terminated or truncated:
            print(f"[EPISODE REWARD] Total: {self.episode_reward:.2f} | Steps: {self.current_step}")

        # Diagnostic: Log premature episode endings
        if terminated and self.current_step < 20:
            print(f"[DEBUG] Short episode at step {self.current_step}: goal={goal_reached}, collision={collision}, extreme_alt={extreme_altitude}")
        if truncated:
            print(f"[TRUNCATED] Episode reached max steps ({self.max_steps}) | Distance to goal: {distance:.2f}m")

        self.previous_distance = distance
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """Build observation with camera image and state vector including height bounds"""
        image = self._get_camera_image()
        current_pos = self._get_position()

        # Get drone orientation
        state = self.client.getMultirotorState()
        orientation = state.kinematics_estimated.orientation
        # Convert quaternion to yaw angle (rotation around Z axis)
        import math
        # Yaw from quaternion: atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
        yaw = math.atan2(
            2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
            1.0 - 2.0 * (orientation.y_val * orientation.y_val + orientation.z_val * orientation.z_val)
        )

        rel_pos = self.goal_pos - current_pos
        distance = np.linalg.norm(rel_pos)
        yaw_to_goal = np.arctan2(rel_pos[1], rel_pos[0])

        # Relative angle: how much the drone needs to turn to face the goal
        relative_yaw = yaw_to_goal - yaw
        # Normalize to [-pi, pi]
        while relative_yaw > np.pi:
            relative_yaw -= 2 * np.pi
        while relative_yaw < -np.pi:
            relative_yaw += 2 * np.pi

        distance_to_ceiling = current_pos[2] - self.max_height
        distance_to_floor = self.min_height - current_pos[2]

        goal_vector = np.array([
            rel_pos[0],
            rel_pos[1],
            rel_pos[2],
            distance,
            yaw_to_goal,
            current_pos[2],
            self.goal_pos[2],
            self.max_height,
            self.min_height,
            distance_to_ceiling,
            distance_to_floor,
            yaw,              # Current drone orientation
            relative_yaw      # Angle to turn to face goal
        ], dtype=np.float32)

        return {
            'image': image,
            'vector': goal_vector
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

    def _get_distance_to_goal(self):
        current_pos = self._get_position()
        return np.linalg.norm(current_pos - self.goal_pos)

    def _check_collision(self):
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided:
            self.collision_count += 1
            return True
        return False

    def _check_extreme_altitude(self, position):
        """
        Check if drone has exceeded extreme altitude boundaries.
        Extreme boundary is 5x the allowed range above the ceiling.
        This prevents the drone from flying way into the sky.

        In NED coordinates: more negative Z = higher altitude
        """
        z_pos = position[2]
        height_range = abs(self.max_height - self.min_height)

        # Extreme ceiling: 5x the range above normal ceiling
        # In NED: "above" = more negative, so subtract
        # e.g., max_height = -1.1, range = 1.5 → extreme = -1.1 - 7.5 = -8.6m
        extreme_ceiling = self.max_height - (5.0 * height_range)

        # Check if drone flew way too HIGH (too negative Z in NED)
        if z_pos < extreme_ceiling:
            return True
        return False

    def _check_goal_reached(self, distance):
        return distance < self.goal_radius

    def _calculate_reward(self, position, distance, collision, extreme_altitude):
        """
        Reward function with altitude maintenance incentive.
        Encourages flying at center of safe zone while progressing toward goal.
        """
        reward = 0.0

        z_pos = position[2]

        # CRITICAL: Check if drone is within bounds FIRST
        # Only give progress rewards if staying within allowed altitude
        in_bounds = self.min_height <= z_pos <= self.max_height

        if in_bounds:
            # ONLY reward progress when within bounds
            if self.previous_distance is not None:
                # Reward for total 3D progress toward goal
                progress = self.previous_distance - distance
                reward += progress * 20.0

                # ADDITIONAL: Reward specifically for horizontal (XY) progress
                # This helps drone learn to move toward goal even if altitude is wrong
                current_xy = position[:2]
                goal_xy = self.goal_pos[:2]
                xy_distance = np.linalg.norm(current_xy - goal_xy)

                if hasattr(self, 'previous_xy_distance'):
                    xy_progress = self.previous_xy_distance - xy_distance
                    # Extra reward for horizontal progress (encourages forward movement)
                    reward += xy_progress * 10.0

                self.previous_xy_distance = xy_distance

            # Forward velocity reward: encourage moving toward goal (where camera can see)
            # Penalize backward movement (flying blind)
            if hasattr(self, 'previous_position'):
                # Calculate actual movement vector
                movement = position - self.previous_position
                movement_xy = movement[:2]  # Only XY movement

                # Direction to goal
                to_goal = self.goal_pos - position
                to_goal_xy = to_goal[:2]

                # Normalize to get direction (avoid division by zero)
                goal_distance_xy = np.linalg.norm(to_goal_xy)
                if goal_distance_xy > 0.1:  # Only if goal is not super close
                    goal_direction = to_goal_xy / goal_distance_xy

                    # Dot product: positive = moving toward goal, negative = moving away
                    forward_velocity = np.dot(movement_xy, goal_direction)

                    # Reward forward movement, penalize backward
                    # This encourages drone to face and move toward goal (using camera vision)
                    if forward_velocity > 0:
                        reward += forward_velocity * 5.0  # Bonus for moving toward goal
                    else:
                        reward += forward_velocity * 10.0  # Stronger penalty for moving backward

            # Reward for facing toward goal
            # Get drone orientation
            state = self.client.getMultirotorState()
            orientation = state.kinematics_estimated.orientation
            import math
            yaw = math.atan2(
                2.0 * (orientation.w_val * orientation.z_val + orientation.x_val * orientation.y_val),
                1.0 - 2.0 * (orientation.y_val * orientation.y_val + orientation.z_val * orientation.z_val)
            )

            # Angle to goal
            to_goal = self.goal_pos - position
            yaw_to_goal = np.arctan2(to_goal[1], to_goal[0])

            # How aligned is the drone with the goal?
            alignment_error = abs(yaw_to_goal - yaw)
            while alignment_error > np.pi:
                alignment_error -= 2 * np.pi
            alignment_error = abs(alignment_error)

            # Reward being aligned with goal (camera pointing toward goal)
            # Max reward when perfectly aligned, zero when perpendicular
            alignment_bonus = (1.0 - alignment_error / np.pi) * 3.0
            reward += alignment_bonus
        else:
            # OUT OF BOUNDS: No progress rewards at all!
            # This makes out-of-bounds flight completely unviable
            # Drone must stay in bounds to get any positive rewards
            if hasattr(self, 'previous_xy_distance'):
                # Still update tracking for when it comes back in bounds
                current_xy = position[:2]
                goal_xy = self.goal_pos[:2]
                self.previous_xy_distance = np.linalg.norm(current_xy - goal_xy)

        # Update previous position for next step
        self.previous_position = position.copy()

        # Small fixed step penalty to encourage efficiency (not distance-based)
        # REDUCED from -0.5 to -0.1 to make staying alive more attractive
        reward -= 0.1

        z_pos = position[2]

        # Survival bonus: reward for staying airborne and within bounds
        # Helps untrained model learn that not crashing = good
        if self.min_height <= z_pos <= self.max_height:
            # Small reward just for staying in valid altitude range
            reward += 0.5

        if collision:
            # Massive penalty: lose all potential future rewards
            # Average successful episode gets ~1000+ total reward over 100 steps
            # Collision should be worse than giving up all that
            reward -= 500.0

        if extreme_altitude:
            # Large penalty for exceeding extreme altitude boundaries
            # This ends the episode - drone flew way too high or too low
            reward -= 300.0

        # DIRECTIONAL vertical movement reward/penalty
        # Reward moving toward target altitude, penalize moving away
        target_height = (self.max_height + self.min_height) / 2.0
        height_range = abs(self.max_height - self.min_height)

        if self.previous_height is not None:
            # Distance from target altitude before and after
            previous_deviation = abs(self.previous_height - target_height)
            current_deviation = abs(z_pos - target_height)

            # Positive if moving toward target, negative if moving away
            altitude_progress = previous_deviation - current_deviation

            # Reward moving toward target altitude, penalize moving away
            # This gives the drone a clear gradient to follow
            reward += altitude_progress * 10.0

        self.previous_height = z_pos

        if z_pos > self.max_height:
            # Exponential penalty outside bounds - gets much worse as you go further
            violation = z_pos - self.max_height
            normalized_violation = violation / (height_range / 2.0)
            # Cap the exponent to prevent overflow (max penalty ~500)
            # exp(5) ≈ 148, so 50 * (exp(5) - 1) ≈ 7350
            capped_violation = min(normalized_violation, 3.0)
            height_penalty = 50.0 * (np.exp(2.0 * capped_violation) - 1.0)
            reward -= height_penalty
        elif z_pos < self.min_height:
            # Exponential penalty below floor
            violation = self.min_height - z_pos
            normalized_violation = violation / (height_range / 2.0)
            # Cap to prevent overflow
            capped_violation = min(normalized_violation, 3.0)
            height_penalty = 50.0 * (np.exp(2.0 * capped_violation) - 1.0)
            reward -= height_penalty
        else:
            # Quadratic reward inside bounds - peaked at center
            deviation_from_target = abs(z_pos - target_height)
            normalized_deviation = deviation_from_target / (height_range / 2.0)
            # Quadratic: 1 - x^2 gives smooth peak at center
            altitude_bonus = 5.0 * (1.0 - normalized_deviation ** 2)
            reward += altitude_bonus

        if distance < self.goal_radius:
            # Large bonus for reaching goal
            reward += 500.0

        # Clip reward to prevent extreme values that cause NaN in training
        # Allow large negative for crashes, but prevent infinity
        reward = np.clip(reward, -1000.0, 1000.0)

        return reward

    def _get_info(self):
        position = self._get_position()
        distance = self._get_distance_to_goal()

        return {
            'position': position.tolist(),
            'goal_position': self.goal_pos.tolist(),
            'distance_to_goal': float(distance),
            'step': self.current_step,
            'collision_count': self.collision_count,
            'height': float(position[2]),
            'max_height': float(self.max_height),
            'min_height': float(self.min_height)
        }

    def render(self):
        return self._get_camera_image()

    def close(self):
        try:
            self.client.armDisarm(False)
            self.client.enableApiControl(False)
        except:
            pass
