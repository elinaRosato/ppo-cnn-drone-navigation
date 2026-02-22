"""
Fly a multi-waypoint mission using controller + trained avoidance model.

The simple controller navigates between waypoints.
The trained model avoids obstacles along the way.

Usage:
    python fly_mission.py --model path/to/model.zip
    python fly_mission.py --model path/to/model.zip --speed 3.0
"""

import os
import math
import time
import argparse
import airsim
import numpy as np
import cv2
from stable_baselines3 import PPO


# --- Default mission waypoints (NED coordinates) ---
# Modify these for your environment
MISSION_WAYPOINTS = [
    (20, 0, -5),      # Forward
    (20, 20, -5),     # Right
    (0, 20, -5),      # Back-left
    (0, 0, -5),       # Return to start
]


STACK_FRAMES = 4
IMG_SIZE = 84


def get_grayscale_frame(client, camera_name="front_center"):
    """Capture RGB scene image and convert to grayscale (84x84)."""
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])

    if responses and responses[0].width > 0:
        r = responses[0]
        raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
        n_ch = len(raw) // (r.width * r.height)
        img = raw.reshape(r.height, r.width, n_ch)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if n_ch == 4 else cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    else:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)


def init_frame_stack(client, camera_name="front_center"):
    """Build initial frame stack by repeating the first captured frame."""
    first = get_grayscale_frame(client, camera_name)
    stack = np.zeros((STACK_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i in range(STACK_FRAMES):
        stack[i] = first
    return stack


def update_frame_stack(stack, new_frame):
    """Shift stack left and insert the newest frame at the end."""
    stack = np.roll(stack, shift=-1, axis=0)
    stack[-1] = new_frame
    return stack


def get_position(client):
    """Get current drone position."""
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)


def fly_to_waypoint(client, model, waypoint, base_speed=1.0,
                    lateral_scale=1.0, vertical_scale=0.5,
                    cruising_altitude=-5.0,
                    goal_radius=3.0, max_steps=1000):
    """
    Fly to a single waypoint using controller + trained model.

    Controller: flies toward waypoint at base_speed
    Model: lateral + vertical correction (perpendicular to goal direction)
    Yaw: faces actual movement direction
    """
    waypoint = np.array(waypoint, dtype=np.float32)
    collisions = 0

    # Initialise frame stack matching training format: (4, 84, 84) CHW
    frame_stack = init_frame_stack(client)

    for step in range(max_steps):
        # Update frame stack with latest grayscale frame
        frame_stack = update_frame_stack(frame_stack, get_grayscale_frame(client))

        # Model predicts correction from stacked frames
        action, _ = model.predict(frame_stack, deterministic=True)

        # Current position
        position = get_position(client)

        # Direction toward waypoint
        dx = waypoint[0] - position[0]
        dy = waypoint[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)

        # Check arrival
        if dist_xy < goal_radius:
            return {'success': True, 'steps': step + 1, 'collisions': collisions}

        # Unit vector toward goal and perpendicular (left)
        ux, uy = dx / dist_xy, dy / dist_xy
        px, py = -uy, ux

        # Model correction
        lateral = float(action[0]) * lateral_scale
        vertical = float(action[1]) * vertical_scale

        # Combined velocity: forward toward goal + lateral perpendicular
        vel_x = base_speed * ux + lateral * px
        vel_y = base_speed * uy + lateral * py
        speed_xy = math.sqrt(vel_x * vel_x + vel_y * vel_y)

        # Yaw faces actual movement direction
        yaw_deg = math.degrees(math.atan2(vel_y, vel_x))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        # Body frame: all horizontal speed is forward, no lateral
        body_vx = speed_xy
        body_vy = 0.0

        # Altitude hold
        altitude_error = cruising_altitude - position[2]
        body_vz = float(np.clip(altitude_error * 0.5, -1.0, 1.0)) + vertical

        # Execute in body frame
        client.moveByVelocityBodyFrameAsync(
            float(body_vx), float(body_vy), float(body_vz), 5.0,
            yaw_mode=yaw_mode
        )

        # Check collision
        if client.simGetCollisionInfo().has_collided:
            collisions += 1
            print(f"    Collision at step {step + 1}!")

        # Progress report every 50 steps
        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}: Distance = {dist_xy:.1f}m")

    return {'success': False, 'steps': max_steps, 'collisions': collisions}


def main(model_path, base_speed=2.0):
    print("=" * 70)
    print("MULTI-WAYPOINT MISSION")
    print("=" * 70)
    print(f"\nModel: {model_path}")
    print(f"Speed: {base_speed} m/s")
    print(f"Waypoints: {len(MISSION_WAYPOINTS)}")
    for i, wp in enumerate(MISSION_WAYPOINTS):
        print(f"  {i + 1}. ({wp[0]}, {wp[1]}, {wp[2]})")
    print("=" * 70)

    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("\nConnected to AirSim!")

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Takeoff
    print("\nTaking off...")
    client.takeoffAsync().join()
    client.moveToZAsync(-5.0, 2.0).join()
    time.sleep(1.0)

    # Visualize waypoints
    client.simFlushPersistentMarkers()
    for i, wp in enumerate(MISSION_WAYPOINTS):
        client.simPlotPoints(
            points=[airsim.Vector3r(float(wp[0]), float(wp[1]), float(wp[2]))],
            color_rgba=[0.0, 1.0, 0.0, 1.0],
            size=40,
            duration=-1,
            is_persistent=True
        )

    # Fly mission
    total_collisions = 0
    waypoints_reached = 0

    input("\nPress ENTER to start mission...")

    for i, waypoint in enumerate(MISSION_WAYPOINTS):
        print(f"\n--- Waypoint {i + 1}/{len(MISSION_WAYPOINTS)}: "
              f"({waypoint[0]}, {waypoint[1]}, {waypoint[2]}) ---")

        result = fly_to_waypoint(
            client, model, waypoint,
            base_speed=base_speed
        )

        total_collisions += result['collisions']

        if result['success']:
            waypoints_reached += 1
            print(f"  ✅ Reached in {result['steps']} steps "
                  f"({result['collisions']} collisions)")
        else:
            print(f"  ❌ Failed after {result['steps']} steps "
                  f"({result['collisions']} collisions)")

    # Land
    print("\nLanding...")
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)

    # Summary
    print(f"\n{'=' * 70}")
    print("MISSION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Waypoints Reached: {waypoints_reached}/{len(MISSION_WAYPOINTS)}")
    print(f"Total Collisions:  {total_collisions}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fly a multi-waypoint mission')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: latest)')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='Base flight speed in m/s (default: 2.0)')
    args = parser.parse_args()

    # Find model if not specified
    model_path = args.model
    if model_path is None:
        base_dir = "./models_simplified"
        if os.path.exists(base_dir):
            run_dirs = [d for d in os.listdir(base_dir)
                        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]
            if run_dirs:
                run_dirs.sort()
                latest = os.path.join(base_dir, run_dirs[-1])
                final = os.path.join(latest, "simplified_avoidance_final.zip")
                if os.path.exists(final):
                    model_path = final

    if model_path is None:
        print("No model found! Train first with: python train.py")
    else:
        main(model_path=model_path, base_speed=args.speed)
