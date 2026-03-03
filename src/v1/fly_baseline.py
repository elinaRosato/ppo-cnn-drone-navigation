"""
Fly the drone from start to a goal using ONLY the simple controller.
No RL model corrections — pure straight-line navigation.
Use this as a baseline to compare against the trained model.

Usage:
    python fly_baseline.py                    # Single run, random direction
    python fly_baseline.py --episodes 5       # Multiple runs
    python fly_baseline.py --distance 30      # Custom distance
"""

import math
import time
import argparse
import airsim
import numpy as np


def get_position(client):
    state = client.getMultirotorState()
    pos = state.kinematics_estimated.position
    return np.array([pos.x_val, pos.y_val, pos.z_val], dtype=np.float32)


def fly_baseline(client, goal_pos, base_speed=2.0, cruising_altitude=-5.0,
                 goal_radius=1.5, max_steps=500):
    """Fly to goal using only the simple controller (no model)."""
    collisions = 0

    for step in range(max_steps):
        position = get_position(client)

        dx = goal_pos[0] - position[0]
        dy = goal_pos[1] - position[1]
        dist_xy = max(math.sqrt(dx * dx + dy * dy), 0.01)

        if dist_xy < goal_radius:
            return {'success': True, 'steps': step + 1, 'collisions': collisions,
                    'distance': dist_xy}

        # Unit vector toward goal
        ux = dx / dist_xy
        uy = dy / dist_xy

        # Velocity straight toward goal
        vx = ux * base_speed
        vy = uy * base_speed

        # Altitude hold (P controller)
        altitude_error = cruising_altitude - position[2]
        vz = float(np.clip(altitude_error * 2.0, -2.0, 2.0))

        # Face goal direction
        yaw_deg = math.degrees(math.atan2(vy, vx))
        yaw_mode = airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)

        client.moveByVelocityAsync(
            float(vx), float(vy), float(vz), 0.2,
            yaw_mode=yaw_mode
        ).join()

        # Check collision
        if client.simGetCollisionInfo().has_collided:
            collisions += 1

        if (step + 1) % 50 == 0:
            print(f"    Step {step + 1}: Distance = {dist_xy:.1f}m")

    final_dist = float(np.linalg.norm(get_position(client)[:2] - goal_pos[:2]))
    return {'success': False, 'steps': max_steps, 'collisions': collisions,
            'distance': final_dist}


def main(episodes=1, distance=50.0, base_speed=2.0):
    cruising_altitude = -5.0

    print("=" * 70)
    print("BASELINE FLIGHT (no model — controller only)")
    print("=" * 70)
    print(f"Episodes:  {episodes}")
    print(f"Distance:  {distance}m")
    print(f"Speed:     {base_speed} m/s")
    print(f"Altitude:  {cruising_altitude}m")
    print("=" * 70)

    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("\nConnected to AirSim!")

    results = {
        'goals_reached': 0,
        'collisions': 0,
        'steps': [],
    }

    for ep in range(episodes):
        # Random goal direction, fixed distance
        angle = np.random.uniform(0, 2 * np.pi)
        goal_x = distance * np.cos(angle)
        goal_y = distance * np.sin(angle)
        goal_pos = np.array([goal_x, goal_y, cruising_altitude], dtype=np.float32)

        print(f"\n--- Episode {ep + 1}/{episodes} ---")
        print(f"Goal: ({goal_x:.1f}, {goal_y:.1f}, {cruising_altitude:.1f})")

        # Reset and takeoff
        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        client.moveToZAsync(cruising_altitude, 2.0).join()

        yaw_to_goal = math.degrees(math.atan2(goal_y, goal_x))
        client.rotateToYawAsync(yaw_to_goal, timeout_sec=2.0).join()
        client.hoverAsync().join()
        time.sleep(0.5)

        # Show goal marker
        client.simFlushPersistentMarkers()
        client.simPlotPoints(
            points=[airsim.Vector3r(float(goal_x), float(goal_y), float(cruising_altitude))],
            color_rgba=[1.0, 0.0, 0.0, 1.0],
            size=40,
            duration=-1,
            is_persistent=True
        )

        result = fly_baseline(client, goal_pos, base_speed=base_speed,
                              cruising_altitude=cruising_altitude)

        results['collisions'] += result['collisions']
        results['steps'].append(result['steps'])

        if result['success']:
            results['goals_reached'] += 1
            print(f"  GOAL REACHED | Steps: {result['steps']} | "
                  f"Collisions: {result['collisions']}")
        else:
            print(f"  FAILED | Steps: {result['steps']} | "
                  f"Distance: {result['distance']:.1f}m | "
                  f"Collisions: {result['collisions']}")

    # Cleanup
    client.armDisarm(False)
    client.enableApiControl(False)

    # Summary
    print(f"\n{'=' * 70}")
    print("BASELINE SUMMARY")
    print(f"{'=' * 70}")
    print(f"Goals Reached:    {results['goals_reached']}/{episodes}")
    print(f"Total Collisions: {results['collisions']}")
    print(f"Avg Steps:        {np.mean(results['steps']):.0f}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Baseline flight — controller only, no model')
    parser.add_argument('--episodes', type=int, default=1,
                        help='Number of episodes (default: 1)')
    parser.add_argument('--distance', type=float, default=50.0,
                        help='Goal distance in meters (default: 50)')
    parser.add_argument('--speed', type=float, default=2.0,
                        help='Base flight speed in m/s (default: 2.0)')
    args = parser.parse_args()

    main(episodes=args.episodes, distance=args.distance, base_speed=args.speed)
