"""
Monitor drone and goal during training.
Shows a green marker above the drone and a red sphere at the goal.

Usage:
    python monitor_drones.py

Press Ctrl+C to stop.
"""

import time
import math
import airsim


def sphere_points(cx, cy, cz, radius, n_points=80):
    """Generate points on a sphere surface using Fibonacci distribution."""
    points = []
    golden_ratio = (1 + math.sqrt(5)) / 2
    for i in range(n_points):
        theta = 2 * math.pi * i / golden_ratio
        phi = math.acos(1 - 2 * (i + 0.5) / n_points)
        x = cx + radius * math.sin(phi) * math.cos(theta)
        y = cy + radius * math.sin(phi) * math.sin(theta)
        z = cz + radius * math.cos(phi)
        points.append(airsim.Vector3r(x, y, z))
    return points


def read_goal():
    """Read goal position from file written by the environment."""
    try:
        with open("goal_position.txt", "r") as f:
            parts = f.read().strip().split()
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        return None


def main():
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
    print("Green dot = drone, Red sphere = goal")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            client.simFlushPersistentMarkers()

            # Drone marker (green dot above drone)
            try:
                state = client.getMultirotorState()
                pos = state.kinematics_estimated.position
                marker = airsim.Vector3r(pos.x_val, pos.y_val, pos.z_val - 15.0)
                client.simPlotPoints(
                    points=[marker],
                    color_rgba=[0.0, 1.0, 0.0, 1.0],
                    size=50.0,
                    duration=-1,
                    is_persistent=True
                )
            except Exception:
                pass

            # Goal sphere (red, radius matches goal_radius)
            goal = read_goal()
            if goal:
                gx, gy, gz = goal
                points = sphere_points(gx, gy, gz, radius=1.5, n_points=80)
                client.simPlotPoints(
                    points=points,
                    color_rgba=[1.0, 0.0, 0.0, 1.0],
                    size=10.0,
                    duration=-1,
                    is_persistent=True
                )

            time.sleep(1.0)

    except KeyboardInterrupt:
        client.simFlushPersistentMarkers()
        print("\nStopped monitoring.")


if __name__ == "__main__":
    main()
