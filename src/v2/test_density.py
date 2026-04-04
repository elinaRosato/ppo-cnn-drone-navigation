"""
Test forest density configurations by placing trees in AirSim.

Connects to AirSim, places trees at the chosen density, and lets you
re-place with a new random layout on each Enter press. Useful for
visually inspecting what each density stage looks like before training.

Usage:
    python test_density.py --density sparse        # 15 m min spacing
    python test_density.py --density medium        # 10 m min spacing
    python test_density.py --density dense         # 6 m min spacing
    python test_density.py --density dense --goal-distance 80
    python test_density.py --min-dist 8            # custom spacing
"""

import os
import math
import random
import argparse
import time
import cosysairsim as airsim

TREE_ACTOR_FILTER = 'StaticMeshActor_UAID_'

DENSITY_PRESETS = {
    'sparse': 15.0,
    'medium': 10.0,
    'dense':   6.0,
}


def place_trees(client, tree_names, ground_z, goal_x, goal_y, min_dist):
    """Place trees in a density-scaled ellipse around the drone-to-goal path."""
    n_trees = len(tree_names)
    DRONE_CLEARANCE = 8.0
    GOAL_CLEARANCE  = 8.0

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

    print(f"  Placed {len(placed)}/{n_trees} trees | "
          f"min spacing {min_dist} m | ellipse {a:.0f}x{b:.0f} m | "
          f"{attempts} attempts")

    ok = fail = 0
    for i, name in enumerate(tree_names):
        if i < len(placed):
            x, y = placed[i]
        else:
            x, y = placed[-1] if placed else (cx, cy)
        pose = airsim.Pose(
            airsim.Vector3r(x, y, ground_z),
            airsim.Quaternionr(0, 0, 0, 1)
        )
        if client.simSetObjectPose(name, pose, teleport=True):
            ok += 1
        else:
            fail += 1

    if fail:
        print(f"  WARNING: {fail} trees failed to move (first: {tree_names[ok]})")
    else:
        print(f"  All {ok} trees moved successfully.")


def main():
    parser = argparse.ArgumentParser(description='Test forest density placement in AirSim')
    density_group = parser.add_mutually_exclusive_group(required=True)
    density_group.add_argument('--density', choices=['sparse', 'medium', 'dense'],
                               help='Named density preset')
    density_group.add_argument('--min-dist', type=float,
                               help='Custom minimum spacing between trees (metres)')
    parser.add_argument('--goal-distance', type=float, default=50.0,
                        help='Distance to place the goal from origin (default: 50 m)')
    args = parser.parse_args()

    min_dist = DENSITY_PRESETS[args.density] if args.density else args.min_dist
    label = args.density if args.density else f'custom ({min_dist} m)'

    airsim_host = os.environ.get('AIRSIM_HOST', '')
    client = airsim.MultirotorClient(ip=airsim_host) if airsim_host else airsim.MultirotorClient()
    client.confirmConnection()
    print(f"Connected to AirSim at {airsim_host or 'localhost'}")
    time.sleep(1.0)

    all_objects = client.simListSceneObjects()
    tree_names = [o for o in all_objects if TREE_ACTOR_FILTER in o]
    print(f"Found {len(tree_names)} tree actors.")
    if not tree_names:
        print("ERROR: no trees found. Check TREE_ACTOR_FILTER in avoidance_env.py.")
        return

    ground_z = 0.0
    sample_pose = client.simGetObjectPose(tree_names[0])
    ground_z = sample_pose.position.z_val
    print(f"Ground z: {ground_z:.2f}")

    print(f"\nDensity: {label} | Goal distance: {args.goal_distance} m")
    print("Press Enter to place a new random layout, Ctrl+C to quit.\n")

    try:
        while True:
            angle = random.uniform(0, 2 * math.pi)
            goal_x = args.goal_distance * math.cos(angle)
            goal_y = args.goal_distance * math.sin(angle)
            print(f"Goal: ({goal_x:.1f}, {goal_y:.1f})")

            client.simFlushPersistentMarkers()
            client.simPlotPoints(
                points=[airsim.Vector3r(float(goal_x), float(goal_y), -1.3)],
                color_rgba=[1.0, 0.0, 0.0, 1.0],
                size=40,
                duration=-1,
                is_persistent=True
            )

            place_trees(client, tree_names, ground_z, goal_x, goal_y, min_dist)

            input("\n[Enter] new layout  [Ctrl+C] quit\n")
    except KeyboardInterrupt:
        client.simFlushPersistentMarkers()
        print("\nDone.")


if __name__ == "__main__":
    main()
