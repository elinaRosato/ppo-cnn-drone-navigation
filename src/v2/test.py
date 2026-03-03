"""
Test trained obstacle avoidance model.
Shows visual goal marker and prints per-episode stats.

Usage:
    python test.py                          # Test latest model, 5 episodes
    python test.py --episodes 10            # More episodes
    python test.py --model path/to/model.zip  # Specific model
"""

import os
import argparse
import numpy as np
from stable_baselines3 import PPO
from avoidance_env import ObstacleAvoidanceEnv


def get_latest_model():
    """Find the latest trained model."""
    base_dir = "./models_simplified"
    if not os.path.exists(base_dir):
        return None

    run_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]

    if not run_dirs:
        return None

    run_dirs.sort()
    latest_run = os.path.join(base_dir, run_dirs[-1])

    # Check for final model
    final_path = os.path.join(latest_run, "simplified_avoidance_final.zip")
    if os.path.exists(final_path):
        return final_path

    # Check for best model
    best_path = os.path.join(latest_run, "best_model", "best_model.zip")
    if os.path.exists(best_path):
        return best_path

    # Check for latest checkpoint
    checkpoint_dir = os.path.join(latest_run, "checkpoints")
    if os.path.exists(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
        if checkpoints:
            checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
            return os.path.join(checkpoint_dir, checkpoints[-1])

    return None


def test(model_path=None, episodes=5, show_marker=True):
    if model_path is None:
        model_path = get_latest_model()
        if model_path is None:
            print("No model found! Train first with: python train.py")
            return

    print(f"Loading model from: {model_path}")

    env = ObstacleAvoidanceEnv(show_visual_marker=show_marker)

    model = PPO.load(model_path)

    print(f"\nRunning {episodes} test episodes...")
    print("Watch AirSim - red dot = goal position\n")

    results = {
        'rewards': [],
        'steps': [],
        'goals_reached': 0,
        'collisions': 0,
        'timeouts': 0
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        ep_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            ep_info = info
            done = terminated or truncated

        results['rewards'].append(total_reward)
        results['steps'].append(steps)

        if ep_info.get('goal_reached'):
            results['goals_reached'] += 1
            status = "GOAL"
        elif ep_info.get('collision'):
            results['collisions'] += 1
            status = "COLLISION"
        else:
            results['timeouts'] += 1
            status = "TIMEOUT"

        print(f"  Episode {ep + 1}/{episodes}: {status} | "
              f"Reward: {total_reward:.1f} | Steps: {steps} | "
              f"Distance: {ep_info.get('distance', 0):.1f}m")

    # Summary
    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Episodes:      {episodes}")
    print(f"Goals Reached: {results['goals_reached']}/{episodes} "
          f"({100 * results['goals_reached'] / episodes:.0f}%)")
    print(f"Collisions:    {results['collisions']}/{episodes} "
          f"({100 * results['collisions'] / episodes:.0f}%)")
    print(f"Timeouts:      {results['timeouts']}/{episodes} "
          f"({100 * results['timeouts'] / episodes:.0f}%)")
    print(f"Avg Reward:    {np.mean(results['rewards']):.1f}")
    print(f"Avg Steps:     {np.mean(results['steps']):.0f}")
    print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test obstacle avoidance model')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to model file (default: latest)')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes (default: 5)')
    parser.add_argument('--no-marker', action='store_true',
                        help='Hide the visual goal marker')
    args = parser.parse_args()

    test(model_path=args.model, episodes=args.episodes, show_marker=not args.no_marker)
