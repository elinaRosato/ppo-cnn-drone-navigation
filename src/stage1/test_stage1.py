"""
Test Stage 1 trained model - Forward Movement
"""

import os
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np

from airsim_env_stage1 import AirSimStage1Env


def get_latest_run_dir(base_dir="./models_stage1"):
    """Find the most recent run directory."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]

    if not run_dirs:
        return None

    run_dirs.sort()
    return os.path.join(base_dir, run_dirs[-1])


def get_model_path(model_arg=None):
    """Find the model to test."""
    if model_arg and os.path.exists(model_arg):
        return model_arg

    # Try to find the latest final model
    latest_run = get_latest_run_dir()
    if latest_run:
        final_path = os.path.join(latest_run, "stage1_forward_final.zip")
        if os.path.exists(final_path):
            return final_path

    return None


def test_model(model_path, num_episodes=5):
    """Test the trained Stage 1 model."""

    print("=" * 70)
    print("TESTING STAGE 1 MODEL - FORWARD MOVEMENT")
    print("=" * 70)

    print(f"\nLoading model from: {model_path}")
    env = AirSimStage1Env()
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")

    print(f"\nRunning {num_episodes} test episodes...")
    print("Success criteria: Drone moves forward consistently")
    print("=" * 70)

    episode_rewards = []
    episode_lengths = []
    total_forward_distances = []

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0
        start_position = None
        max_forward_distance = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 70)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

            # Track forward progress
            if 'vector' in obs:
                current_pos = obs['vector'][0][:3]  # x, y, z
                if start_position is None:
                    start_position = current_pos.copy()
                else:
                    # Calculate forward distance (primarily X direction in AirSim)
                    forward_distance = current_pos[0] - start_position[0]
                    max_forward_distance = max(max_forward_distance, forward_distance)

            if step_count % 100 == 0:
                print(f"  Step {step_count}: Reward so far = {episode_reward:.2f}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)
        total_forward_distances.append(max_forward_distance)

        print(f"  Final Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Max Forward Distance: {max_forward_distance:.2f}m")

    print("\n" + "=" * 70)
    print("TEST SUMMARY - STAGE 1")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"\nReward Statistics:")
    print(f"  Average: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"  Max: {np.max(episode_rewards):.2f}")
    print(f"  Min: {np.min(episode_rewards):.2f}")
    print(f"\nEpisode Length:")
    print(f"  Average: {np.mean(episode_lengths):.1f} steps")
    print(f"\nForward Movement:")
    print(f"  Average Distance: {np.mean(total_forward_distances):.2f}m")
    print(f"  Max Distance: {np.max(total_forward_distances):.2f}m")

    # Evaluation
    avg_reward = np.mean(episode_rewards)
    avg_forward = np.mean(total_forward_distances)

    print("\n" + "=" * 70)
    print("EVALUATION")
    print("=" * 70)

    if avg_reward > 5000 and avg_forward > 50:
        print("EXCELLENT! Drone learned forward movement very well.")
        print("Ready for Stage 2 (Goal Seeking).")
    elif avg_reward > 2000 and avg_forward > 20:
        print("GOOD! Drone learned basic forward movement.")
        print("Consider more training or proceed to Stage 2.")
    elif avg_reward > 500:
        print("FAIR. Drone shows some forward movement.")
        print("Recommend more training before Stage 2.")
    else:
        print("NEEDS WORK. Drone hasn't learned forward movement well.")
        print("Continue training Stage 1.")

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Stage 1: Forward Movement')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (auto-detects if not specified)')
    parser.add_argument('--episodes', type=int, default=3,
                        help='Number of test episodes (default: 3)')

    args = parser.parse_args()

    model_path = get_model_path(args.model)

    if model_path is None:
        print("ERROR: No model found!")
        print("Specify a model path with --model or train a model first.")
        exit(1)

    test_model(model_path, args.episodes)
