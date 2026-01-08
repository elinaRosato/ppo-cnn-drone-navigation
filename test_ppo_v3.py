"""
Test trained PPO V3 model for AirSim drone navigation
"""

import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np

from airsim_env_v3 import AirSimDroneEnv


def make_env(randomize_goals=False, randomize_bounds=False, goal_pos=None):
    """Create test environment"""
    if goal_pos is None:
        goal_pos = (20, 20, -2.5)

    env = AirSimDroneEnv(
        randomize_goals=randomize_goals,
        randomize_height_bounds=randomize_bounds,
        goal_range_x=(15, 30),
        goal_range_y=(15, 30),
        height_bound_ranges={
            'max_height': (-1.0, -1.8),
            'min_height': (-2.5, -4.0)
        },
        default_max_height=-1.5,
        default_min_height=-3.0,
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=2.0  # 2m radius around goal
    )
    return Monitor(env)


def test_model(model_path, num_episodes=5, randomize=True):
    """Test the trained model"""

    print("=" * 70)
    print("TESTING PPO V3 MODEL")
    print("=" * 70)

    print(f"\nLoading model from: {model_path}")
    env = make_env(randomize_goals=randomize, randomize_bounds=randomize)
    env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")

    print(f"\nRunning {num_episodes} test episodes...")
    if randomize:
        print("Mode: Randomized goals and height bounds")
    else:
        print("Mode: Fixed goal and height bounds")
    print("=" * 70)

    episode_rewards = []
    episode_lengths = []
    success_count = 0
    collision_count = 0

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        step_count = 0

        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("-" * 70)

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            step_count += 1

            if done:
                distance = info[0].get('distance_to_goal', float('inf'))
                collision = info[0].get('collision', False)

                if distance < 1.0:
                    success_count += 1
                    print(f"  SUCCESS: Goal reached in {step_count} steps")
                elif collision:
                    collision_count += 1
                    print(f"  COLLISION at step {step_count}")
                else:
                    print(f"  Episode ended at step {step_count}")

        episode_rewards.append(episode_reward)
        episode_lengths.append(step_count)

        print(f"  Final Distance: {info[0].get('distance_to_goal', 0):.2f}m")
        print(f"  Total Reward: {episode_reward:.2f}")
        print(f"  Steps: {step_count}")
        print(f"  Final Height: {info[0].get('height', 0):.2f}m")

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Episodes: {num_episodes}")
    print(f"Success Rate: {success_count}/{num_episodes} ({100*success_count/num_episodes:.1f}%)")
    print(f"Collision Rate: {collision_count}/{num_episodes} ({100*collision_count/num_episodes:.1f}%)")
    print(f"\nAverage Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Average Episode Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test trained PPO V3 model')
    parser.add_argument('--model', type=str,
                        default='./models_v3_quality/best_model/best_model.zip',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=5,
                        help='Number of test episodes (default: 5)')
    parser.add_argument('--fixed', action='store_true',
                        help='Use fixed goal instead of random goals')

    args = parser.parse_args()

    test_model(args.model, args.episodes, randomize=not args.fixed)
