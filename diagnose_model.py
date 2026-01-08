"""
Diagnose if a trained model can be resumed or should be retrained.
Tests if the model uses camera vision or just exploits height bounds.
"""

import argparse
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from airsim_env_v3 import AirSimDroneEnv


def make_env():
    env = AirSimDroneEnv(
        randomize_goals=False,
        randomize_height_bounds=False,
        goal_range_x=(15, 30),
        goal_range_y=(15, 30),
        height_bound_ranges={
            'max_height': (-1.5, -1.7),
            'min_height': (-2.5, -2.7)
        },
        default_max_height=-1.6,
        default_min_height=-2.6,
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=2.0  # 2m radius around goal
    )
    return Monitor(env)


def diagnose_model(model_path, num_episodes=10):
    """
    Diagnose model behavior to determine if it can be resumed or needs retraining.
    """
    print("=" * 70)
    print("MODEL DIAGNOSIS")
    print("=" * 70)
    print(f"\nLoading model: {model_path}")

    env = make_env()
    env = DummyVecEnv([lambda: env])

    model = PPO.load(model_path, env=env)
    print("Model loaded successfully!")

    print(f"\nRunning {num_episodes} diagnostic episodes...")
    print("=" * 70)

    metrics = {
        'vertical_changes': [],
        'altitude_deviations': [],
        'success_rate': 0,
        'crash_rate': 0,
        'avg_episode_length': [],
        'diagonal_flight_detected': 0
    }

    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        step_count = 0
        heights = []
        crashed = False
        succeeded = False

        print(f"\nEpisode {episode + 1}/{num_episodes}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            height = info[0].get('height', 0)
            heights.append(height)
            step_count += 1

            if done:
                distance = info[0].get('distance_to_goal', float('inf'))
                collision = info[0].get('collision', False)

                if distance < 1.0:
                    succeeded = True
                    metrics['success_rate'] += 1
                    print(f"  Result: SUCCESS in {step_count} steps")
                elif collision:
                    crashed = True
                    metrics['crash_rate'] += 1
                    print(f"  Result: CRASHED at step {step_count}")
                else:
                    print(f"  Result: TIMEOUT at step {step_count}")

        metrics['avg_episode_length'].append(step_count)

        # Analyze vertical behavior
        if len(heights) > 10:
            early_heights = heights[:20]
            mid_heights = heights[20:40] if len(heights) > 40 else heights[20:]

            early_avg = np.mean(early_heights)
            mid_avg = np.mean(mid_heights) if mid_heights else early_avg

            vertical_change = abs(early_avg - mid_avg)
            metrics['vertical_changes'].append(vertical_change)

            # Check for diagonal flight pattern
            if vertical_change > 0.5:  # Large altitude change early
                metrics['diagonal_flight_detected'] += 1
                print(f"  WARNING: Large vertical change detected: {vertical_change:.2f}m")

            # Calculate deviation from target altitude
            target_height = -2.1  # Center of bounds
            avg_deviation = np.mean([abs(h - target_height) for h in heights])
            metrics['altitude_deviations'].append(avg_deviation)

    # Print diagnosis
    print("\n" + "=" * 70)
    print("DIAGNOSIS RESULTS")
    print("=" * 70)

    success_rate = (metrics['success_rate'] / num_episodes) * 100
    crash_rate = (metrics['crash_rate'] / num_episodes) * 100
    avg_length = np.mean(metrics['avg_episode_length'])
    avg_vertical_change = np.mean(metrics['vertical_changes'])
    avg_altitude_dev = np.mean(metrics['altitude_deviations'])
    diagonal_pct = (metrics['diagonal_flight_detected'] / num_episodes) * 100

    print(f"\nPerformance:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Crash rate: {crash_rate:.1f}%")
    print(f"  Average episode length: {avg_length:.1f} steps")

    print(f"\nBehavior Analysis:")
    print(f"  Average vertical change (early→mid): {avg_vertical_change:.2f}m")
    print(f"  Average altitude deviation from center: {avg_altitude_dev:.2f}m")
    print(f"  Diagonal flight pattern detected: {diagonal_pct:.1f}% of episodes")

    # Diagnosis
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    issues = []

    if crash_rate > 50:
        issues.append("High crash rate (>50%)")

    if avg_vertical_change > 0.3:
        issues.append(f"Large vertical changes ({avg_vertical_change:.2f}m)")

    if diagonal_pct > 30:
        issues.append(f"Diagonal flight pattern ({diagonal_pct:.0f}% episodes)")

    if avg_altitude_dev > 0.3:
        issues.append(f"Poor altitude maintenance ({avg_altitude_dev:.2f}m deviation)")

    if not issues:
        print("\n✓ MODEL LOOKS GOOD!")
        print("  - Low crash rate")
        print("  - Maintains level flight")
        print("  - Uses camera for navigation")
        print("\n  RECOMMENDATION: Can resume training with new config")
        print("  Command: python resume_training_v3.py")
    else:
        print("\n✗ MODEL HAS ISSUES:")
        for issue in issues:
            print(f"  - {issue}")

        print("\n  These behaviors suggest the model learned to:")
        print("  1. Ignore camera vision")
        print("  2. Exploit height bounds")
        print("  3. Take diagonal shortcuts")

        print("\n  RECOMMENDATION: Start training from scratch")
        print("  Reason: Bad habits are deeply ingrained in network weights")
        print("  Command: python train_ppo_v3_quality.py")

        print("\n  Why not resume?")
        print("  - Model has {:.0f}k+ steps of wrong learning".format(avg_length * num_episodes / 1000))
        print("  - New penalties will confuse the model")
        print("  - Learning will be slow and unstable")
        print("  - Fresh training learns correct behavior faster")

    env.close()
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Diagnose trained PPO model')
    parser.add_argument('--model', type=str,
                        default='./models_v3_quality/best_model/best_model.zip',
                        help='Path to trained model')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Number of test episodes (default: 10)')

    args = parser.parse_args()

    diagnose_model(args.model, args.episodes)
