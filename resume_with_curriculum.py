"""
Resume training from checkpoint but START curriculum from beginning.

This is useful when:
- You have a partially trained model
- You want to add curriculum learning
- But start the curriculum from easy (not at checkpoint's progress level)

The model keeps its learned weights, but curriculum resets to 0.
"""

import os
import sys
import glob
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_v3 import AirSimDroneEnv


def make_env(randomize_goals=True, randomize_bounds=True):
    env = AirSimDroneEnv(
        randomize_goals=randomize_goals,
        randomize_height_bounds=randomize_bounds,
        goal_range_x=(15, 30),  # Will be adjusted by curriculum
        goal_range_y=(15, 30),  # Will be adjusted by curriculum
        height_bound_ranges={
            'max_height': (-1.5, -1.7),
            'min_height': (-2.5, -2.7)
        },
        default_max_height=-1.6,
        default_min_height=-2.6,
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=2.0,
        # Curriculum learning enabled
        curriculum_learning=True,
        curriculum_start_distance=10.0,  # Start easy: 10m goals
        curriculum_end_distance=30.0,    # End hard: 30m goals
        curriculum_timesteps=200000      # Reach full difficulty at 200k steps
    )
    return Monitor(env)


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = glob.glob(os.path.join(checkpoint_dir, "ppo_*.zip"))
    if not checkpoints:
        return None

    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def resume_with_curriculum(checkpoint_path=None, additional_timesteps=None):
    """Resume training from checkpoint but reset curriculum to start"""

    print("=" * 70)
    print("RESUME WITH CURRICULUM (RESET)")
    print("=" * 70)

    if checkpoint_path is None:
        print("\nSearching for latest checkpoint...")
        checkpoint_path = find_latest_checkpoint("./models_v3_quality/checkpoints/")

        if checkpoint_path is None:
            print("No checkpoints found!")
            print("Please specify a checkpoint path.")
            sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading checkpoint: {checkpoint_path}")

    try:
        filename = os.path.basename(checkpoint_path)
        parts = filename.split('_')
        for i, part in enumerate(parts):
            if part.isdigit():
                completed_steps = int(part)
                break
        else:
            completed_steps = 0
        print(f"Checkpoint trained for: {completed_steps:,} steps")
    except:
        completed_steps = 0
        print("Could not determine completed steps from filename")

    print("\nCreating environment with curriculum...")
    env = make_env(randomize_goals=True, randomize_bounds=True)
    env = DummyVecEnv([lambda: env])

    # IMPORTANT: Reset curriculum counter to 0
    # This makes curriculum start from easy even though model has training
    # Access the actual AirSimDroneEnv through Monitor wrapper
    env.envs[0].env.total_timesteps = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading model...")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print("Model loaded successfully!")

    print("\n" + "=" * 70)
    print("IMPORTANT: CURRICULUM RESET")
    print("=" * 70)
    print(f"  Model has: {completed_steps:,} steps of training")
    print(f"  Curriculum starts at: 0 steps (EASY - 10m goals)")
    print("\nWhy?")
    print("  - Old training had bad rewards/bounds")
    print("  - Model learned bad habits")
    print("  - Curriculum helps retrain from easy tasks")
    print("  - Model keeps neural network weights (not starting from random)")
    print("=" * 70)

    if additional_timesteps is None:
        additional_timesteps = 500_000  # Train for full curriculum

    print(f"\nTraining Plan:")
    print(f"  Model already has: {completed_steps:,} steps")
    print(f"  Additional training: {additional_timesteps:,} steps")
    print(f"  Total after: {completed_steps + additional_timesteps:,} steps")
    print("\nCurriculum Schedule (from step 0):")
    print("  Step 0-50k:   Goals 10-15m away (EASY)")
    print("  Step 50-100k: Goals 15-20m away")
    print("  Step 100-150k: Goals 20-25m away (MEDIUM)")
    print("  Step 150-200k: Goals 25-30m away")
    print("  Step 200k+:   Goals 30m away (HARD)")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_v3_curriculum/checkpoints/",
        name_prefix="ppo_curriculum"
    )

    eval_env = make_env(randomize_goals=True, randomize_bounds=True)
    eval_env = DummyVecEnv([lambda: eval_env])
    # Reset curriculum for eval environment too
    eval_env.envs[0].env.total_timesteps = 0

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_v3_curriculum/best_model/",
        log_path="./logs_v3_curriculum/eval/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print(f"\n{'=' * 70}")
    print("RESUMING WITH CURRICULUM")
    print(f"{'=' * 70}")
    print("\nIMPORTANT:")
    print("  1. Make sure AirSim (Blocks.exe) is running")
    print("  2. Verify settings.json has ClockSpeed=5.0")
    print("  3. Watch curriculum progress in episode logs")
    print("\nMonitoring:")
    print("  - Checkpoints: every 10,000 steps -> ./models_v3_curriculum/checkpoints/")
    print("  - Evaluation: every 5,000 steps -> ./models_v3_curriculum/best_model/")
    print("  - TensorBoard: tensorboard --logdir=./logs_v3_curriculum/tensorboard/")
    print(f"{'=' * 70}\n")

    input("Press ENTER to start curriculum training...")

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False  # Keep counting total steps
        )

        final_model_path = "./models_v3_curriculum/ppo_curriculum_resumed"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        interrupted_model_path = "./models_v3_curriculum/ppo_curriculum_interrupted"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()

    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume with curriculum (reset)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: auto-detect latest)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=500000,
        help="Additional timesteps to train (default: 500k for full curriculum)"
    )

    args = parser.parse_args()

    resume_with_curriculum(
        checkpoint_path=args.checkpoint,
        additional_timesteps=args.steps
    )
