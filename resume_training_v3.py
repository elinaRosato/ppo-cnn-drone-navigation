"""
Resume PPO training from a saved checkpoint.
Automatically continues curriculum from where checkpoint left off.

Usage:
    # Auto-detect latest checkpoint from baby_steps, quality, or curriculum
    python resume_training_v3.py

    # Specify checkpoint
    python resume_training_v3.py --checkpoint ./models_v3_baby_steps/checkpoints/ppo_baby_steps_50000_steps.zip

    # Specify additional training steps
    python resume_training_v3.py --steps 100000
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
        goal_range_x=(15, 30),  # Will be overridden by curriculum
        goal_range_y=(15, 30),  # Will be overridden by curriculum
        height_bound_ranges={
            'max_height': (-1.0, -1.2),  # Ceiling
            'min_height': (-3.5, -3.7)   # Floor (INCREASED for larger drones)
        },
        default_max_height=-1.1,
        default_min_height=-3.6,  # 2.5m range
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=3.0,  # Updated to 3m
        # Curriculum learning enabled
        curriculum_learning=True,
        curriculum_start_distance=3.0,   # Baby steps start
        curriculum_end_distance=35.0,
        curriculum_timesteps=300000
    )
    return Monitor(env)


def find_latest_checkpoint(checkpoint_dir):
    """Find the most recent checkpoint in the directory"""
    if not os.path.exists(checkpoint_dir):
        return None

    # Look for baby_steps checkpoints (recommended)
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "ppo_baby_steps_*.zip"))

    # Fallback to other checkpoint patterns
    if not checkpoints:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "ppo_*.zip"))

    if not checkpoints:
        return None

    latest = max(checkpoints, key=os.path.getmtime)
    return latest


def resume_training(checkpoint_path=None, additional_timesteps=None):
    """Resume training from a checkpoint"""

    print("=" * 70)
    print("RESUME TRAINING - PPO V3")
    print("=" * 70)

    if checkpoint_path is None:
        print("\nSearching for latest checkpoint...")
        # Try baby_steps directory first (recommended)
        checkpoint_path = find_latest_checkpoint("./models_v3_baby_steps/checkpoints/")

        # Fallback to other directories
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint("./models_v3_quality/checkpoints/")

        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint("./models_v3_curriculum/checkpoints/")

        if checkpoint_path is None:
            print("No checkpoints found!")
            print("Searched in:")
            print("  - ./models_v3_baby_steps/checkpoints/")
            print("  - ./models_v3_quality/checkpoints/")
            print("  - ./models_v3_curriculum/checkpoints/")
            print("\nPlease specify a checkpoint path or train from scratch first.")
            sys.exit(1)

    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    print(f"\nLoading checkpoint: {checkpoint_path}")

    try:
        filename = os.path.basename(checkpoint_path)
        completed_steps = int(filename.split('_')[-2])
        print(f"Checkpoint trained for: {completed_steps:,} steps")
    except:
        completed_steps = 0
        print("Could not determine completed steps from filename")

    print("\nCreating environment...")
    env = make_env(randomize_goals=True, randomize_bounds=True)
    env = DummyVecEnv([lambda: env])

    # IMPORTANT: Set curriculum progress to match checkpoint
    # This ensures curriculum continues from where it left off
    env.envs[0].env.total_timesteps = completed_steps

    # Calculate current curriculum stage
    curriculum_timesteps = 300000
    curriculum_start_dist = 3.0
    curriculum_end_dist = 35.0
    progress_pct = min(100, (completed_steps / curriculum_timesteps) * 100)
    current_dist = curriculum_start_dist + (progress_pct / 100.0) * (curriculum_end_dist - curriculum_start_dist)

    print(f"Curriculum progress set to: {completed_steps:,} steps ({progress_pct:.1f}%)")
    print(f"Current goal distance: {current_dist:.1f}m")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("\nLoading model...")
    model = PPO.load(checkpoint_path, env=env, device=device)
    print("Model loaded successfully!")

    if additional_timesteps is None:
        target_total = 500_000
        additional_timesteps = max(0, target_total - completed_steps)

    if additional_timesteps <= 0:
        print(f"\nModel already trained for {completed_steps:,} steps")
        print("No additional training needed (target: 500k)")
        env.close()
        return

    print(f"\nTraining Plan:")
    print(f"  Already completed: {completed_steps:,} steps")
    print(f"  Additional training: {additional_timesteps:,} steps")
    print(f"  Total after resume: {completed_steps + additional_timesteps:,} steps")

    # Determine save directory based on checkpoint origin
    if "baby_steps" in checkpoint_path:
        save_dir = "./models_v3_baby_steps"
        log_dir = "./logs_v3_baby_steps"
        prefix = "ppo_baby_steps"
    elif "curriculum" in checkpoint_path:
        save_dir = "./models_v3_curriculum"
        log_dir = "./logs_v3_curriculum"
        prefix = "ppo_curriculum"
    else:
        save_dir = "./models_v3_quality"
        log_dir = "./logs_v3_quality"
        prefix = "ppo_quality"

    print(f"  Saving to: {save_dir}/")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=f"{save_dir}/checkpoints/",
        name_prefix=prefix
    )

    eval_env = make_env(randomize_goals=True, randomize_bounds=True)
    eval_env = DummyVecEnv([lambda: eval_env])

    # IMPORTANT: Set eval environment curriculum to match training progress
    eval_env.envs[0].env.total_timesteps = completed_steps

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model/",
        log_path=f"{log_dir}/eval/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print(f"\n{'=' * 70}")
    print("RESUMING TRAINING")
    print(f"{'=' * 70}")
    print("\nIMPORTANT:")
    print("  1. Make sure AirSim (Blocks.exe) is running")
    print("  2. Verify settings.json has ClockSpeed=5.0")
    print("\nMonitoring:")
    print("  - Checkpoints: every 10,000 steps -> ./models_v3_quality/checkpoints/")
    print("  - Evaluation: every 5,000 steps -> ./models_v3_quality/best_model/")
    print("  - TensorBoard: tensorboard --logdir=./logs_v3_quality/tensorboard/")
    print(f"{'=' * 70}\n")

    input("Press ENTER to resume training...")

    try:
        model.learn(
            total_timesteps=additional_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )

        final_model_path = "./models_v3_quality/ppo_quality_final"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        interrupted_model_path = "./models_v3_quality/ppo_quality_interrupted"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()

    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Resume PPO v3 training from checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint (default: auto-detect latest)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Additional timesteps to train (default: auto to reach 500k)"
    )

    args = parser.parse_args()

    resume_training(
        checkpoint_path=args.checkpoint,
        additional_timesteps=args.steps
    )
