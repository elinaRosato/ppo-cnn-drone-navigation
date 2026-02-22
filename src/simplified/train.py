"""
Train obstacle avoidance model.

The model learns to output lateral/vertical corrections
based on camera input to avoid obstacles.
Navigation is handled by a simple controller.

Usage:
    python train.py                     # New training, 200k steps
    python train.py --steps 500000      # New training, 500k steps
    python train.py --resume            # Resume from latest checkpoint
    python train.py --resume --steps 400000  # Resume, train to 400k total
"""

import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import numpy as np

from avoidance_env import ObstacleAvoidanceEnv


def make_env():
    def _init():
        env = ObstacleAvoidanceEnv()
        return Monitor(env)
    return _init


def get_latest_run_dir(base_dir):
    """Find the most recent run directory."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]

    if not run_dirs:
        return None

    run_dirs.sort()
    return os.path.join(base_dir, run_dirs[-1])


def train(resume=False, target_steps=None):
    base_model_dir = "./models_simplified"
    base_log_dir = "./logs_simplified"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)

    print("=" * 70)
    print("OBSTACLE AVOIDANCE TRAINING")
    print("=" * 70)
    print("\nModel input:  Depth image (84x84)")
    print("Model output: Lateral + vertical correction")
    print("Controller:   Flies toward goal automatically")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nCreating environment...")
    env = DummyVecEnv([make_env()])

    run_dir = None
    log_dir = None

    if resume:
        run_dir = get_latest_run_dir(base_model_dir)

        if run_dir:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            latest_checkpoint = None

            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
                    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

            if latest_checkpoint:
                run_name = os.path.basename(run_dir)
                log_dir = os.path.join(base_log_dir, run_name, "tensorboard")

                print(f"\nResuming from checkpoint: {latest_checkpoint}")
                model = PPO.load(latest_checkpoint, env=env, device=device)
                model.ent_coef = 0.01
                model.tensorboard_log = log_dir
                checkpoint_steps = int(latest_checkpoint.split('_')[-2])
                model.num_timesteps = checkpoint_steps
                model._num_timesteps_at_start = checkpoint_steps
                print(f"   Resuming from step: {checkpoint_steps:,}")
                print(f"   Run directory: {run_dir}")
            else:
                print("\nNo checkpoint found in latest run! Starting from scratch...")
                resume = False
                run_dir = None
        else:
            print("\nNo previous runs found! Starting from scratch...")
            resume = False

    if not resume:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(base_model_dir, f"run_{timestamp}")
        log_dir = os.path.join(base_log_dir, f"run_{timestamp}", "tensorboard")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nNew run directory: {run_dir}")

        print("\nCreating new model (CnnPolicy)")
        model = PPO(
            "CnnPolicy",
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="simplified_avoidance"
    )

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecTransposeImage(eval_env)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=2000,
        n_eval_episodes=3,
        deterministic=True,
        render=False
    )

    total_timesteps = target_steps if target_steps else 200_000
    current_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    remaining_timesteps = max(0, total_timesteps - current_steps)

    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print(f"{'=' * 70}")
    print(f"Target timesteps: {total_timesteps:,}")
    if current_steps > 0:
        print(f"Current progress: {current_steps:,}")
        print(f"Remaining steps: {remaining_timesteps:,}")
    print(f"{'=' * 70}\n")

    if remaining_timesteps <= 0:
        print("Already reached target timesteps! Nothing to train.")
        env.close()
        eval_env.close()
        return

    print("CHECKLIST:")
    print("  [ ] AirSim is running with your environment")
    print("  [ ] GPU/CUDA available (recommended)")
    print("")

    input("Press ENTER when ready...")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="avoidance"
        )

        final_model_path = os.path.join(run_dir, "simplified_avoidance_final")
        model.save(final_model_path)
        print(f"\nModel saved to: {final_model_path}")
        print(f"\nTo test: python test.py")
        print(f"To fly a mission: python fly_mission.py --model {final_model_path}.zip")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save(os.path.join(run_dir, "simplified_avoidance_interrupted"))

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train obstacle avoidance model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--steps', type=int, default=None,
                        help='Target total timesteps (default: 200000)')
    args = parser.parse_args()

    train(resume=args.resume, target_steps=args.steps)
