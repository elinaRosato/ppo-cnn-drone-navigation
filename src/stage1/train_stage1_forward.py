"""
STAGE 1: Learn Forward Movement
Goal: Drone learns that moving forward is rewarded
Duration: 50k steps (~30 min - 1 hour)

Success Criteria:
- Episode length > 400 steps
- Forward velocity > 2.0 m/s consistently
- Mean reward > +300 per episode
"""

import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_stage1 import AirSimStage1Env


def make_env():
    env = AirSimStage1Env()
    return Monitor(env)


def get_latest_run_dir(base_dir="./models_stage1"):
    """Find the most recent run directory."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")]

    if not run_dirs:
        return None

    # Sort by timestamp in folder name
    run_dirs.sort()
    return os.path.join(base_dir, run_dirs[-1])


def train(resume=False, target_steps=None):
    base_model_dir = "./models_stage1"
    base_log_dir = "./logs_stage1"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE 1: FORWARD MOVEMENT TRAINING")
    print("=" * 70)
    print("\nObjective: Learn that moving forward is good")
    print("Reward: +1.0 per step if moving forward")
    print("Episode length: 500 steps max")
    print("\nWhat to expect:")
    print("  - First 10k steps: Random movement")
    print("  - 10k-30k steps: Starts moving forward more")
    print("  - 30k+ steps: Consistently moves forward")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nCreating environment...")
    env = make_env()
    env = DummyVecEnv([lambda: env])

    # Check if we should resume from checkpoint
    run_dir = None
    log_dir = None

    if resume:
        # Find the latest run directory
        run_dir = get_latest_run_dir(base_model_dir)

        if run_dir:
            checkpoint_dir = os.path.join(run_dir, "checkpoints")
            latest_checkpoint = None

            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoints:
                    # Sort by step number (extract number from filename like "stage1_forward_20000_steps.zip")
                    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
                    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

            if latest_checkpoint:
                # Use the same log dir as the run we're resuming
                run_name = os.path.basename(run_dir)
                log_dir = os.path.join(base_log_dir, run_name, "tensorboard")

                print(f"\n✅ Resuming from checkpoint: {latest_checkpoint}")
                model = PPO.load(latest_checkpoint, env=env, device=device)
                model.tensorboard_log = log_dir
                # Extract step count from filename and set it manually
                # Filename format: "stage1_forward_20000_steps.zip"
                checkpoint_steps = int(latest_checkpoint.split('_')[-2])
                model.num_timesteps = checkpoint_steps
                model._num_timesteps_at_start = checkpoint_steps
                print(f"   Resuming from step: {checkpoint_steps:,}")
                print(f"   Run directory: {run_dir}")
            else:
                print("\n⚠️  No checkpoint found in latest run! Starting from scratch...")
                resume = False
                run_dir = None
        else:
            print("\n⚠️  No previous runs found! Starting from scratch...")
            resume = False

    if not resume:
        # Create new timestamped run directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(base_model_dir, f"run_{timestamp}")
        log_dir = os.path.join(base_log_dir, f"run_{timestamp}", "tensorboard")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\n📁 New run directory: {run_dir}")
        print("\nCreating new PPO model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=8192,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.2,  # Higher exploration for this simple task
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                )
            ),
            verbose=1,
            tensorboard_log=log_dir,
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="stage1_forward"
    )

    total_timesteps = target_steps if target_steps else 50_000
    current_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    remaining_timesteps = max(0, total_timesteps - current_steps)

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 1 TRAINING")
    print(f"{'=' * 70}")
    print(f"Target timesteps: {total_timesteps:,}")
    if current_steps > 0:
        print(f"Current progress: {current_steps:,}")
        print(f"Remaining steps: {remaining_timesteps:,}")
    print(f"{'=' * 70}\n")

    if remaining_timesteps <= 0:
        print("✅ Already reached target timesteps! Nothing to train.")
        env.close()
        return

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from checkpoint step count when resuming
            tb_log_name="stage1"  # Fixed name for continuous logging
        )

        final_model_path = os.path.join(run_dir, "stage1_forward_final")
        model.save(final_model_path)
        print(f"\n✅ Stage 1 model saved to: {final_model_path}")
        print(f"\n📊 Next: Run train_stage2_goals.py to add goal seeking")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save(os.path.join(run_dir, "stage1_forward_interrupted"))

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 1: Forward Movement')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--steps', type=int, default=None,
                        help='Target total timesteps (default: 50000)')
    args = parser.parse_args()

    train(resume=args.resume, target_steps=args.steps)
