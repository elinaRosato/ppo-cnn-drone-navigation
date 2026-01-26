"""
STAGE 3: Add Altitude Constraints
Resume from Stage 2, add min/max altitude bounds
Duration: 250k steps (~3-5 hours)

Success Criteria:
- Reaches goal 40%+ of episodes while respecting altitude
- Mean reward > +600 per episode
- Altitude violations < 10% of steps
"""

import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_stage3 import AirSimStage3Env


def make_env():
    env = AirSimStage3Env(
        goal_range_x=(5, 20),
        goal_range_y=(5, 20),
        goal_range_z=(-20, -10),  # Safe altitude range (10-20m above ground)
        goal_radius=3.0,
        min_altitude=-25.0,  # Floor at 25m
        max_altitude=-10.0   # Ceiling at 10m (above Blocks obstacles)
    )
    return Monitor(env)


def get_latest_run_dir(base_dir):
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


def get_stage2_model_path():
    """Find the final model from the latest stage2 run."""
    latest_run = get_latest_run_dir("./models_stage2")
    if latest_run:
        final_path = os.path.join(latest_run, "stage2_goals_final.zip")
        if os.path.exists(final_path):
            return final_path
    # Fallback to old path structure
    old_path = "./models_stage2/stage2_goals_final.zip"
    if os.path.exists(old_path):
        return old_path
    return None


def train(resume=False, target_steps=None):
    base_model_dir = "./models_stage3"
    base_log_dir = "./logs_stage3"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)

    print("=" * 70)
    print("STAGE 3: ALTITUDE CONSTRAINTS TRAINING")
    print("=" * 70)
    print("\nObjective: Navigate to goals while respecting altitude bounds")
    print("Resumes from: Stage 2 goal seeking model")
    print("Adds: Min/max altitude constraints, varying goal heights")
    print("\nWhat to expect:")
    print("  - First 50k steps: Learns altitude bounds exist")
    print("  - 50k-150k steps: Navigates while respecting bounds")
    print("  - 150k+ steps: Reaches goals at various altitudes 40%+ of time")
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
                    # Sort by step number (extract number from filename like "stage3_altitude_25000_steps.zip")
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

        # Try to load Stage 2 model
        stage2_path = get_stage2_model_path()
        if stage2_path:
            print(f"\n✅ Loading Stage 2 model from: {stage2_path}")
            model = PPO.load(stage2_path, env=env, device=device)
            model.tensorboard_log = log_dir
            # Reset timesteps - this is a NEW stage, not a resume
            model.num_timesteps = 0
            model._num_timesteps_at_start = 0
            print("✅ Successfully loaded Stage 2 model - will continue learning")
        else:
            print("\n⚠️  Stage 2 model not found!")
            print("⚠️  Creating new model - you should train Stage 2 first!")
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
                ent_coef=0.1,  # Moderate exploration
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
        save_freq=25000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="stage3_altitude"
    )

    eval_env = make_env()
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = target_steps if target_steps else 250_000
    current_steps = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    remaining_timesteps = max(0, total_timesteps - current_steps)

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 3 TRAINING")
    print(f"{'=' * 70}")
    print(f"Target timesteps: {total_timesteps:,}")
    if current_steps > 0:
        print(f"Current progress: {current_steps:,}")
        print(f"Remaining steps: {remaining_timesteps:,}")
    print(f"{'=' * 70}\n")

    if remaining_timesteps <= 0:
        print("✅ Already reached target timesteps! Nothing to train.")
        env.close()
        eval_env.close()
        return

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from checkpoint step count when resuming
            tb_log_name="stage3"  # Fixed name for continuous logging
        )

        final_model_path = os.path.join(run_dir, "stage3_altitude_final")
        model.save(final_model_path)
        print(f"\n✅ Stage 3 model saved to: {final_model_path}")
        print(f"\n📊 Next: Run train_stage4_obstacles.py to add obstacle avoidance")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save(os.path.join(run_dir, "stage3_altitude_interrupted"))

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Stage 3: Altitude Constraints')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--steps', type=int, default=None,
                        help='Target total timesteps (default: 250000)')
    args = parser.parse_args()

    train(resume=args.resume, target_steps=args.steps)
