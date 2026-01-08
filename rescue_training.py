"""
Rescue your current training by restarting curriculum with ULTRA-EASY goals.

Takes your partially trained model and restarts with 3m goals, giving it a chance
to learn the basics before tackling harder navigation.
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


def make_env():
    env = AirSimDroneEnv(
        randomize_goals=True,
        randomize_height_bounds=True,
        goal_range_x=(15, 30),
        goal_range_y=(15, 30),
        height_bound_ranges={
            'max_height': (-1.0, -1.2),  # Ceiling
            'min_height': (-3.5, -3.7)   # Floor (INCREASED for larger drones)
        },
        default_max_height=-1.1,
        default_min_height=-3.6,  # 2.5m range for wiggle room
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=3.0,  # Easier
        # RESCUE curriculum: ultra-easy
        curriculum_learning=True,
        curriculum_start_distance=3.0,   # Start VERY easy
        curriculum_end_distance=35.0,
        curriculum_timesteps=300000
    )
    return Monitor(env)


def rescue_training():
    print("=" * 70)
    print("RESCUE TRAINING - RESTART WITH ULTRA-EASY CURRICULUM")
    print("=" * 70)

    # Find latest checkpoint
    checkpoint_dir = "./models_v3_curriculum/checkpoints/"
    if os.path.exists(checkpoint_dir):
        checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.zip"))
        if checkpoints:
            checkpoint_path = max(checkpoints, key=os.path.getmtime)
            print(f"\nFound checkpoint: {checkpoint_path}")
        else:
            print("\nNo checkpoints found, starting from scratch")
            checkpoint_path = None
    else:
        print("\nNo checkpoints found, starting from scratch")
        checkpoint_path = None

    env = make_env()
    env = DummyVecEnv([lambda: env])

    # RESET curriculum to 0
    env.envs[0].env.total_timesteps = 0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if checkpoint_path:
        print(f"\nLoading model from: {checkpoint_path}")
        model = PPO.load(checkpoint_path, env=env, device=device)
        print("Model loaded - keeping learned weights")
    else:
        print("\nCreating new model...")
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.1,  # 10x higher - much more exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                )
            ),
            verbose=1,
            tensorboard_log="./logs_v3_baby_steps/tensorboard/",
            device=device
        )

    print("\n" + "=" * 70)
    print("RESCUE PLAN")
    print("=" * 70)
    print("Your model was struggling with 10m+ goals")
    print("New plan:")
    print("  Step 0-50k:   Goals 3-5m away (BABY STEPS)")
    print("  Step 50-100k: Goals 5-10m away (LEARNING)")
    print("  Step 100-200k: Goals 10-20m away (PRACTICING)")
    print("  Step 200k+:   Goals 20-35m away (MASTERY)")
    print("\nGoal radius: 3.0m (easier to reach)")
    print("\nNew reward features:")
    print("  - Horizontal progress rewarded separately")
    print("  - Drone gets feedback even if altitude is wrong")
    print("  - Easier to learn 'move forward = good'")
    print("=" * 70)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_v3_baby_steps/checkpoints/",
        name_prefix="ppo_rescued"
    )

    eval_env = make_env()
    eval_env = DummyVecEnv([lambda: eval_env])
    eval_env.envs[0].env.total_timesteps = 0

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_v3_baby_steps/best_model/",
        log_path="./logs_v3_baby_steps/eval/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print(f"\n{'=' * 70}")
    print("STARTING RESCUE TRAINING")
    print(f"{'=' * 70}")
    print("\nExpectations:")
    print("  - First 10-20k steps: Should start reaching 3-5m goals")
    print("  - Watch for 'SUCCESS' in episode logs")
    print("  - Curriculum will gradually increase difficulty")
    print("\nIMPORTANT:")
    print("  1. Make sure AirSim is running")
    print("  2. Verify ClockSpeed=5.0 in settings.json")
    print(f"{'=' * 70}\n")

    input("Press ENTER to start rescue training...")

    try:
        model.learn(
            total_timesteps=500_000,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True,
            reset_num_timesteps=False
        )

        final_path = "./models_v3_baby_steps/ppo_rescued_final"
        model.save(final_path)
        print(f"\nRescued model saved to: {final_path}")

    except KeyboardInterrupt:
        print("\nInterrupted!")
        model.save("./models_v3_baby_steps/ppo_rescued_interrupted")

    finally:
        env.close()
        eval_env.close()

    print("\n" + "=" * 70)
    print("RESCUE TRAINING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    rescue_training()
