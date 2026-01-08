"""
PPO V3 training with ULTRA-EASY curriculum.
Baby steps approach: Start with goals right next to the drone.

Training progression:
  Step 0-50k:   Goals 3-5m away (BABY STEPS - just move a little)
  Step 50k-150k: Goals 5-15m away (LEARNING - basic navigation)
  Step 150k-300k: Goals 15-25m away (PRACTICING - obstacle avoidance)
  Step 300k+:   Goals 25-35m away (MASTERY - full difficulty)
"""

import os
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
            'max_height': (-1.0, -1.2),  # Ceiling: random between -1.0 and -1.2
            'min_height': (-3.5, -3.7)   # Floor: random between -3.5 and -3.7 (INCREASED for larger drones)
        },
        default_max_height=-1.1,   # Default ceiling
        default_min_height=-3.6,   # Default floor: gives 2.5m range (was 1.5m)
        img_height=84,
        img_width=84,
        max_steps=500,
        goal_radius=3.0,  # Larger radius for easier starts
        # ULTRA-EASY curriculum
        curriculum_learning=True,
        curriculum_start_distance=3.0,   # VERY EASY: 3m goals
        curriculum_end_distance=35.0,    # Full difficulty: 35m goals
        curriculum_timesteps=300000      # Slower progression over 300k steps
    )
    return Monitor(env)


def train():
    os.makedirs("models_v3_baby_steps", exist_ok=True)
    os.makedirs("logs_v3_baby_steps", exist_ok=True)

    print("=" * 70)
    print("PPO V3 BABY STEPS CURRICULUM")
    print("=" * 70)
    print("\nUltra-Easy Curriculum Schedule:")
    print("  Step 0:       Goals 3m away (BABY STEPS)")
    print("  Step 50k:     Goals 8m away (TODDLER)")
    print("  Step 100k:    Goals 13m away (CHILD)")
    print("  Step 150k:    Goals 18m away (TEEN)")
    print("  Step 200k:    Goals 23m away (YOUNG ADULT)")
    print("  Step 250k:    Goals 28m away (ADULT)")
    print("  Step 300k+:   Goals 35m away (EXPERT)")
    print("\nGoal radius: 3.0m (easier to reach)")
    print("\nWhy ultra-easy start?")
    print("  - Drone learns: 'moving forward gets reward'")
    print("  - Quick successes build correct behavior")
    print("  - Avoids learning bad habits from frustration")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nCreating environment...")
    env = make_env(randomize_goals=True, randomize_bounds=True)
    env = DummyVecEnv([lambda: env])

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    print("\nCreating PPO model...")
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
        ent_coef=0.1,  # 10x higher - much more exploration to discover working behaviors
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

    print("\nModel Architecture:")
    print(model.policy)

    total_params = sum(p.numel() for p in model.policy.parameters())
    trainable_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_v3_baby_steps/checkpoints/",
        name_prefix="ppo_baby_steps"
    )

    eval_env = make_env(randomize_goals=True, randomize_bounds=True)
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_v3_baby_steps/best_model/",
        log_path="./logs_v3_baby_steps/eval/",
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 500_000

    print(f"\n{'=' * 70}")
    print("STARTING BABY STEPS TRAINING")
    print(f"{'=' * 70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Estimated time: 6-10 hours")
    print("\nWhat to expect:")
    print("  First 50k steps: Drone learns to move 3-5m (should succeed!)")
    print("  Next 100k steps: Gradually increases distance")
    print("  By 300k steps: Can handle full 35m navigation")
    print("\nMonitoring:")
    print("  - Watch for 'SUCCESS' in episode logs")
    print("  - Checkpoints: every 10,000 steps -> ./models_v3_baby_steps/checkpoints/")
    print("  - Evaluation: every 5,000 steps -> ./models_v3_baby_steps/best_model/")
    print("  - TensorBoard: tensorboard --logdir=./logs_v3_baby_steps/tensorboard/")
    print("\nIMPORTANT:")
    print("  1. Launch AirSim: Blocks.exe")
    print("  2. Verify settings.json has ClockSpeed=5.0")
    print(f"{'=' * 70}\n")

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = "./models_v3_baby_steps/ppo_baby_steps_final"
        model.save(final_model_path)
        print(f"\nFinal model saved to: {final_model_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        interrupted_model_path = "./models_v3_baby_steps/ppo_baby_steps_interrupted"
        model.save(interrupted_model_path)
        print(f"Model saved to: {interrupted_model_path}")

    finally:
        env.close()
        eval_env.close()

    print("\n" + "=" * 70)
    print("BABY STEPS TRAINING COMPLETED")
    print("=" * 70)
    print("\nThe model learned progressively:")
    print("  1. Baby steps (3-5m goals) - Building confidence")
    print("  2. Easy navigation (5-15m goals) - Learning basics")
    print("  3. Medium navigation (15-25m goals) - Obstacle avoidance")
    print("  4. Expert navigation (25-35m goals) - Full mastery")
    print("\nTo test the model:")
    print("  python test_ppo_v3.py --model ./models_v3_baby_steps/best_model/best_model.zip")
    print("=" * 70)


if __name__ == "__main__":
    train()
