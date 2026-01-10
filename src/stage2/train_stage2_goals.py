"""
STAGE 2: Add Goal Seeking
Resume from Stage 1, add goal navigation
Duration: 200k steps (~2-4 hours)

Success Criteria:
- Reaches goal 30%+ of episodes
- Mean reward > +500 per episode
- Episode length > 100 steps (navigating toward goals, not wandering)
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_stage2 import AirSimStage2Env


def make_env():
    env = AirSimStage2Env(
        goal_range_x=(5, 20),
        goal_range_y=(5, 20),
        goal_radius=3.0
    )
    return Monitor(env)


def train():
    os.makedirs("models_stage2", exist_ok=True)
    os.makedirs("logs_stage2", exist_ok=True)

    print("=" * 70)
    print("STAGE 2: GOAL SEEKING TRAINING")
    print("=" * 70)
    print("\nObjective: Navigate to random goal positions")
    print("Resumes from: Stage 1 forward movement model")
    print("Adds: Goal positions, progress rewards")
    print("\nWhat to expect:")
    print("  - First 50k steps: Starts moving toward goals")
    print("  - 50k-150k steps: Reaches goals occasionally")
    print("  - 150k+ steps: Reaches goals 30%+ of time")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nCreating environment...")
    env = make_env()
    env = DummyVecEnv([lambda: env])

    # Try to load Stage 1 model
    stage1_path = "./models_stage1/stage1_forward_final.zip"
    if os.path.exists(stage1_path):
        print(f"\n✅ Loading Stage 1 model from: {stage1_path}")
        model = PPO.load(stage1_path, env=env, device=device)
        print("✅ Successfully loaded Stage 1 model - will continue learning")
    else:
        print(f"\n⚠️  Stage 1 model not found at {stage1_path}")
        print("⚠️  Creating new model - you should train Stage 1 first!")
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
            ent_coef=0.15,  # Slightly higher exploration for new task
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                )
            ),
            verbose=1,
            tensorboard_log="./logs_stage2/tensorboard/",
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path="./models_stage2/checkpoints/",
        name_prefix="stage2_goals"
    )

    eval_env = make_env()
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_stage2/best_model/",
        log_path="./logs_stage2/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 200_000

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 2 TRAINING")
    print(f"{'=' * 70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Estimated time: 2-4 hours")
    print(f"{'=' * 70}\n")

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = "./models_stage2/stage2_goals_final"
        model.save(final_model_path)
        print(f"\n✅ Stage 2 model saved to: {final_model_path}")
        print(f"\n📊 Next: Run train_stage3_altitude.py to add altitude constraints")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save("./models_stage2/stage2_goals_interrupted")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
