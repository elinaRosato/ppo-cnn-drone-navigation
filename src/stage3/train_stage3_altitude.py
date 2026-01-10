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
        goal_range_z=(-5, -1),  # Varying altitudes
        goal_radius=3.0,
        min_altitude=-10.0,
        max_altitude=0.0
    )
    return Monitor(env)


def train():
    os.makedirs("models_stage3", exist_ok=True)
    os.makedirs("logs_stage3", exist_ok=True)

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

    # Try to load Stage 2 model
    stage2_path = "./models_stage2/stage2_goals_final.zip"
    if os.path.exists(stage2_path):
        print(f"\n✅ Loading Stage 2 model from: {stage2_path}")
        model = PPO.load(stage2_path, env=env, device=device)
        print("✅ Successfully loaded Stage 2 model - will continue learning")
    else:
        print(f"\n⚠️  Stage 2 model not found at {stage2_path}")
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
            tensorboard_log="./logs_stage3/tensorboard/",
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path="./models_stage3/checkpoints/",
        name_prefix="stage3_altitude"
    )

    eval_env = make_env()
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_stage3/best_model/",
        log_path="./logs_stage3/eval/",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 250_000

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 3 TRAINING")
    print(f"{'=' * 70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Estimated time: 3-5 hours")
    print(f"{'=' * 70}\n")

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = "./models_stage3/stage3_altitude_final"
        model.save(final_model_path)
        print(f"\n✅ Stage 3 model saved to: {final_model_path}")
        print(f"\n📊 Next: Run train_stage4_obstacles.py to add obstacle avoidance")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save("./models_stage3/stage3_altitude_interrupted")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
