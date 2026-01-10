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
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_stage1 import AirSimStage1Env


def make_env():
    env = AirSimStage1Env()
    return Monitor(env)


def train():
    os.makedirs("models_stage1", exist_ok=True)
    os.makedirs("logs_stage1", exist_ok=True)

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

    print("\nCreating PPO model...")
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
        tensorboard_log="./logs_stage1/tensorboard/",
        device=device
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models_stage1/checkpoints/",
        name_prefix="stage1_forward"
    )

    total_timesteps = 50_000

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 1 TRAINING")
    print(f"{'=' * 70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Estimated time: 30 min - 1 hour")
    print(f"{'=' * 70}\n")

    input("Press ENTER when AirSim is ready...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback],
            progress_bar=True
        )

        final_model_path = "./models_stage1/stage1_forward_final"
        model.save(final_model_path)
        print(f"\n✅ Stage 1 model saved to: {final_model_path}")
        print(f"\n📊 Next: Run train_stage2_goals.py to add goal seeking")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save("./models_stage1/stage1_forward_interrupted")

    finally:
        env.close()


if __name__ == "__main__":
    train()
