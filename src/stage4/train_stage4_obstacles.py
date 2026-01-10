"""
STAGE 4: Add Obstacle Avoidance
Resume from Stage 3, add obstacles using Blocks environment
Duration: 400k steps (~4-8 hours)

Success Criteria:
- Reaches goal 30%+ of episodes (with obstacles)
- Mean reward > +500 per episode
- Collision rate < 20% of episodes

IMPORTANT: Use the 'Blocks' environment in AirSim for this stage!
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch

from airsim_env_stage4 import AirSimStage4Env


def make_env():
    env = AirSimStage4Env(
        goal_range_x=(10, 30),  # Further goals
        goal_range_y=(10, 30),
        goal_range_z=(-5, -1),
        goal_radius=3.0,
        min_altitude=-10.0,
        max_altitude=0.0
    )
    return Monitor(env)


def train():
    os.makedirs("models_stage4", exist_ok=True)
    os.makedirs("logs_stage4", exist_ok=True)

    print("=" * 70)
    print("STAGE 4: OBSTACLE AVOIDANCE TRAINING")
    print("=" * 70)
    print("\n⚠️  IMPORTANT: Make sure you're using the 'Blocks' environment!")
    print("   (Empty environment won't teach obstacle avoidance)")
    print("\nObjective: Navigate to goals while avoiding obstacles")
    print("Resumes from: Stage 3 altitude constraints model")
    print("Adds: Obstacles, camera-based vision for avoidance")
    print("\nWhat to expect:")
    print("  - First 100k steps: Frequent collisions, learning obstacles exist")
    print("  - 100k-250k steps: Starts avoiding simple obstacles")
    print("  - 250k+ steps: Navigates around obstacles 30%+ of time")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    print("\nCreating environment...")
    env = make_env()
    env = DummyVecEnv([lambda: env])

    # Try to load Stage 3 model
    stage3_path = "./models_stage3/stage3_altitude_final.zip"
    if os.path.exists(stage3_path):
        print(f"\n✅ Loading Stage 3 model from: {stage3_path}")
        model = PPO.load(stage3_path, env=env, device=device)
        print("✅ Successfully loaded Stage 3 model - will continue learning")
    else:
        print(f"\n⚠️  Stage 3 model not found at {stage3_path}")
        print("⚠️  Creating new model - you should train Stage 3 first!")
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
            ent_coef=0.15,  # Higher exploration for obstacles
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(
                    pi=[256, 256],
                    vf=[256, 256]
                )
            ),
            verbose=1,
            tensorboard_log="./logs_stage4/tensorboard/",
            device=device
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path="./models_stage4/checkpoints/",
        name_prefix="stage4_obstacles"
    )

    eval_env = make_env()
    eval_env = DummyVecEnv([lambda: eval_env])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models_stage4/best_model/",
        log_path="./logs_stage4/eval/",
        eval_freq=20000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    total_timesteps = 400_000

    print(f"\n{'=' * 70}")
    print("STARTING STAGE 4 TRAINING")
    print(f"{'=' * 70}")
    print(f"Total timesteps: {total_timesteps:,}")
    print("Estimated time: 4-8 hours")
    print(f"{'=' * 70}\n")

    print("⚠️  CHECKLIST BEFORE STARTING:")
    print("  [ ] AirSim is running with 'Blocks' environment")
    print("  [ ] Stage 3 model completed training")
    print("  [ ] GPU/CUDA available (if desired)")
    print("")

    input("Press ENTER when ready...")

    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )

        final_model_path = "./models_stage4/stage4_obstacles_final"
        model.save(final_model_path)
        print(f"\n✅ Stage 4 model saved to: {final_model_path}")
        print(f"\n🎉 CURRICULUM COMPLETE! You have a fully trained autonomous drone!")
        print(f"\n📊 To test your trained drone:")
        print(f"   1. Use the model at: {final_model_path}.zip")
        print(f"   2. Switch to PX4 if you want to test with real flight controller")
        print(f"   3. Deploy to real drone for field testing")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        model.save("./models_stage4/stage4_obstacles_interrupted")

    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    train()
