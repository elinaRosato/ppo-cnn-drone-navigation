"""
Train obstacle avoidance model.

The model learns to output lateral/vertical corrections
based on camera input to avoid obstacles.
Navigation is handled by a simple controller.

Usage:
    python train.py                                    # New training, 200k steps
    python train.py --steps 500000                     # New training, 500k steps
    python train.py --resume                           # Resume from latest checkpoint
    python train.py --resume --steps 400000            # Resume, train to 400k total
    python train.py --ros2                             # Use ROS2 bridge for images (~30 Hz)
    python train.py --resume --ros2 --density-stage 1  # Resume at sparse+medium density
    python train.py --resume --ros2 --density-stage 2  # Resume at all densities

Environment variables:
    AIRSIM_HOST   IP of the machine running AirSim (default: localhost).
                  Set this when training from WSL2:
                    export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
"""

import os
import argparse
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch


class BoundedStdCnnPolicy(ActorCriticCnnPolicy):
    """CnnPolicy where action std is hard-capped at 1.0 (the action range width).

    SB3's default LOG_STD_MAX=2 allows std up to ~7.4, causing sampled actions
    to be almost always clipped to ±1 — exploration wastes its budget outside
    the valid range. Overriding _get_action_dist_from_latent to clamp log_std
    keeps std within [0.01, 1.0] so exploration stays meaningful inside [-1, 1].
    """
    _LOG_STD_MAX = 0.0   # e^0    = 1.0  — std never exceeds the action range
    _LOG_STD_MIN = -4.6  # e^-4.6 ≈ 0.01 — still allows tight exploitation

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        log_std = torch.clamp(self.log_std, self._LOG_STD_MIN, self._LOG_STD_MAX)
        return self.action_dist.proba_distribution(mean_actions, log_std)


class ValidationCallback(BaseCallback):
    """Runs deterministic validation episodes every N training steps.

    Uses model.predict(deterministic=True) so results reflect the true
    policy quality, not the stochastic exploration used during training.
    Logs to validation/success_rate and curriculum/density_stage in TensorBoard.

    Curriculum advancement: if validation success rate exceeds density_threshold
    for two consecutive evaluations, the forest density stage is incremented
    (sparse only → sparse+medium → all densities). Stage is stored on the
    ObstacleAvoidanceEnv as density_stage and updated via VecEnv.set_attr().

    Validation uses the same training environment and therefore the same
    density_stage that is active during training — no separate eval env needed.
    The stage is only advanced *after* all val episodes finish, so all val
    episodes in one run see a consistent density config.
    """

    STAGE_NAMES = ['sparse only', 'sparse + medium', 'all densities']

    def __init__(self, val_episodes=30, val_every_n_steps=70_000,
                 density_threshold=0.80,
                 verbose=0):
        super().__init__(verbose)
        self.val_episodes = val_episodes
        self.val_every_n_steps = val_every_n_steps
        self.density_threshold = density_threshold
        self.last_val_step = 0
        self._passes_in_a_row = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_val_step < self.val_every_n_steps:
            return
        self.last_val_step = self.num_timesteps

        # Read density stage directly from the unwrapped env.
        # DummyVecEnv.get_attr() / set_attr() operate on the Monitor wrapper, not
        # on ObstacleAvoidanceEnv itself. Gymnasium's Wrapper.__getattr__ forwards
        # reads transparently, but set_attr() creates a shadow attribute on Monitor
        # that the env never sees. Using .unwrapped bypasses all wrappers.
        current_stage = self.training_env.envs[0].unwrapped.density_stage
        print(f"\n[Validation] Running {self.val_episodes} deterministic episodes "
              f"at step {self.num_timesteps:,} | "
              f"density stage={current_stage} ({self.STAGE_NAMES[current_stage]})...")

        env = self.training_env
        successes = 0
        val_avg_laterals = []
        val_avg_abs_laterals = []

        # Reset once before the loop. After each episode ends with done=True,
        # DummyVecEnv auto-resets and returns the new initial obs from step(),
        # so the next iteration starts immediately without an extra reset().
        obs = env.reset()
        for _ in range(self.val_episodes):
            done = False
            info = {}
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = env.step(action)
                done = bool(dones[0])
                info = infos[0]
            if info.get('goal_reached', False):
                successes += 1
            if 'avg_lateral' in info:
                val_avg_laterals.append(info['avg_lateral'])
                val_avg_abs_laterals.append(info['avg_abs_lateral'])

        val_success_rate = successes / self.val_episodes
        self.logger.record('validation/success_rate', val_success_rate)
        self.logger.record('validation/step', self.num_timesteps)
        self.logger.record('curriculum/density_stage', current_stage)
        if val_avg_laterals:
            self.logger.record('validation/lateral_avg', sum(val_avg_laterals) / len(val_avg_laterals))
            self.logger.record('validation/lateral_abs_avg', sum(val_avg_abs_laterals) / len(val_avg_abs_laterals))

        print(f"[Validation] Success rate: {val_success_rate:.0%} "
              f"({successes}/{self.val_episodes})")

        # Curriculum: advance density stage after 2 consecutive passes above threshold
        if current_stage < 2:
            if val_success_rate >= self.density_threshold:
                self._passes_in_a_row += 1
                print(f"[Curriculum] Pass {self._passes_in_a_row}/2 "
                      f"({val_success_rate:.0%} ≥ {self.density_threshold:.0%})")
                if self._passes_in_a_row >= 2:
                    new_stage = current_stage + 1
                    for env in self.training_env.envs:
                        env.unwrapped.density_stage = new_stage
                    self._passes_in_a_row = 0
                    self.logger.record('curriculum/density_stage', new_stage)
                    print(f"[Curriculum] Density stage advanced to {new_stage}: "
                          f"{self.STAGE_NAMES[new_stage]}")
            else:
                if self._passes_in_a_row > 0:
                    print(f"[Curriculum] Pass streak reset (was {self._passes_in_a_row})")
                self._passes_in_a_row = 0

        self.logger.dump(self.num_timesteps)
        print()
        # Sync model._last_obs so the next rollout starts from the obs that
        # DummyVecEnv returned after the last validation episode ended.
        # When done=True, DummyVecEnv auto-resets and returns the new episode's
        # initial obs — so `obs` already IS that fresh observation. Assigning
        # it directly avoids a second reset() call (and a second tree placement)
        # which was causing the double Goal print at every validation boundary.
        self.model._last_obs = obs




class SuccessRateCallback(BaseCallback):
    """Logs episode success rate to TensorBoard.

    Tracks outcomes over a rolling window of `window` episodes so the metric
    reflects recent performance rather than the cumulative average.
    Logs every `window` completed episodes.
    """

    def __init__(self, success_window=50, verbose=0):
        super().__init__(verbose)
        self.success_window = success_window
        self.episode_count = 0
        # Rolling windows (never clear — smoothed over last N episodes)
        self.outcomes = []
        self.avg_laterals = []
        self.avg_abs_laterals = []
        # Per-rollout accumulators (clear each rollout)
        self.ep_proximity_rewards = []
        self.ep_straight_bonuses = []
        self.ep_action_norm_penalties = []

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                self.outcomes.append(1 if info.get('goal_reached', False) else 0)
                if len(self.outcomes) > self.success_window:
                    self.outcomes.pop(0)
                if 'avg_lateral' in info:
                    self.avg_laterals.append(info['avg_lateral'])
                    self.avg_abs_laterals.append(info['avg_abs_lateral'])
                    if len(self.avg_laterals) > self.success_window:
                        self.avg_laterals.pop(0)
                        self.avg_abs_laterals.pop(0)
                if 'ep_proximity_reward' in info:
                    self.ep_proximity_rewards.append(info['ep_proximity_reward'])
                    self.ep_straight_bonuses.append(info['ep_straight_bonus'])
                    self.ep_action_norm_penalties.append(info['ep_action_norm_penalty'])
        return True

    def _on_rollout_end(self) -> None:
        if not self.outcomes:
            return
        # Success rate and lateral: smoothed rolling window
        self.logger.record('rollout/success_rate', sum(self.outcomes) / len(self.outcomes))
        self.logger.record('rollout/episodes', self.episode_count)
        if self.avg_laterals:
            self.logger.record('rollout/lateral_avg', sum(self.avg_laterals) / len(self.avg_laterals))
            self.logger.record('rollout/lateral_abs_avg', sum(self.avg_abs_laterals) / len(self.avg_abs_laterals))
        # Reward components: per-rollout average then clear
        if self.ep_proximity_rewards:
            self.logger.record('reward/proximity', sum(self.ep_proximity_rewards) / len(self.ep_proximity_rewards))
            self.logger.record('reward/straight_bonus', sum(self.ep_straight_bonuses) / len(self.ep_straight_bonuses))
            self.logger.record('reward/action_norm', sum(self.ep_action_norm_penalties) / len(self.ep_action_norm_penalties))
            self.ep_proximity_rewards.clear()
            self.ep_straight_bonuses.clear()
            self.ep_action_norm_penalties.clear()

from avoidance_env import ObstacleAvoidanceEnv


def make_env(ros2_bridge=None):
    def _init():
        env = ObstacleAvoidanceEnv(ros2_bridge=ros2_bridge)
        return Monitor(env, info_keywords=('goal_reached',))
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


def train(resume=False, target_steps=None, use_ros2=False, density_stage=0):
    base_model_dir = "./models_v2"
    base_log_dir = "./logs_v2"
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir, exist_ok=True)

    print("=" * 70)
    print("OBSTACLE AVOIDANCE TRAINING")
    print("=" * 70)
    print("\nModel input:  4x grayscale frames (4, 128, 128)")
    print("Model output: Lateral correction only")
    print("Controller:   Flies toward goal automatically")
    image_src = "ROS2 bridge (~30 Hz)" if use_ros2 else "AirSim Python API (~1-5 Hz)"
    print(f"Image source: {image_src}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    ros2_bridge = None
    if use_ros2:
        from ros2_bridge import ROS2CameraBridge
        ros2_bridge = ROS2CameraBridge()
        ros2_bridge.start()
        print("\nROS2 bridge started — waiting for first frames...")
        import time
        # Give the subscriber a moment to receive at least one frame before
        # the environment's reset() tries to capture an image
        timeout = 10.0
        t0 = time.time()
        while not ros2_bridge.has_frame:
            if time.time() - t0 > timeout:
                print("WARNING: ROS2 bridge received no frames after "
                      f"{timeout:.0f}s. Is the AirSim ROS2 node running?")
                break
            time.sleep(0.1)
        if ros2_bridge.has_frame:
            print("ROS2 bridge ready.")

    print("\nCreating environment...")
    env = DummyVecEnv([make_env(ros2_bridge=ros2_bridge)])

    if density_stage > 0:
        for e in env.envs:
            e.unwrapped.density_stage = density_stage
        stage_names = ['sparse only', 'sparse + medium', 'all densities']
        print(f"Density stage set to {density_stage}: {stage_names[density_stage]}")

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
                model = PPO.load(latest_checkpoint, env=env, device=device,
                                 custom_objects={"policy_class": BoundedStdCnnPolicy})
                model.tensorboard_log = log_dir
                print(f"   Resuming with ent_coef: {model.ent_coef:.4f}")
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
            BoundedStdCnnPolicy,
            env,
            learning_rate=1e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=5,
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
        save_freq=30000,
        save_path=os.path.join(run_dir, "checkpoints"),
        name_prefix="simplified_avoidance"
    )
    success_callback = SuccessRateCallback(success_window=50)
    validation_callback = ValidationCallback(val_episodes=30, val_every_n_steps=70_000,
                                             density_threshold=0.80)

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
        return

    print("CHECKLIST:")
    print("  [ ] AirSim is running with your environment")
    if use_ros2:
        print("  [ ] ROS2 bridge is running (ros2 launch airsim_ros_pkgs airsim_node.launch.py ...)")
    print("  [ ] GPU/CUDA available (recommended)")
    print("")

    input("Press ENTER when ready...")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, success_callback, validation_callback],
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
        if ros2_bridge is not None:
            ros2_bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train obstacle avoidance model')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from latest checkpoint')
    parser.add_argument('--steps', type=int, default=None,
                        help='Target total timesteps (default: 200000)')
    parser.add_argument('--ros2', action='store_true',
                        help='Use ROS2 bridge for high-frequency image capture (~30 Hz). '
                             'Requires the AirSim ROS2 node to be running and AIRSIM_HOST set.')
    parser.add_argument('--density-stage', type=int, default=0, choices=[0, 1, 2],
                        help='Starting forest density stage: 0=sparse only (default), '
                             '1=sparse+medium, 2=all densities. '
                             'Useful when resuming a run that trained on sparse only.')
    args = parser.parse_args()

    train(resume=args.resume, target_steps=args.steps, use_ros2=args.ros2,
          density_stage=args.density_stage)
