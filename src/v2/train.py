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
import torch


class ValidationCallback(BaseCallback):
    """Runs deterministic validation episodes every N training episodes.

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
                 density_threshold=0.80, verbose=0):
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

        val_success_rate = successes / self.val_episodes
        self.logger.record('validation/success_rate', val_success_rate)
        self.logger.record('validation/step', self.num_timesteps)
        self.logger.record('curriculum/density_stage', current_stage)

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


class EntropyScheduleCallback(BaseCallback):
    """Reduces ent_coef on a fixed step schedule.

    Default schedule:
        0 steps    → 0.10  (high exploration at start)
        200k steps → 0.08
        400k steps → 0.06
        600k steps → 0.04
        800k steps → 0.02
        1M  steps  → 0.01  (held for remainder of training)

    step_offset: shifts the schedule so it runs relative to the current
    timestep rather than from 0. Use this when resuming at a higher density
    stage — e.g. resuming at step 1.28M with step_offset=1.28M restarts the
    full 0.10→0.01 ramp from the current position, giving the policy room to
    explore the new denser environment.
    """

    SCHEDULE = [
        (0,         0.10),
        (200_000,   0.08),
        (400_000,   0.06),
        (600_000,   0.04),
        (800_000,   0.02),
        (1_000_000, 0.01),
    ]

    def __init__(self, step_offset=0, verbose=0):
        super().__init__(verbose)
        self._current_stage = -1
        self.step_offset = step_offset

    def _on_step(self) -> bool:
        effective_step = self.num_timesteps - self.step_offset
        stage = 0
        for i, (threshold, _) in enumerate(self.SCHEDULE):
            if effective_step >= threshold:
                stage = i
        if stage != self._current_stage:
            coef = self.SCHEDULE[stage][1]
            self.model.ent_coef = coef
            self._current_stage = stage
            print(f"[EntropySchedule] step={self.num_timesteps:,}: ent_coef → {coef}")
            self.logger.record('train/ent_coef_scheduled', coef)
        return True


class SuccessRateCallback(BaseCallback):
    """Logs episode success rate to TensorBoard.

    Tracks outcomes over a rolling window of `window` episodes so the metric
    reflects recent performance rather than the cumulative average.
    Logs every `window` completed episodes.
    """

    def __init__(self, window=50, verbose=0):
        super().__init__(verbose)
        self.window = window
        self.episode_count = 0
        self.outcomes = []  # rolling window: 1 = success, 0 = failure

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for done, info in zip(dones, infos):
            if done:
                self.episode_count += 1
                self.outcomes.append(1 if info.get('goal_reached', False) else 0)
                if len(self.outcomes) > self.window:
                    self.outcomes.pop(0)
                if self.episode_count % self.window == 0:
                    success_rate = sum(self.outcomes) / len(self.outcomes)
                    self.logger.record('rollout/success_rate', success_rate)
                    self.logger.record('rollout/episodes', self.episode_count)
        return True

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
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.10,
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
    success_callback = SuccessRateCallback(window=50)
    validation_callback = ValidationCallback(val_episodes=30, val_every_n_steps=70_000,
                                             density_threshold=0.80)
    # When resuming at a higher density stage the entropy schedule would
    # immediately snap to 0.01 (its value at 1M+ steps). Reset it relative
    # to the current checkpoint so the policy can explore the new environment.
    entropy_offset = model.num_timesteps if density_stage > 0 else 0
    entropy_callback = EntropyScheduleCallback(step_offset=entropy_offset)

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
            callback=[checkpoint_callback, success_callback, validation_callback,
                      entropy_callback],
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
