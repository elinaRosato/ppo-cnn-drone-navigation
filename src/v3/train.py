"""
Train obstacle avoidance model — v3 (monocular depth estimation).

Actor input : 3 stacked estimated depth frames (3,192,192) float32 [0,1]
              + state vector [speed_norm, lateral_offset_norm]
Critic input: actor features + privileged GT min depth [0,1] (15 m range)
Depth source: Depth Anything V2 Small applied to RGB camera feed

Usage:
    python train.py                          # New training, 2M steps
    python train.py --steps 1000000         # New training, custom step count
    python train.py --resume                # Resume from latest checkpoint
    python train.py --resume --steps 2000000
    python train.py --ros2                  # Use ROS2 bridge (~30 Hz)
    python train.py --resume --ros2 --density-stage 1

Environment variables:
    AIRSIM_HOST   IP of the machine running AirSim (default: localhost).
"""

import os
import argparse
import time
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch
import torch.nn as nn


# ── Asymmetric Actor-Critic policy ────────────────────────────────────────────
#
# Actor  : CNN(depth_stack) + MLP(state) → 288 features → MLP[64,64] → action
# Critic : same 288 features + privileged(1,) → MLP[128,128] → value
#
# The actor sees estimated depth (float32 [0,1]) — no depth sensor needed at deploy.
# The critic additionally sees GT min depth from the AirSim API during training.

ACTOR_FEATURES_DIM = 288   # 256 (CNN head) + 32 (state MLP)
PRIVILEGED_DIM     = 1     # min_depth_norm (GT, critic only)


class ActorFeaturesExtractor(BaseFeaturesExtractor):
    """Extracts features from 'image' (CNN) and 'state' (MLP). Ignores 'privileged'.

    Input 'image' is float32 [0, 1] (estimated depth stack) — no /255 normalisation.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = ACTOR_FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)
        assert features_dim == ACTOR_FEATURES_DIM, \
            f"features_dim must be {ACTOR_FEATURES_DIM} (256 CNN + 32 state MLP)"

        img_space = observation_space["image"]
        n_ch = img_space.shape[0]   # number of stacked frames (3 in v3)

        self.cnn = nn.Sequential(
            nn.Conv2d(n_ch, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.cnn(torch.zeros(1, *img_space.shape)).shape[1]

        self.cnn_head  = nn.Sequential(nn.Linear(cnn_out, 256), nn.ReLU())
        self.state_mlp = nn.Sequential(
            nn.Linear(observation_space["state"].shape[0], 32), nn.ReLU()
        )

    def forward(self, obs):
        # Image is already float32 [0, 1] — no normalisation needed
        img_feat   = self.cnn_head(self.cnn(obs["image"]))
        state_feat = self.state_mlp(obs["state"])
        return torch.cat([img_feat, state_feat], dim=1)


class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """Actor-critic where the critic additionally receives privileged GT depth.

    Actor path : ActorFeaturesExtractor (depth_stack+state) → MLP[64,64] → action
    Critic path: ActorFeaturesExtractor features + privileged → MLP[128,128] → value

    The deployed actor only requires an RGB camera + depth estimator.
    """

    _LOG_STD_MAX = 0.0
    _LOG_STD_MIN = -4.6

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        self.critic_net = nn.Sequential(
            nn.Linear(ACTOR_FEATURES_DIM + PRIVILEGED_DIM, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        ).to(self.device)
        self.value_net = nn.Linear(128, 1).to(self.device)
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        log_std = torch.clamp(self.log_std, self._LOG_STD_MIN, self._LOG_STD_MAX)
        return self.action_dist.proba_distribution(mean_actions, log_std)

    def _critic_value(self, features, privileged):
        return self.value_net(self.critic_net(torch.cat([features, privileged], dim=1)))

    def forward(self, obs, deterministic=False):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions      = distribution.get_actions(deterministic=deterministic)
        log_prob     = distribution.log_prob(actions)
        values       = self._critic_value(features, obs["privileged"])
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob     = distribution.log_prob(actions)
        entropy      = distribution.entropy()
        values       = self._critic_value(features, obs["privileged"])
        return values, log_prob, entropy

    def predict_values(self, obs):
        features = self.extract_features(obs)
        return self._critic_value(features, obs["privileged"])


# ── VecNormalize checkpoint saver ─────────────────────────────────────────────

class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.save_freq = save_freq
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= self.save_freq:
            self._last_saved = self.num_timesteps
            path = os.path.join(
                self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl"
            )
            self.training_env.save(path)
        return True


class ValidationCallback(BaseCallback):
    STAGE_NAMES = ['sparse only', 'sparse + medium', 'all densities']

    def __init__(self, val_episodes=30, val_every_n_steps=70_000,
                 density_threshold=0.80, verbose=0):
        super().__init__(verbose)
        self.val_episodes       = val_episodes
        self.val_every_n_steps  = val_every_n_steps
        self.density_threshold  = density_threshold
        self.last_val_step      = 0
        self._passes_in_a_row   = 0

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_val_step < self.val_every_n_steps:
            return
        self.last_val_step = self.num_timesteps

        current_stage = self.training_env.envs[0].unwrapped.density_stage
        print(f"\n[Validation] Running {self.val_episodes} deterministic episodes "
              f"at step {self.num_timesteps:,} | "
              f"density stage={current_stage} ({self.STAGE_NAMES[current_stage]})...")

        for e in self.training_env.envs:
            e.unwrapped.training_mode = False

        env = self.training_env
        successes = 0
        val_avg_laterals     = []
        val_avg_abs_laterals = []

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
            self.logger.record('validation/lateral_avg',
                               sum(val_avg_laterals) / len(val_avg_laterals))
            self.logger.record('validation/lateral_abs_avg',
                               sum(val_avg_abs_laterals) / len(val_avg_abs_laterals))

        for e in self.training_env.envs:
            e.unwrapped.training_mode = True

        print(f"[Validation] Success rate: {val_success_rate:.0%} "
              f"({successes}/{self.val_episodes})")

        if current_stage < 2:
            if val_success_rate >= self.density_threshold:
                self._passes_in_a_row += 1
                print(f"[Curriculum] Pass {self._passes_in_a_row}/2 "
                      f"({val_success_rate:.0%} ≥ {self.density_threshold:.0%})")
                if self._passes_in_a_row >= 2:
                    new_stage = current_stage + 1
                    for env_ in self.training_env.envs:
                        env_.unwrapped.density_stage = new_stage
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
        self.model._last_obs = obs


class SuccessRateCallback(BaseCallback):
    def __init__(self, success_window=50, verbose=0):
        super().__init__(verbose)
        self.success_window = success_window
        self.episode_count  = 0
        self.outcomes       = []
        self.avg_laterals       = []
        self.avg_abs_laterals   = []
        self.ep_proximity_rewards    = []
        self.ep_straight_bonuses     = []
        self.ep_action_norm_penalties = []
        self.ep_drift_penalties      = []

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
                    self.ep_drift_penalties.append(info.get('ep_drift_penalty', 0.0))
        return True

    def _on_rollout_end(self) -> None:
        if not self.outcomes:
            return
        self.logger.record('rollout/success_rate',
                           sum(self.outcomes) / len(self.outcomes))
        self.logger.record('rollout/episodes', self.episode_count)
        if self.avg_laterals:
            self.logger.record('rollout/lateral_avg',
                               sum(self.avg_laterals) / len(self.avg_laterals))
            self.logger.record('rollout/lateral_abs_avg',
                               sum(self.avg_abs_laterals) / len(self.avg_abs_laterals))
        if self.ep_proximity_rewards:
            self.logger.record('reward/proximity',
                               sum(self.ep_proximity_rewards) / len(self.ep_proximity_rewards))
            self.logger.record('reward/straight_bonus',
                               sum(self.ep_straight_bonuses) / len(self.ep_straight_bonuses))
            self.logger.record('reward/action_norm',
                               sum(self.ep_action_norm_penalties) / len(self.ep_action_norm_penalties))
            self.logger.record('reward/drift',
                               sum(self.ep_drift_penalties) / len(self.ep_drift_penalties))
            self.ep_proximity_rewards.clear()
            self.ep_straight_bonuses.clear()
            self.ep_action_norm_penalties.clear()
            self.ep_drift_penalties.clear()


from avoidance_env import ObstacleAvoidanceEnv


def make_env(ros2_bridge=None, depth_estimator=None):
    def _init():
        env = ObstacleAvoidanceEnv(
            ros2_bridge=ros2_bridge,
            depth_estimator=depth_estimator,
        )
        return Monitor(env, info_keywords=('goal_reached',))
    return _init


def get_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None
    run_dirs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    ]
    if not run_dirs:
        return None
    run_dirs.sort()
    return os.path.join(base_dir, run_dirs[-1])


def train(resume=False, target_steps=None, use_ros2=False, density_stage=0):
    _here = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = os.path.join(_here, "../../models_v3")
    base_log_dir   = os.path.join(_here, "../../logs_v3")
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir,   exist_ok=True)

    print("=" * 70)
    print("OBSTACLE AVOIDANCE TRAINING — v3 (monocular depth estimation)")
    print("=" * 70)
    print("\nActor input:  3x estimated depth frames (3,192,192) float32 [0,1]")
    print("              + state [speed_norm, lat_offset_norm]")
    print("Critic input: actor features + privileged GT min depth [0,1]")
    print("Depth model:  Depth Anything V2 Small")
    image_src = "ROS2 bridge (~30 Hz)" if use_ros2 else "AirSim Python API (~1-5 Hz)"
    print(f"Image source: {image_src}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Initialise depth estimator (shared across all env instances)
    from depth_estimator import DepthEstimator
    print("\nLoading depth estimator...")
    depth_estimator = DepthEstimator(target_size=(192, 192), device=device)

    ros2_bridge = None
    if use_ros2:
        from ros2_bridge import ROS2CameraBridge
        ros2_bridge = ROS2CameraBridge()
        ros2_bridge.start()
        print("\nROS2 bridge started — waiting for first frames...")
        timeout = 10.0
        t0 = time.time()
        while not ros2_bridge.has_frame:
            if time.time() - t0 > timeout:
                print(f"WARNING: ROS2 bridge received no frames after {timeout:.0f}s.")
                break
            time.sleep(0.1)
        if ros2_bridge.has_frame:
            print("ROS2 bridge ready.")

    print("\nCreating environment...")
    raw_env = DummyVecEnv([make_env(ros2_bridge=ros2_bridge, depth_estimator=depth_estimator)])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0)

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
            checkpoint_dir  = os.path.join(run_dir, "checkpoints")
            latest_checkpoint = None

            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.zip')]
                if checkpoints:
                    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
                    latest_checkpoint = os.path.join(checkpoint_dir, checkpoints[-1])

            if latest_checkpoint:
                run_name = os.path.basename(run_dir)
                log_dir  = os.path.join(base_log_dir, run_name, "tensorboard")

                pkl_files = sorted(
                    [f for f in os.listdir(checkpoint_dir)
                     if f.startswith("vecnormalize_") and f.endswith(".pkl")],
                    key=lambda x: int(x.split('_')[1]),
                )
                if pkl_files:
                    stats_path = os.path.join(checkpoint_dir, pkl_files[-1])
                    env = VecNormalize.load(stats_path, raw_env)
                    env.training = True
                    print(f"   Loaded VecNormalize stats: {pkl_files[-1]}")

                print(f"\nResuming from checkpoint: {latest_checkpoint}")
                model = PPO.load(
                    latest_checkpoint, env=env, device=device,
                    custom_objects={"policy_class": AsymmetricActorCriticPolicy},
                )
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

        model = PPO(
            AsymmetricActorCriticPolicy,
            env,
            learning_rate=lambda p: 3e-5 + (1.47e-4 - 3e-5) * p,
            n_steps=4096,
            batch_size=512,
            n_epochs=9,
            gamma=0.99,
            gae_lambda=0.940,
            clip_range=0.15,
            ent_coef=0.005,
            vf_coef=1.0,
            max_grad_norm=0.993,
            target_kl=0.033,
            policy_kwargs=dict(
                features_extractor_class=ActorFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=ACTOR_FEATURES_DIM),
                net_arch=dict(pi=[64, 64], vf=[]),
                normalize_images=False,   # images are already float32 [0,1]
                log_std_init=-0.111,
            ),
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )

    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    checkpoint_callback = CheckpointCallback(
        save_freq=30000,
        save_path=checkpoint_dir,
        name_prefix="avoidance_v3",
    )
    vecnorm_callback = VecNormalizeCheckpointCallback(
        save_path=checkpoint_dir,
        save_freq=30000,
    )
    success_callback    = SuccessRateCallback(success_window=50)
    validation_callback = ValidationCallback(
        val_episodes=30, val_every_n_steps=70_000, density_threshold=0.80
    )

    total_timesteps    = target_steps if target_steps else 2_000_000
    current_steps      = model.num_timesteps if hasattr(model, 'num_timesteps') else 0
    remaining_timesteps = max(0, total_timesteps - current_steps)

    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print(f"{'=' * 70}")
    print(f"Target timesteps: {total_timesteps:,}")
    if current_steps > 0:
        print(f"Current progress: {current_steps:,}")
        print(f"Remaining steps:  {remaining_timesteps:,}")
    print(f"{'=' * 70}\n")

    if remaining_timesteps <= 0:
        print("Already reached target timesteps!")
        env.close()
        return

    input("Press ENTER when ready...")

    try:
        model.learn(
            total_timesteps=remaining_timesteps,
            callback=[checkpoint_callback, vecnorm_callback,
                      success_callback, validation_callback],
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="avoidance_v3",
        )

        final_path = os.path.join(run_dir, "avoidance_v3_final")
        model.save(final_path)
        env.save(os.path.join(run_dir, "vecnormalize_final.pkl"))
        print(f"\nModel saved to: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted!")
        interrupted_path = os.path.join(run_dir, "avoidance_v3_interrupted")
        model.save(interrupted_path)
        env.save(os.path.join(run_dir, "vecnormalize_interrupted.pkl"))

    finally:
        env.close()
        if ros2_bridge is not None:
            ros2_bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train obstacle avoidance model (v3)')
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--steps',         type=int,  default=None)
    parser.add_argument('--ros2',          action='store_true')
    parser.add_argument('--density-stage', type=int,  default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    train(
        resume=args.resume,
        target_steps=args.steps,
        use_ros2=args.ros2,
        density_stage=args.density_stage,
    )
