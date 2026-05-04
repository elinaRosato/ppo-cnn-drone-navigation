"""
Train GPS-free obstacle avoidance — v6.

Actor input : 3 stacked estimated depth frames (3,144,192) float32 [0,1]  [4:3 IMX219]
            + accumulated smoothed action scalar (1,) — GPS-free lateral displacement proxy
Critic input: privileged vector (45-dim): lat_offset + velocities + forward tree positions (x,y only) + GT depth
Depth source : Depth Anything V2 Small applied to RGB camera feed

Architecture:
  Actor  : CNN → 256 features → cat(state 1) → 257 → MLP[128,128] → action
  Critic : fully separate MLP (privileged only, never shares weights with actor)

Curriculum:
  50-episode batch win rate (win = survive full corridor without collision).
  Advancement at ≥80% win rate, handled inside the env step().
  Callback logs metrics and saves model on improvement.

Usage:
    python train.py                       # new run
    python train.py --resume              # resume latest checkpoint
    python train.py --steps 3000000      # custom step count
    python train.py --ros2               # use ROS2 bridge (~30 Hz)
    python train.py --resume --ros2
    python train.py --stage 1            # start at stage 2 (0-indexed)
"""

import argparse
import collections
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from avoidance_env import DroneAvoidanceEnv, PRIVILEGED_DIM

CNN_FEATURES_DIM   = 256   # CNN head output dim
STATE_DIM          = 1     # accumulated action scalar
ACTOR_FEATURES_DIM = CNN_FEATURES_DIM + STATE_DIM   # 257 — total actor features


# ── Actor feature extractor (CNN + accumulated action state) ──────────────────

class ActorFeaturesExtractor(BaseFeaturesExtractor):
    """
    Extracts features from 'image' (3-frame depth stack) and 'state' (accumulated
    smoothed action — GPS-free lateral displacement proxy).

    Pipeline: image → CNN → 256-dim → cat(state 1-dim) → 257-dim output
    'privileged' key is present in obs dict but deliberately ignored here.
    Image is float32 [0,1] — no /255 normalisation needed.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = ACTOR_FEATURES_DIM):
        super().__init__(observation_space, features_dim=features_dim)

        n_ch = observation_space["image"].shape[0]   # = stack_frames = 3

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
            cnn_out = self.cnn(torch.zeros(1, *observation_space["image"].shape)).shape[1]

        self.cnn_head = nn.Sequential(nn.Linear(cnn_out, CNN_FEATURES_DIM), nn.ReLU())

    def forward(self, obs):
        cnn_features = self.cnn_head(self.cnn(obs["image"]))   # (B, 256)
        state        = obs["state"]                             # (B, 1)
        return torch.cat([cnn_features, state], dim=1)          # (B, 257)


# ── Asymmetric actor-critic policy ────────────────────────────────────────────

class AsymmetricActorCriticPolicy(MultiInputActorCriticPolicy):
    """
    Actor path : ActorFeaturesExtractor → MLP[64,64] → action
    Critic path: privileged vector → separate MLP[256,256] → value

    The critic never touches the CNN or any actor weights.
    Deployed actor only needs an RGB camera + depth estimator — no privileged info.
    """

    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        # Fully separate critic — takes ONLY the privileged vector
        self.critic_net = nn.Sequential(
            nn.Linear(PRIVILEGED_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(self.device)
        self.value_net = nn.Linear(256, 1).to(self.device)
        # Rebuild optimizer so new critic parameters are included
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs
        )

    def _critic_value(self, privileged):
        return self.value_net(self.critic_net(privileged))

    def forward(self, obs, deterministic=False):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions      = distribution.get_actions(deterministic=deterministic)
        log_prob     = distribution.log_prob(actions)
        values       = self._critic_value(obs["privileged"])
        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        features     = self.extract_features(obs)
        latent_pi    = self.mlp_extractor.forward_actor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob     = distribution.log_prob(actions)
        entropy      = distribution.entropy()
        values       = self._critic_value(obs["privileged"])
        return values, log_prob, entropy

    def predict_values(self, obs):
        return self._critic_value(obs["privileged"])


# ── VecNormalize checkpoint saver ─────────────────────────────────────────────

class VecNormalizeCheckpointCallback(BaseCallback):
    def __init__(self, save_path, save_freq, verbose=0):
        super().__init__(verbose)
        self.save_path   = save_path
        self.save_freq   = save_freq
        self._last_saved = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_saved >= self.save_freq:
            self._last_saved = self.num_timesteps
            path = os.path.join(
                self.save_path, f"vecnormalize_{self.num_timesteps}_steps.pkl"
            )
            self.training_env.save(path)
        return True


# ── Training callback ─────────────────────────────────────────────────────────

class AvoidanceCallback(BaseCallback):
    """
    Logs per-rollout metrics and saves the model when 50-episode batch win rate improves.
    Curriculum advancement is handled inside DroneAvoidanceEnv.step().

    Logs per rollout:
      rollout/win_pct, rollout/episodes, rollout/mean_fwd_progress
      rollout/mean_abs_lat_offset, rollout/lat_avg_signed, rollout/collision_rate
      reward/forward, reward/action_smooth, reward/depth_penalty, reward/lateral_penalty
      curriculum/stage (logged once at rollout end, not every step)
      curriculum/batch_50ep_win_rate (on batch completion)
      curriculum/rolling_50ep_win_rate (rolling window, updated every episode)
    """

    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir      = save_dir
        self.best_win_rate = 0.0

        # Per-rollout accumulators (reset in _on_rollout_end)
        self._rollout_wins       = 0
        self._rollout_eps        = 0
        self._rollout_collisions = 0
        self._rollout_fwd        = []
        self._rollout_lat_abs    = []
        self._rollout_lat_signed = []
        self._rollout_fwd_rew    = []
        self._rollout_act_rew    = []
        self._rollout_depth_pen  = []
        self._rollout_lat_pen    = []

        # Rolling 50-episode window for win rate (mirrors v3)
        self._rolling_wins = collections.deque(maxlen=50)

    def _on_step(self) -> bool:
        dones = self.locals.get('dones', [])
        infos = self.locals.get('infos', [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            self._rollout_eps += 1
            won = info.get('survived', False)
            if won:
                self._rollout_wins += 1
            self._rolling_wins.append(1 if won else 0)

            if info.get('collided', False):
                self._rollout_collisions += 1
            if 'fwd_progress' in info:
                self._rollout_fwd.append(info['fwd_progress'])
            if 'lat_offset' in info:
                self._rollout_lat_abs.append(abs(info['lat_offset']))
            if 'lat_avg_signed' in info:
                self._rollout_lat_signed.append(info['lat_avg_signed'])
            if 'ep_fwd_reward' in info:
                self._rollout_fwd_rew.append(info['ep_fwd_reward'])
            if 'ep_action_reward' in info:
                self._rollout_act_rew.append(info['ep_action_reward'])
            if 'ep_depth_penalty' in info:
                self._rollout_depth_pen.append(info['ep_depth_penalty'])
            if 'ep_lateral_penalty' in info:
                self._rollout_lat_pen.append(info['ep_lateral_penalty'])

            # Rolling 50-ep win rate — logged every episode for a smooth curve
            if self._rolling_wins:
                self.logger.record(
                    "curriculum/rolling_50ep_win_rate",
                    sum(self._rolling_wins) / len(self._rolling_wins),
                )

        return True

    def _on_rollout_end(self) -> None:
        n = self._rollout_eps

        if n > 0:
            win_pct = self._rollout_wins / n
            self.logger.record("rollout/win_pct",       win_pct)
            self.logger.record("rollout/episodes",       n)
            self.logger.record("rollout/collision_rate", self._rollout_collisions / n)
            print(
                f"[ROLLOUT] Win: {win_pct:.0%} ({self._rollout_wins}/{n})  "
                f"Collision: {self._rollout_collisions}/{n}  "
                f"| S{DroneAvoidanceEnv._current_curriculum_stage + 1}"
            )

        def _mean(lst):
            return sum(lst) / len(lst) if lst else None

        def _log_if(key, lst):
            v = _mean(lst)
            if v is not None:
                self.logger.record(key, v)

        _log_if("rollout/mean_fwd_progress",    self._rollout_fwd)
        _log_if("rollout/mean_abs_lat_offset",  self._rollout_lat_abs)
        _log_if("rollout/lat_avg_signed",       self._rollout_lat_signed)
        _log_if("reward/forward",               self._rollout_fwd_rew)
        _log_if("reward/action_smooth",         self._rollout_act_rew)
        _log_if("reward/depth_penalty",         self._rollout_depth_pen)
        _log_if("reward/lateral_penalty",       self._rollout_lat_pen)

        # Curriculum stage logged once per rollout (not every step)
        self.logger.record(
            "curriculum/stage",
            DroneAvoidanceEnv._current_curriculum_stage + 1,
        )

        # Reset rollout accumulators
        self._rollout_wins       = 0
        self._rollout_eps        = 0
        self._rollout_collisions = 0
        self._rollout_fwd.clear()
        self._rollout_lat_abs.clear()
        self._rollout_lat_signed.clear()
        self._rollout_fwd_rew.clear()
        self._rollout_act_rew.clear()
        self._rollout_depth_pen.clear()
        self._rollout_lat_pen.clear()

        # Log and save on batch completion
        if DroneAvoidanceEnv._batch_complete:
            win_rate = DroneAvoidanceEnv._batch_win_rate
            self.logger.record("curriculum/batch_50ep_win_rate", win_rate)
            DroneAvoidanceEnv._batch_complete = False

            if win_rate >= self.best_win_rate + 0.05:
                self.best_win_rate = win_rate
                ts    = datetime.now().strftime("%m%d_%H%M")
                stage = DroneAvoidanceEnv._current_curriculum_stage + 1
                path  = os.path.join(
                    self.save_dir,
                    f"best_avoidance_v6_{ts}_win{int(win_rate * 100)}_s{stage}"
                )
                self.model.save(path)
                print(f"[SAVE] New best win rate: {win_rate:.0%} (S{stage}) → {path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_latest_run_dir(base_dir):
    if not os.path.exists(base_dir):
        return None
    run_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    )
    return os.path.join(base_dir, run_dirs[-1]) if run_dirs else None


# ── Main training function ────────────────────────────────────────────────────

def train(resume=False, checkpoint=None, target_steps=None, use_ros2=False, start_stage=0):
    _here          = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = os.path.join(_here, "../../models_v6")
    base_log_dir   = os.path.join(_here, "../../logs_v6")
    os.makedirs(base_model_dir, exist_ok=True)
    os.makedirs(base_log_dir,   exist_ok=True)

    print("=" * 70)
    print("GPS-FREE OBSTACLE AVOIDANCE — v6")
    print("=" * 70)
    print("\nActor input:  3× estimated depth frames (3,144,192) float32 [0,1]  [4:3 IMX219 FOV 62°]")
    print(f"Critic input: privileged {PRIVILEGED_DIM}-dim "
          "(lat_offset + vel + forward tree positions)")
    print("Action:       lateral ±1.0  |  forward speed coupled: fwd = v × (1 − |a|)")
    print("Win condition: cross 30 m forward without collision (timeout=300 steps)")
    image_src = "ROS2 bridge (~30 Hz)" if use_ros2 else "AirSim Python API"
    print(f"Image source: {image_src}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    # Depth estimator
    from depth_estimator import DepthEstimator
    print("\nLoading Depth Anything V2 Small...")
    depth_estimator = DepthEstimator(target_size=(192, 144), device=device)

    # Optional ROS2 bridge
    ros2_bridge = None
    if use_ros2:
        from ros2_bridge import ROS2CameraBridge
        ros2_bridge = ROS2CameraBridge(target_size=(192, 144))
        ros2_bridge.start()
        print("\nROS2 bridge started — waiting for first frames...")
        t0 = time.time()
        while not ros2_bridge.has_frame:
            if time.time() - t0 > 10.0:
                print("WARNING: ROS2 bridge received no frames after 10s.")
                break
            time.sleep(0.1)
        if ros2_bridge.has_frame:
            print("ROS2 bridge ready.")

    # Set curriculum start stage
    if start_stage > 0:
        DroneAvoidanceEnv._current_curriculum_stage = start_stage
        print(f"Starting at curriculum stage {start_stage + 1}")

    print("\nCreating environment...")
    raw_env = DummyVecEnv([lambda: Monitor(
        DroneAvoidanceEnv(
            ros2_bridge=ros2_bridge,
            depth_estimator=depth_estimator,
        )
    )])
    env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, gamma=0.99, clip_reward=10.0)

    run_dir = None
    log_dir = None

    if resume:
        run_dir = get_latest_run_dir(base_model_dir)
        if run_dir:
            ckpt_dir = os.path.join(run_dir, "checkpoints")
            latest   = None
            if os.path.exists(ckpt_dir):
                zips = [f for f in os.listdir(ckpt_dir)
                        if f.endswith('.zip') and f.startswith('avoidance_v6_')]
                if zips:
                    zips.sort(key=lambda x: int(x.split('_')[-2]))
                    latest = os.path.join(ckpt_dir, zips[-1])

            if latest:
                run_name = os.path.basename(run_dir)
                log_dir  = os.path.join(base_log_dir, run_name, "tensorboard")

                pkl_files = sorted(
                    [f for f in os.listdir(ckpt_dir)
                     if f.startswith("vecnormalize_") and f.endswith(".pkl")],
                    key=lambda x: int(x.split('_')[1]),
                )
                if pkl_files:
                    stats_path = os.path.join(ckpt_dir, pkl_files[-1])
                    env = VecNormalize.load(stats_path, raw_env)
                    env.training = True
                    print(f"   Loaded VecNormalize: {pkl_files[-1]}")

                print(f"\nResuming from: {latest}")
                model = PPO.load(
                    latest, env=env, device=device,
                    custom_objects={"policy_class": AsymmetricActorCriticPolicy},
                )
                model.tensorboard_log = log_dir
                model.ent_coef      = 0.003
                model.target_kl     = 0.05
                model.learning_rate = 3e-5
                ckpt_steps = int(latest.split('_')[-2])
                model.num_timesteps = ckpt_steps
                model._num_timesteps_at_start = ckpt_steps
                print(f"   Resuming from step: {ckpt_steps:,}")
            else:
                print("\nNo checkpoint found — starting from scratch.")
                resume = False
                run_dir = None
        else:
            print("\nNo previous runs found — starting from scratch.")
            resume = False

    if checkpoint and not resume:
        # Start a fresh run but load weights from a specific checkpoint zip.
        # Looks for a vecnormalize_<N>_steps.pkl in the same checkpoints dir.
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir   = os.path.join(base_model_dir, f"run_{timestamp}")
        log_dir   = os.path.join(base_log_dir, f"run_{timestamp}", "tensorboard")
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nNew run from checkpoint: {checkpoint}")

        ckpt_dir = os.path.dirname(checkpoint)
        pkl_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("vecnormalize_") and f.endswith(".pkl")],
            key=lambda x: int(x.split('_')[1]),
        )
        if pkl_files:
            stats_path = os.path.join(ckpt_dir, pkl_files[-1])
            env = VecNormalize.load(stats_path, raw_env)
            env.training = True
            print(f"   Loaded VecNormalize: {pkl_files[-1]}")

        model = PPO.load(
            checkpoint, env=env, device=device,
            custom_objects={"policy_class": AsymmetricActorCriticPolicy},
        )
        model.tensorboard_log = log_dir
        model.ent_coef  = 0.003
        model.target_kl = 0.05
        model.num_timesteps = 0
        model._num_timesteps_at_start = 0
        print("   Weights loaded — timestep counter reset to 0 for new run.")

    if not resume and not checkpoint:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir   = os.path.join(base_model_dir, f"run_{timestamp}")
        log_dir   = os.path.join(base_log_dir, f"run_{timestamp}", "tensorboard")
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        print(f"\nNew run: {run_dir}")

        model = PPO(
            AsymmetricActorCriticPolicy,
            env,
            learning_rate=3e-5,
            n_steps=12288,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.003,
            vf_coef=2.0,
            max_grad_norm=0.5,
            clip_range=0.15,
            target_kl=0.05,
            policy_kwargs=dict(
                features_extractor_class=ActorFeaturesExtractor,
                features_extractor_kwargs=dict(features_dim=ACTOR_FEATURES_DIM),  # 257
                net_arch=dict(pi=[128, 128], vf=[]),
                normalize_images=False,
                share_features_extractor=True,
            ),
            verbose=1,
            tensorboard_log=log_dir,
            device=device,
        )

    ckpt_dir = os.path.join(run_dir, "checkpoints")
    callbacks = [
        CheckpointCallback(
            save_freq=30_000, save_path=ckpt_dir, name_prefix="avoidance_v6"
        ),
        VecNormalizeCheckpointCallback(save_path=ckpt_dir, save_freq=30_000),
        AvoidanceCallback(save_dir=ckpt_dir),
    ]

    total_steps     = target_steps or 3_000_000
    current_steps   = getattr(model, 'num_timesteps', 0)
    remaining_steps = max(0, total_steps - current_steps)

    print(f"\n{'=' * 70}")
    print("STARTING TRAINING")
    print(f"{'=' * 70}")
    print(f"Target: {total_steps:,} steps")
    if current_steps > 0:
        print(f"Progress: {current_steps:,}  |  Remaining: {remaining_steps:,}")
    print(f"{'=' * 70}\n")

    if remaining_steps <= 0:
        print("Already at target steps.")
        env.close()
        return

    input("Press ENTER when ready...")

    try:
        model.learn(
            total_timesteps=remaining_steps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name="avoidance_v6",
        )
        final_path = os.path.join(run_dir, "avoidance_v6_final")
        model.save(final_path)
        env.save(os.path.join(run_dir, "vecnormalize_final.pkl"))
        print(f"\nModel saved: {final_path}")

    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        model.save(os.path.join(run_dir, "avoidance_v6_interrupted"))
        env.save(os.path.join(run_dir, "vecnormalize_interrupted.pkl"))

    finally:
        env.close()
        if ros2_bridge is not None:
            ros2_bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GPS-free avoidance — v6")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific .zip checkpoint to load weights from (starts a new run)")
    parser.add_argument("--steps",      type=int, default=None)
    parser.add_argument("--ros2",       action="store_true")
    parser.add_argument("--stage",      type=int, default=0, choices=[0, 1, 2],
                        help="Starting curriculum stage (0=sparse, 1=medium, 2=dense)")
    args = parser.parse_args()

    train(
        resume=args.resume,
        checkpoint=args.checkpoint,
        target_steps=args.steps,
        use_ros2=args.ros2,
        start_stage=args.stage,
    )
