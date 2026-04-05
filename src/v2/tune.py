"""
Hyperparameter optimization for PPO drone obstacle avoidance using Optuna.

Runs N trials of --trial-steps each. Each trial trains a fresh model and is
evaluated by validation success rate logged during training. Bad trials are
pruned early (MedianPruner) to save compute.

By default tunes at density stage 0 (sparse forest). Use --stage 1 or --stage 2
to tune at a harder stage, in which case provide a --checkpoint to start from
so the model is not learning from scratch on a hard environment.

Results are saved to a JSON file. Use --show-best to print the best config
without running a new study.

Usage:
    python tune.py --trials 30                          # 30 trials at stage 0
    python tune.py --trials 30 --trial-steps 500000    # longer trials
    python tune.py --trials 20 --stage 1 --checkpoint path/to/stage0_best.zip
    python tune.py --show-best                          # print best config found so far
    python tune.py --trials 30 --ros2                  # use ROS2 bridge
"""

import os
import json
import argparse
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticCnnPolicy
import torch

from avoidance_env import ObstacleAvoidanceEnv


# ── Bounded std policy (same as train.py) ────────────────────────────────────

class BoundedStdCnnPolicy(ActorCriticCnnPolicy):
    _LOG_STD_MAX = 0.0
    _LOG_STD_MIN = -4.6

    def _get_action_dist_from_latent(self, latent_pi):
        mean_actions = self.action_net(latent_pi)
        log_std = torch.clamp(self.log_std, self._LOG_STD_MIN, self._LOG_STD_MAX)
        return self.action_dist.proba_distribution(mean_actions, log_std)


# ── Optuna-aware validation callback ─────────────────────────────────────────

class TuningValidationCallback(BaseCallback):
    """Runs deterministic validation every val_every_n_steps and reports to Optuna.

    Reports the validation success rate as an intermediate value so Optuna's
    MedianPruner can kill underperforming trials early. Raises TrialPruned when
    the pruner decides the trial should stop.

    The density stage is fixed for the entire trial — no curriculum advancement.
    All trials must be comparable on the same stage.
    """

    def __init__(self, trial, val_episodes=20, val_every_n_steps=70_000, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.val_episodes = val_episodes
        self.val_every_n_steps = val_every_n_steps
        self.last_val_step = 0
        self._val_count = 0
        self._success_rates = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.num_timesteps - self.last_val_step < self.val_every_n_steps:
            return
        self.last_val_step = self.num_timesteps
        self._val_count += 1

        env = self.training_env
        successes = 0
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
        self._success_rates.append(val_success_rate)
        print(f"  [Val #{self._val_count}] step={self.num_timesteps:,} "
              f"success={val_success_rate:.0%}")

        self.trial.report(val_success_rate, step=self._val_count)
        if self.trial.should_prune():
            print(f"  [Optuna] Trial pruned at step {self.num_timesteps:,}")
            raise optuna.TrialPruned()

        self.model._last_obs = obs

    def mean_success_rate(self):
        """Return the mean of the last 3 validation success rates (or all if fewer)."""
        recent = self._success_rates[-3:] if len(self._success_rates) >= 3 else self._success_rates
        return sum(recent) / len(recent) if recent else 0.0


# ── Objective function ────────────────────────────────────────────────────────

def make_objective(trial_steps, density_stage, checkpoint_path, use_ros2):
    def objective(trial):
        # ── Sample hyperparameters ────────────────────────────────────────────
        lr          = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        clip_range  = trial.suggest_categorical("clip_range", [0.1, 0.15, 0.2, 0.25])
        n_epochs    = trial.suggest_int("n_epochs", 3, 10)
        gae_lambda  = trial.suggest_float("gae_lambda", 0.90, 0.99)
        vf_coef     = trial.suggest_float("vf_coef", 0.3, 1.0)
        target_kl   = trial.suggest_float("target_kl", 0.005, 0.05, log=True)
        ent_coef    = trial.suggest_float("ent_coef", 0.001, 0.05, log=True)
        log_std_init = trial.suggest_float("log_std_init", -2.0, 0.0)

        print(f"\n{'='*60}")
        print(f"Trial {trial.number} hyperparameters:")
        for k, v in trial.params.items():
            print(f"  {k}: {v}")
        print(f"{'='*60}")

        # ── Environment ───────────────────────────────────────────────────────
        ros2_bridge = None
        if use_ros2:
            from ros2_bridge import ROS2CameraBridge
            ros2_bridge = ROS2CameraBridge()
            ros2_bridge.start()
            import time
            t0 = time.time()
            while not ros2_bridge.has_frame:
                if time.time() - t0 > 10:
                    break
                time.sleep(0.1)

        def make_env():
            env = ObstacleAvoidanceEnv(ros2_bridge=ros2_bridge)
            return Monitor(env, info_keywords=('goal_reached',))

        env = DummyVecEnv([make_env])

        if density_stage > 0:
            for e in env.envs:
                e.unwrapped.density_stage = density_stage

        # ── Model ─────────────────────────────────────────────────────────────
        device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            if checkpoint_path:
                model = PPO.load(
                    checkpoint_path, env=env, device=device,
                    custom_objects={"policy_class": BoundedStdCnnPolicy}
                )
                model.learning_rate = lr
                model.clip_range = clip_range
                model.n_epochs = n_epochs
                model.gae_lambda = gae_lambda
                model.vf_coef = vf_coef
                model.target_kl = target_kl
                model.ent_coef = ent_coef
                print(f"  Loaded checkpoint: {checkpoint_path}")
            else:
                model = PPO(
                    BoundedStdCnnPolicy,
                    env,
                    learning_rate=lr,
                    n_steps=2048,
                    batch_size=256,
                    n_epochs=n_epochs,
                    gamma=0.99,
                    gae_lambda=gae_lambda,
                    clip_range=clip_range,
                    ent_coef=ent_coef,
                    vf_coef=vf_coef,
                    max_grad_norm=0.5,
                    target_kl=target_kl,
                    policy_kwargs={"log_std_init": log_std_init},
                    verbose=0,
                    device=device,
                )

            val_callback = TuningValidationCallback(
                trial=trial,
                val_episodes=20,
                val_every_n_steps=70_000,
            )

            model.learn(
                total_timesteps=trial_steps,
                callback=[val_callback],
                progress_bar=False,
                reset_num_timesteps=checkpoint_path is None,
            )

            score = val_callback.mean_success_rate()

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"  Trial {trial.number} failed with error: {e}")
            score = 0.0
        finally:
            env.close()
            if ros2_bridge is not None:
                ros2_bridge.stop()

        print(f"  Trial {trial.number} final score: {score:.3f}")
        return score

    return objective


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Hyperparameter tuning for PPO drone avoidance')
    parser.add_argument('--trials', type=int, default=30,
                        help='Number of Optuna trials (default: 30)')
    parser.add_argument('--trial-steps', type=int, default=300_000,
                        help='Training steps per trial (default: 300000)')
    parser.add_argument('--stage', type=int, default=0, choices=[0, 1, 2],
                        help='Density stage to tune at (default: 0 = sparse only)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint to start each trial from (recommended for --stage 1/2)')
    parser.add_argument('--study-name', type=str, default=None,
                        help='Optuna study name (default: auto-generated)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Optuna storage URL for persistence, e.g. sqlite:///tune.db')
    parser.add_argument('--show-best', action='store_true',
                        help='Print best params from existing study and exit')
    parser.add_argument('--ros2', action='store_true',
                        help='Use ROS2 bridge for images')
    args = parser.parse_args()

    stage_names = ['sparse', 'sparse+medium', 'all']
    study_name = args.study_name or f"ppo_drone_stage{args.stage}_{stage_names[args.stage]}"
    storage = args.storage or f"sqlite:///tune_stage{args.stage}.db"

    if args.show_best:
        try:
            study = optuna.load_study(study_name=study_name, storage=storage)
            print(f"\nBest trial: #{study.best_trial.number}")
            print(f"Best score: {study.best_value:.4f}")
            print("Best hyperparameters:")
            for k, v in study.best_params.items():
                print(f"  {k}: {v}")
        except Exception as e:
            print(f"Could not load study: {e}")
        return

    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=True,
    )

    print(f"\nOptuna study: {study_name}")
    print(f"Storage:      {storage}")
    print(f"Stage:        {args.stage} ({stage_names[args.stage]})")
    print(f"Trials:       {args.trials}")
    print(f"Steps/trial:  {args.trial_steps:,}")
    if args.checkpoint:
        print(f"Checkpoint:   {args.checkpoint}")
    print()

    objective = make_objective(
        trial_steps=args.trial_steps,
        density_stage=args.stage,
        checkpoint_path=args.checkpoint,
        use_ros2=args.ros2,
    )

    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    # ── Results ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("TUNING COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial:  #{study.best_trial.number}")
    print(f"Best score:  {study.best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")

    # Save best params to JSON
    out_path = f"best_params_stage{args.stage}.json"
    result = {
        "study_name": study_name,
        "stage": args.stage,
        "best_trial": study.best_trial.number,
        "best_score": study.best_value,
        "params": study.best_params,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nBest params saved to: {out_path}")
    print(f"\nTo use in training, pass these to train.py or update train.py directly.")

    # Print importance if enough trials ran
    if len(study.trials) >= 10:
        print("\nHyperparameter importances:")
        try:
            importances = optuna.importance.get_param_importances(study)
            for param, importance in importances.items():
                print(f"  {param}: {importance:.4f}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
