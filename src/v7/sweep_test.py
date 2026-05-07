"""
Parameter sweep — v7 GPS-free avoidance.

Loads the depth estimator and model once, then runs every configuration in
CONFIGS sequentially, printing per-episode results and a ranked summary table.

Edit CONFIGS below to define what to sweep.

Usage:
    python3 sweep_test.py                        # all configs, 30 eps each
    python3 sweep_test.py --episodes 50
    python3 sweep_test.py --model path/to.zip
    python3 sweep_test.py --stage 2
    python3 sweep_test.py --ros2
"""

import os
import sys
import argparse
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from avoidance_env import DroneAvoidanceEnv, FRAME_STRIDE as _D_STRIDE, ACTION_MOMENTUM as _D_MOM
from depth_estimator import DepthEstimator
from train import AsymmetricActorCriticPolicy


# ── Sweep configurations ───────────────────────────────────────────────────────
# Each dict defines one run. All keys except 'name' are optional —
# omit any key to use the trained default.
#
#   fwd    float   forward speed m/s      (trained: 1.0)
#   lat    float   max lateral speed m/s  (trained: 0.8)
#   stride int     frame stride           (trained: 4)
#   mom    float   action momentum        (trained: 0.3)
#   hz     int     control rate Hz        (trained: 10)

CONFIGS = [
    # ── Reference (trained defaults) ──────────────────────────────────────
    {"name": "baseline"},

    # ── Best single-run result ─────────────────────────────────────────────
    {"name": "s3·1.5/1.5·m0.1",   "fwd": 1.5, "lat": 1.5, "stride": 3, "mom": 0.1},

    # ── Momentum sweep  (stride=3, fwd=1.5, lat=1.5) ──────────────────────
    {"name": "s3·1.5/1.5·m0.0",   "fwd": 1.5, "lat": 1.5, "stride": 3, "mom": 0.0},
    {"name": "s3·1.5/1.5·m0.2",   "fwd": 1.5, "lat": 1.5, "stride": 3, "mom": 0.2},
    {"name": "s3·1.5/1.5·m0.3",   "fwd": 1.5, "lat": 1.5, "stride": 3, "mom": 0.3},

    # ── Lateral speed sweep  (stride=3, fwd=1.5, mom=0.1) ─────────────────
    {"name": "s3·1.5/0.8·m0.1",   "fwd": 1.5, "lat": 0.8, "stride": 3, "mom": 0.1},
    {"name": "s3·1.5/1.2·m0.1",   "fwd": 1.5, "lat": 1.2, "stride": 3, "mom": 0.1},
    {"name": "s3·1.5/2.0·m0.1",   "fwd": 1.5, "lat": 2.0, "stride": 3, "mom": 0.1},

    # ── Forward speed sweep  (stride=3, lat=1.5, mom=0.1) ─────────────────
    {"name": "s3·1.0/1.5·m0.1",   "fwd": 1.0, "lat": 1.5, "stride": 3, "mom": 0.1},
    {"name": "s3·2.0/1.5·m0.1",   "fwd": 2.0, "lat": 1.5, "stride": 3, "mom": 0.1},

    # ── Asymmetric lat > fwd  (stride=3, mom=0.1) ──────────────────────────
    {"name": "s3·1.33/1.5·m0.1",  "fwd": 1.33, "lat": 1.5,  "stride": 3, "mom": 0.1},
    {"name": "s3·1.33/1.8·m0.1",  "fwd": 1.33, "lat": 1.8,  "stride": 3, "mom": 0.1},
    {"name": "s3·1.33/1.33·m0.1", "fwd": 1.33, "lat": 1.33, "stride": 3, "mom": 0.1},

    # ── Stride sweep  (fwd=1.5, lat=1.5, mom=0.1) ─────────────────────────
    {"name": "s2·1.5/1.5·m0.1",   "fwd": 1.5, "lat": 1.5, "stride": 2, "mom": 0.1},
    {"name": "s4·1.5/1.5·m0.1",   "fwd": 1.5, "lat": 1.5, "stride": 4, "mom": 0.1},
    {"name": "s5·1.5/1.5·m0.1",   "fwd": 1.5, "lat": 1.5, "stride": 5, "mom": 0.1},

    # ── 20 Hz + matched stride  (fwd=1.5, lat=1.5, mom=0.1) ──────────────
    {"name": "20hz·s8·1.5/1.5·m0.1", "fwd": 1.5, "lat": 1.5, "stride": 8, "mom": 0.1, "hz": 20},
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_best_model(base_dir):
    if not os.path.exists(base_dir):
        return None
    run_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    )
    if not run_dirs:
        return None
    ckpt_dir = os.path.join(base_dir, run_dirs[-1], "checkpoints")
    if os.path.exists(ckpt_dir):
        best = [f for f in os.listdir(ckpt_dir)
                if f.startswith("best_avoidance_v7_") and f.endswith(".zip")]
        if best:
            best.sort(key=lambda n: int(n.split("_win")[1].split("_")[0])
                      if "_win" in n else 0, reverse=True)
            return os.path.join(ckpt_dir, best[0])
    return None


def resolve_path(model_path):
    here = SCRIPT_DIR
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        candidate = os.path.normpath(os.path.join(here, "../..", model_path))
        if os.path.exists(candidate) or os.path.exists(candidate + ".zip"):
            model_path = candidate
    if not os.path.exists(model_path) and not model_path.endswith(".zip"):
        model_path += ".zip"
    return model_path


def make_env(cfg, depth_estimator, ros2_bridge):
    fwd    = cfg.get("fwd",    1.0)
    lat    = cfg.get("lat",    0.8)
    stride = cfg.get("stride", _D_STRIDE)
    mom    = cfg.get("mom",    _D_MOM)
    hz     = cfg.get("hz",     10)
    steps  = int(round(30 * hz / fwd))   # auto-scale to 30 m corridor

    # Capture values in default args to avoid closure issues
    def _factory(fwd=fwd, lat=lat, stride=stride, mom=mom, hz=hz, steps=steps):
        return Monitor(DroneAvoidanceEnv(
            depth_estimator=depth_estimator,
            ros2_bridge=ros2_bridge,
            fixed_speed=fwd,
            max_lateral=lat,
            step_hz=hz,
            max_steps=steps,
            frame_stride=stride,
            action_momentum=mom,
        ))

    return DummyVecEnv([_factory]), steps, hz


# ── Per-config run ────────────────────────────────────────────────────────────

def run_config(model, cfg, episodes, depth_estimator, ros2_bridge, stage):
    name = cfg["name"]
    fwd  = cfg.get("fwd",    1.0)
    lat  = cfg.get("lat",    0.8)
    stride = cfg.get("stride", _D_STRIDE)
    mom  = cfg.get("mom",    _D_MOM)
    hz   = cfg.get("hz",     10)

    env, max_steps, step_hz = make_env(cfg, depth_estimator, ros2_bridge)

    if stage > 0:
        DroneAvoidanceEnv._current_curriculum_stage = stage

    model.set_env(env)

    print(f"\n{'─'*64}")
    print(f"  {name}")
    print(f"  fwd={fwd} m/s  lat={lat} m/s  stride={stride}  "
          f"mom={mom}  hz={hz}  max_steps={max_steps}")
    print(f"{'─'*64}")

    wins = collisions = timeouts = 0
    ep_steps, ep_fwd, ep_lat, ep_rews = [], [], [], []

    obs = env.reset()

    for ep in range(episodes):
        done = False
        total_rew = 0.0
        steps = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_rew += float(reward[0])
            steps     += 1
            done       = bool(dones[0])
            last_info  = infos[0]

        survived = last_info.get("survived", False)
        collided = last_info.get("collided", False)
        fwd_p    = last_info.get("fwd_progress", 0.0)
        lat_p    = abs(last_info.get("lat_offset", 0.0))

        ep_steps.append(steps)
        ep_rews.append(total_rew)
        ep_fwd.append(fwd_p)
        ep_lat.append(lat_p)

        if survived:
            wins += 1; status = "WIN"
        elif collided:
            collisions += 1; status = "COLLISION"
        else:
            timeouts += 1; status = "TIMEOUT"

        stage_n = DroneAvoidanceEnv._current_curriculum_stage + 1
        print(f"  Ep {ep+1:>3}/{episodes}: {status:<9} | "
              f"Steps {steps:>5}  Fwd {fwd_p:>5.1f}m  "
              f"Lat {lat_p:>4.2f}m  Rew {total_rew:>6.1f}  S{stage_n}")

    env.close()

    result = {
        "name":       name,
        "cfg":        cfg,
        "episodes":   episodes,
        "wins":       wins,
        "collisions": collisions,
        "timeouts":   timeouts,
        "win_pct":    wins / episodes,
        "col_pct":    collisions / episodes,
        "avg_steps":  float(np.mean(ep_steps)),
        "avg_fwd":    float(np.mean(ep_fwd)),
        "avg_lat":    float(np.mean(ep_lat)),
        "avg_reward": float(np.mean(ep_rews)),
    }

    print(f"\n  → Win: {wins}/{episodes} ({100*wins/episodes:.0f}%)  "
          f"Col: {collisions}/{episodes}  "
          f"Fwd: {result['avg_fwd']:.1f}m  "
          f"|Lat|: {result['avg_lat']:.2f}m  "
          f"Reward: {result['avg_reward']:.1f}")

    return result


# ── Summary table ─────────────────────────────────────────────────────────────

def print_summary(results, episodes):
    ranked = sorted(results, key=lambda r: (r["win_pct"], r["avg_reward"]), reverse=True)
    NW = 24

    print(f"\n{'='*76}")
    print(f"SWEEP SUMMARY — {len(results)} configs × {episodes} episodes")
    print(f"{'='*76}")
    print(f"{'#':<3} {'Name':<{NW}} {'Win%':>5} {'Col%':>5} "
          f"{'Fwd':>6} {'|Lat|':>6} {'Steps':>6} {'Reward':>7}")
    print(f"{'─'*76}")

    for rank, r in enumerate(ranked, 1):
        tag = "  ◀ BEST" if rank == 1 else ""
        print(f"{rank:<3} {r['name']:<{NW}} "
              f"{100*r['win_pct']:>4.0f}%  "
              f"{100*r['col_pct']:>4.0f}%  "
              f"{r['avg_fwd']:>5.1f}m "
              f"{r['avg_lat']:>5.2f}m "
              f"{r['avg_steps']:>6.0f} "
              f"{r['avg_reward']:>7.1f}"
              f"{tag}")

    print(f"{'='*76}")
    best = ranked[0]
    cfg  = best["cfg"]
    print(f"\nBest: {best['name']}  —  "
          f"fwd={cfg.get('fwd',1.0)} lat={cfg.get('lat',0.8)} "
          f"stride={cfg.get('stride',_D_STRIDE)} "
          f"mom={cfg.get('mom',_D_MOM)} "
          f"hz={cfg.get('hz',10)}")
    print(f"      Win {100*best['win_pct']:.0f}%  "
          f"({best['wins']}/{best['episodes']} episodes)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    base_model_dir = os.path.normpath(os.path.join(SCRIPT_DIR, "../../models_v7"))

    parser = argparse.ArgumentParser(description="Parameter sweep — v7 avoidance")
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per configuration (default: 30)")
    parser.add_argument("--stage",    type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--ros2",     action="store_true")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = find_best_model(base_model_dir)
        if model_path is None:
            print("No model found. Train first or specify --model.")
            return
    else:
        model_path = resolve_path(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print("PPO-CNN v7 — Parameter Sweep")
    print("=" * 64)
    print(f"Model:    {model_path}")
    print(f"Device:   {device}")
    print(f"Stage:    {args.stage + 1}")
    print(f"Configs:  {len(CONFIGS)}")
    print(f"Episodes: {args.episodes} per config  "
          f"(~{len(CONFIGS) * args.episodes} total)")

    print("\nLoading Depth Anything V2 Small...")
    depth_estimator = DepthEstimator(target_size=(192, 144), device=device)
    print("Estimator ready.")

    ros2_bridge = None
    if args.ros2:
        from ros2_bridge import ROS2CameraBridge
        ros2_bridge = ROS2CameraBridge(target_size=(192, 144))
        ros2_bridge.start()
        print("ROS2 bridge started — waiting for first frame...")
        t0 = time.time()
        while not ros2_bridge.has_frame:
            if time.time() - t0 > 10.0:
                print("WARNING: no frames after 10s.")
                break
            time.sleep(0.1)
        if ros2_bridge.has_frame:
            print("ROS2 bridge ready.")

    if args.stage > 0:
        DroneAvoidanceEnv._current_curriculum_stage = args.stage

    # Load model once with a temporary env (needed to set obs/action spaces)
    print("\nLoading model...")
    tmp_env, _, _ = make_env(CONFIGS[0], depth_estimator, ros2_bridge)
    model = PPO.load(
        model_path, env=tmp_env, device=device,
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
    )
    tmp_env.close()
    print("Model loaded.")

    input("\nPress ENTER to start sweep...")
    print()

    results = []
    t_start = time.time()

    for i, cfg in enumerate(CONFIGS):
        print(f"\n[{i+1}/{len(CONFIGS)}]", end="")
        result = run_config(model, cfg, args.episodes,
                            depth_estimator, ros2_bridge, args.stage)
        results.append(result)

        elapsed = time.time() - t_start
        done_n  = i + 1
        left_n  = len(CONFIGS) - done_n
        if left_n > 0:
            eta = (elapsed / done_n) * left_n
            print(f"\n  Elapsed: {elapsed/60:.1f} min  "
                  f"ETA: {eta/60:.1f} min  "
                  f"({left_n} configs remaining)")

    print_summary(results, args.episodes)

    if ros2_bridge is not None:
        ros2_bridge.stop()


if __name__ == "__main__":
    main()
