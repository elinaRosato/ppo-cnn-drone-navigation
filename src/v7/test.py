"""
Test trained GPS-free obstacle avoidance model — v7.

Loads the best saved model (or a specific checkpoint) and runs evaluation episodes.
No gradient updates — pure inference with deterministic policy.

Usage:
    python test.py                             # best model, 10 episodes
    python test.py --episodes 20               # more episodes
    python test.py --model path/to/model.zip   # specific checkpoint
    python test.py --stage 2                   # test at dense curriculum stage
    python test.py --fast 2                    # 2× speed / 20 Hz
    python test.py --ros2                      # use ROS2 bridge for camera images
"""

import argparse
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from avoidance_env import DroneAvoidanceEnv
from train import AsymmetricActorCriticPolicy


def find_best_model(base_dir):
    """Return path to highest win-rate best_* model in the latest run."""
    if not os.path.exists(base_dir):
        return None

    run_dirs = sorted(
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run_")
    )
    if not run_dirs:
        return None

    latest_run = os.path.join(base_dir, run_dirs[-1])
    ckpt_dir   = os.path.join(latest_run, "checkpoints")

    if os.path.exists(ckpt_dir):
        best = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("best_avoidance_v7_") and f.endswith(".zip")
        ]
        if best:
            def _win(name):
                try:
                    return int(name.split("_win")[1].split("_")[0])
                except (IndexError, ValueError):
                    return 0
            best.sort(key=_win, reverse=True)
            return os.path.join(ckpt_dir, best[0])

        zips = [
            f for f in os.listdir(ckpt_dir)
            if f.startswith("avoidance_v7_") and f.endswith(".zip")
        ]
        if zips:
            zips.sort(key=lambda x: int(x.split("_")[-2]))
            return os.path.join(ckpt_dir, zips[-1])

    for name in ("avoidance_v7_final.zip", "avoidance_v7_interrupted.zip"):
        path = os.path.join(latest_run, name)
        if os.path.exists(path):
            return path

    return None


def test(model_path=None, episodes=10, use_ros2=False, speed_scale=1, stage=0,
         fwd_speed=1.0, lat_speed=0.8, frame_stride=None, action_momentum=None,
         hz=None, max_steps=None):
    _here          = os.path.dirname(os.path.abspath(__file__))
    base_model_dir = os.path.join(_here, "../../models_v7")

    if model_path is None:
        model_path = find_best_model(base_model_dir)
        if model_path is None:
            print("No model found in models_v7/. Train first or specify --model.")
            return

    # Resolve relative paths — try CWD first, then repo root (2 levels up from src/v7/)
    if not os.path.isabs(model_path) and not os.path.exists(model_path):
        repo_root = os.path.normpath(os.path.join(_here, "../.."))
        candidate = os.path.join(repo_root, model_path)
        if os.path.exists(candidate) or os.path.exists(candidate + ".zip"):
            model_path = candidate
    if not os.path.exists(model_path) and not model_path.endswith(".zip"):
        model_path += ".zip"

    step_hz = hz if hz is not None else 10 * speed_scale
    # Auto-scale max_steps to preserve the same physical corridor distance (30m)
    if max_steps is None:
        max_steps = int(round(30 * step_hz / fwd_speed))

    print("=" * 70)
    print("GPS-FREE OBSTACLE AVOIDANCE — v7 TEST")
    print("=" * 70)
    print(f"Model:    {model_path}")
    print(f"Episodes: {episodes}")
    if hz is not None:
        print(f"Hz:       {step_hz} Hz  (speed unchanged: {fwd_speed} m/s fwd)")
    elif speed_scale > 1:
        print(f"Speed:    {speed_scale}× / {step_hz} Hz  (deploy at 1× / 10 Hz)")
    else:
        print(f"Speed:    1× / 10 Hz  (deployment speed)")
    from avoidance_env import FRAME_STRIDE as _DEFAULT_STRIDE, ACTION_MOMENTUM as _DEFAULT_MOM
    _stride = frame_stride    if frame_stride    is not None else _DEFAULT_STRIDE
    _mom    = action_momentum if action_momentum is not None else _DEFAULT_MOM
    print(f"Fwd:      {fwd_speed} m/s  Lat: {lat_speed} m/s  FrameStride: {_stride}  Momentum: {_mom}")
    print(f"MaxSteps: {max_steps}  ({max_steps / step_hz:.0f}s timeout @ {step_hz}Hz)")
    print(f"Images:   {'ROS2 bridge' if use_ros2 else 'AirSim Python API'}")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    from depth_estimator import DepthEstimator
    print("\nLoading Depth Anything V2 Small...")
    depth_estimator = DepthEstimator(target_size=(192, 144), device=device)

    ros2_bridge = None
    if use_ros2:
        from ros2_bridge import ROS2CameraBridge
        ros2_bridge = ROS2CameraBridge(target_size=(192, 144))
        ros2_bridge.start()
        print("ROS2 bridge started — waiting for first frames...")
        t0 = time.time()
        while not ros2_bridge.has_frame:
            if time.time() - t0 > 10.0:
                print("WARNING: ROS2 bridge received no frames after 10s.")
                break
            time.sleep(0.1)
        if ros2_bridge.has_frame:
            print("ROS2 bridge ready.")

    if stage > 0:
        DroneAvoidanceEnv._current_curriculum_stage = stage
        print(f"Curriculum stage forced to: {stage + 1}")

    print("\nCreating environment...")
    env = DummyVecEnv([lambda: Monitor(
        DroneAvoidanceEnv(
            ros2_bridge=ros2_bridge,
            depth_estimator=depth_estimator,
            fixed_speed=fwd_speed,
            max_lateral=lat_speed,
            speed_scale=speed_scale,
            step_hz=step_hz,
            max_steps=max_steps,
            frame_stride=frame_stride,
            action_momentum=action_momentum,
        )
    )])

    print("Loading model...")
    model = PPO.load(
        model_path,
        env=env,
        device=device,
        custom_objects={"policy_class": AsymmetricActorCriticPolicy},
    )
    print("Model loaded.\n")

    input("Press ENTER to start testing...")
    print()

    wins       = 0
    collisions = 0
    timeouts   = 0
    ep_steps   = []
    ep_fwd     = []
    ep_lat     = []
    ep_rewards = []

    # Single reset before the loop — DummyVecEnv auto-resets on done=True,
    # so calling env.reset() inside the loop would cause a double takeoff.
    obs = env.reset()

    for ep in range(episodes):
        done      = False
        total_rew = 0.0
        steps     = 0
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = env.step(action)
            total_rew += float(reward[0])
            steps     += 1
            done       = bool(dones[0])
            last_info  = infos[0]
        # obs is now the auto-reset observation for the next episode

        survived = last_info.get("survived", False)
        collided = last_info.get("collided", False)
        fwd      = last_info.get("fwd_progress", 0.0)
        lat      = abs(last_info.get("lat_offset", 0.0))

        ep_steps.append(steps)
        ep_rewards.append(total_rew)
        ep_fwd.append(fwd)
        ep_lat.append(lat)

        if survived:
            wins += 1
            status = "WIN"
        elif collided:
            collisions += 1
            status = "COLLISION"
        else:
            timeouts += 1
            status = "TIMEOUT"

        stage_n = DroneAvoidanceEnv._current_curriculum_stage + 1
        print(
            f"  Ep {ep + 1:>3}/{episodes}: {status:<9} | "
            f"Steps: {steps:>4}  Fwd: {fwd:>5.1f} m  Lat: {lat:>4.2f} m  "
            f"Reward: {total_rew:>6.1f}  S{stage_n}"
        )

    print(f"\n{'=' * 70}")
    print("TEST SUMMARY")
    print(f"{'=' * 70}")
    print(f"Episodes:   {episodes}")
    print(f"Win:        {wins}/{episodes}  ({100 * wins / episodes:.0f}%)")
    print(f"Collision:  {collisions}/{episodes}  ({100 * collisions / episodes:.0f}%)")
    print(f"Timeout:    {timeouts}/{episodes}  ({100 * timeouts / episodes:.0f}%)")
    print(f"Avg steps:  {np.mean(ep_steps):.0f}")
    print(f"Avg fwd:    {np.mean(ep_fwd):.1f} m")
    print(f"Avg |lat|:  {np.mean(ep_lat):.2f} m")
    print(f"Avg reward: {np.mean(ep_rewards):.1f}")
    print(f"{'=' * 70}")

    env.close()
    if ros2_bridge is not None:
        ros2_bridge.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPS-free avoidance model — v7")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to model .zip (default: highest win-rate model in latest run)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes (default: 10)")
    parser.add_argument("--ros2",     action="store_true",
                        help="Use ROS2 bridge for camera images (match training setup)")
    parser.add_argument("--stage",    type=int, default=0, choices=[0, 1, 2],
                        help="Curriculum stage to test at (0=sparse, 1=medium, 2=dense)")
    parser.add_argument("--fast",      type=int,   default=1, choices=[1, 2, 4],
                        help="Speed multiplier: 2=2×/20Hz, 4=4×/40Hz (1=deploy speed/10Hz)")
    parser.add_argument("--fwd-speed",    type=float, default=1.0,
                        help="Forward speed in m/s (default: 1.0)")
    parser.add_argument("--lat-speed",    type=float, default=0.8,
                        help="Max lateral speed in m/s (default: 0.8)")
    parser.add_argument("--frame-stride",    type=int,   default=None,
                        help="Temporal stride between stacked frames (default: 4, trained value)")
    parser.add_argument("--action-momentum", type=float, default=None,
                        help="Lateral action smoothing coefficient (default: 0.3, trained value)")
    parser.add_argument("--hz",              type=int,   default=None,
                        help="Control loop frequency in Hz without changing speed (e.g. 40)")
    parser.add_argument("--max-steps",       type=int,   default=None,
                        help="Episode step limit (default: auto-scaled to 30m corridor)")
    args = parser.parse_args()

    test(
        model_path  = args.model,
        episodes    = args.episodes,
        use_ros2    = args.ros2,
        speed_scale = args.fast,
        stage       = args.stage,
        fwd_speed       = args.fwd_speed,
        lat_speed       = args.lat_speed,
        frame_stride    = args.frame_stride,
        action_momentum = args.action_momentum,
        hz              = args.hz,
        max_steps       = args.max_steps,
    )
