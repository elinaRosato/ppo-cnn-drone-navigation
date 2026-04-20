"""
Training launcher — automates the setup steps after you hit Play in UE5.

Steps:
  1. Waits for AirSim to accept connections (port 41451)
  2. Starts the ROS2 airsim_node
  3. Waits for the camera topic to appear
  4. Starts TensorBoard (http://localhost:6006)
  5. Starts training (or resume / tune)

This script manages all subprocesses and shuts them down cleanly on Ctrl+C.

Usage:
    python3 launch_training.py                      # new training
    python3 launch_training.py --resume             # resume latest checkpoint
    python3 launch_training.py --steps 500000       # new training, custom step count
    python3 launch_training.py --tune               # hyperparameter tuning
    python3 launch_training.py --tune --trials 50   # tuning with 50 trials

Prerequisites (must be done before running):
  1. Open UE5 and hit Play — AirSim needs to be running first.
  2. (WSL2 only) export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

# ── ROS2 paths — adjust if your installation differs ─────────────────────────
ROS2_SETUP   = "/opt/ros/jazzy/setup.bash"
AIRSIM_SETUP = os.path.expanduser("~/Cosys-AirSim/ros2/install/setup.bash")

# ── Python interpreter — use venv if present ─────────────────────────────────
_VENV_PYTHON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../venv/bin/python3")
PYTHON = os.path.normpath(_VENV_PYTHON) if os.path.exists(os.path.normpath(_VENV_PYTHON)) else "python3"

# Camera topic to wait for before starting training
CAMERA_TOPIC = "/airsim_node/SimpleFlight/front_center_Scene/image"

# ── Script directory (so relative imports work from any cwd) ─────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

procs = []


def cleanup(sig=None, frame=None):
    print("\n[LAUNCHER] Shutting down all processes...")
    for p in reversed(procs):
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception:
            pass
    sys.exit(0)


signal.signal(signal.SIGINT,  cleanup)
signal.signal(signal.SIGTERM, cleanup)


def run_bg(cmd, env=None):
    p = subprocess.Popen(
        cmd, shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    procs.append(p)
    return p


def wait_for_airsim(timeout=120):
    host = os.environ.get("AIRSIM_HOST", "localhost")
    print(f"[LAUNCHER] Waiting for AirSim at {host}:41451", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, 41451), timeout=1)
            s.close()
            print(" ready!")
            return True
        except OSError:
            print(".", end="", flush=True)
            time.sleep(2)
    print(" TIMED OUT")
    return False


def wait_for_topic(topic, timeout=30):
    source = f"source {ROS2_SETUP} && source {AIRSIM_SETUP}"
    print(f"[LAUNCHER] Waiting for ROS2 topic {topic}", end="", flush=True)
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = subprocess.run(
            f"{source} && ros2 topic list 2>/dev/null",
            shell=True, capture_output=True, text=True,
            executable="/bin/bash",
        )
        if topic in result.stdout:
            print(" ready!")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMED OUT")
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch AirSim training pipeline")
    parser.add_argument("--resume",     action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--steps",      type=int, default=None, help="Total training steps")
    parser.add_argument("--ros2",       action="store_true", default=True,
                        help="Use ROS2 bridge for images (default: True)")
    parser.add_argument("--no-ros2",    dest="ros2", action="store_false",
                        help="Disable ROS2 bridge, use AirSim API directly")
    parser.add_argument("--density-stage", type=int, default=0, choices=[0, 1, 2])
    parser.add_argument("--tune",       action="store_true", help="Run hyperparameter tuning")
    parser.add_argument("--trials",     type=int, default=30, help="Optuna trials (tune only)")
    parser.add_argument("--trial-steps",type=int, default=200_000, help="Steps per trial (tune only)")
    args = parser.parse_args()

    print("=" * 60)
    print("PPO-CNN Drone Navigation — Training Launcher")
    print("=" * 60)
    print()
    print("Step 1: Open UE5 and hit Play (AirSim must be running).")
    print()

    # ── Wait for AirSim ───────────────────────────────────────────────────────
    if not wait_for_airsim(timeout=120):
        print("[LAUNCHER] AirSim not reachable. Is UE5 running with Play active?")
        sys.exit(1)

    time.sleep(3)   # let AirSim finish initialising

    # ── Start ROS2 airsim_node (only needed when using ROS2 bridge) ───────────
    if args.ros2:
        host = os.environ.get("AIRSIM_HOST", "localhost")
        print("[LAUNCHER] Starting ROS2 airsim_node...")
        run_bg(
            f"source {ROS2_SETUP} && "
            f"source {AIRSIM_SETUP} && "
            f"ros2 launch airsim_ros_pkgs airsim_node.launch.py host:={host}"
        )

        if not wait_for_topic(CAMERA_TOPIC, timeout=30):
            print("[LAUNCHER] Camera topic not available. Check airsim_node logs.")
            cleanup()

        print("[LAUNCHER] Waiting 2s for topic to stabilise...")
        time.sleep(2)

    # ── Start TensorBoard ─────────────────────────────────────────────────────
    print("[LAUNCHER] Starting TensorBoard...")
    run_bg(f"tensorboard --logdir={os.path.join(SCRIPT_DIR, '../../logs_v2')} --port=6006")
    time.sleep(2)
    print("[LAUNCHER] TensorBoard → http://localhost:6006")
    print()

    # ── Build training command ────────────────────────────────────────────────
    if args.tune:
        cmd = f"{PYTHON} {os.path.join(SCRIPT_DIR, 'tune.py')} --trials {args.trials}"
        cmd += f" --trial-steps {args.trial_steps}"
        if args.ros2:
            cmd += " --ros2"
        print(f"[LAUNCHER] Starting tuning: {cmd}")
    else:
        cmd = f"{PYTHON} {os.path.join(SCRIPT_DIR, 'train.py')}"
        if args.resume:
            cmd += " --resume"
        if args.steps:
            cmd += f" --steps {args.steps}"
        if args.ros2:
            cmd += " --ros2"
        if args.density_stage > 0:
            cmd += f" --density-stage {args.density_stage}"
        print(f"[LAUNCHER] Starting training: {cmd}")

    print("[LAUNCHER] Ctrl+C stops everything.")
    print()

    training = subprocess.Popen(
        cmd, shell=True,
        executable="/bin/bash",
        cwd=SCRIPT_DIR,
    )
    procs.append(training)
    training.wait()
    cleanup()


if __name__ == "__main__":
    main()
