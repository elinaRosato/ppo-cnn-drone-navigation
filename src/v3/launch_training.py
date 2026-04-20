"""
Training launcher — v3 (monocular depth estimation).

Steps:
  1. Waits for AirSim to accept connections (port 41451)
  2. Starts the ROS2 airsim_node
  3. Waits for the camera topic to appear
  4. Starts TensorBoard (http://localhost:6006)
  5. Starts training (or resume)

Usage:
    python3 launch_training.py                     # new training
    python3 launch_training.py --resume            # resume latest checkpoint
    python3 launch_training.py --steps 2000000     # custom step count
    python3 launch_training.py --no-ros2           # use AirSim API directly

Prerequisites:
  1. Open UE5 and hit Play — AirSim must be running.
  2. (WSL2 only) export AIRSIM_HOST=$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time

ROS2_SETUP   = "/opt/ros/jazzy/setup.bash"
AIRSIM_SETUP = os.path.expanduser("~/Cosys-AirSim/ros2/install/setup.bash")

CAMERA_TOPIC = "/airsim_node/SimpleFlight/front_center_Scene/image"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Use venv if present in the project root
_VENV_PYTHON = os.path.normpath(os.path.join(SCRIPT_DIR, "../../venv/bin/python3"))
PYTHON = _VENV_PYTHON if os.path.exists(_VENV_PYTHON) else "python3"

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


def wait_for_topic(topic, timeout=60):
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
    parser = argparse.ArgumentParser(description="Launch v3 training pipeline")
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--steps",       type=int, default=None)
    parser.add_argument("--ros2",        action="store_true", default=True)
    parser.add_argument("--no-ros2",     dest="ros2", action="store_false")
    parser.add_argument("--density-stage", type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    print("=" * 60)
    print("PPO-CNN Drone Navigation v3 — Training Launcher")
    print("=" * 60)
    print()
    print("Step 1: Open UE5 and hit Play (AirSim must be running).")
    print()

    if not wait_for_airsim(timeout=120):
        print("[LAUNCHER] AirSim not reachable. Is UE5 running with Play active?")
        sys.exit(1)

    time.sleep(3)

    if args.ros2:
        host = os.environ.get("AIRSIM_HOST", "localhost")
        print("[LAUNCHER] Starting ROS2 airsim_node...")
        run_bg(
            f"source {ROS2_SETUP} && "
            f"source {AIRSIM_SETUP} && "
            f"ros2 launch airsim_ros_pkgs airsim_node.launch.py host:={host}"
        )

        if not wait_for_topic(CAMERA_TOPIC, timeout=60):
            print("[LAUNCHER] Camera topic not available. Check airsim_node logs.")
            cleanup()

        print("[LAUNCHER] Waiting 2s for topic to stabilise...")
        time.sleep(2)

    print("[LAUNCHER] Starting TensorBoard...")
    run_bg(f"tensorboard --logdir={os.path.join(SCRIPT_DIR, '../../logs_v3')} --port=6006")
    time.sleep(2)
    print("[LAUNCHER] TensorBoard → http://localhost:6006")
    print()

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
