"""
Sweep test launcher — v7 GPS-free avoidance.

Steps:
  1. Waits for AirSim to accept connections (port 41451)
  2. Starts the ROS2 airsim_node
  3. Waits for the camera topic
  4. Runs sweep_test.py

Usage:
    python3 launch_sweep_test.py
    python3 launch_sweep_test.py --model path/to.zip
    python3 launch_sweep_test.py --episodes 50
    python3 launch_sweep_test.py --stage 2
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
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))

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


def run_bg(cmd):
    p = subprocess.Popen(
        cmd, shell=True, executable="/bin/bash",
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
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
            shell=True, capture_output=True, text=True, executable="/bin/bash",
        )
        if topic in result.stdout:
            print(" ready!")
            return True
        print(".", end="", flush=True)
        time.sleep(2)
    print(" TIMED OUT")
    return False


def main():
    parser = argparse.ArgumentParser(description="Launch v7 parameter sweep")
    parser.add_argument("--model",    type=str, default=None)
    parser.add_argument("--episodes", type=int, default=30,
                        help="Episodes per configuration (default: 30)")
    parser.add_argument("--stage",    type=int, default=0, choices=[0, 1, 2])
    args = parser.parse_args()

    print("=" * 60)
    print("PPO-CNN Drone Navigation v7 — Parameter Sweep")
    print("=" * 60)
    print()

    if not wait_for_airsim(timeout=120):
        print("[LAUNCHER] AirSim not reachable. Is UE5 running with Play active?")
        sys.exit(1)

    time.sleep(3)

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

    cmd = f"{PYTHON} {os.path.join(SCRIPT_DIR, 'sweep_test.py')} --ros2"
    if args.model:
        cmd += f" --model {args.model}"
    if args.episodes != 30:
        cmd += f" --episodes {args.episodes}"
    if args.stage > 0:
        cmd += f" --stage {args.stage}"

    print(f"[LAUNCHER] Starting sweep: {cmd}")
    print("[LAUNCHER] Ctrl+C stops everything.\n")

    p = subprocess.Popen(cmd, shell=True, executable="/bin/bash", cwd=SCRIPT_DIR)
    procs.append(p)
    p.wait()
    cleanup()


if __name__ == "__main__":
    main()
