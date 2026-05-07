"""
Test launcher — v7 GPS-free avoidance.

Steps:
  1. Waits for AirSim to accept connections (port 41451)
  2. Starts the ROS2 airsim_node
  3. Waits for the camera topic to appear
  4. Starts testing

Usage:
    python3 launch_test.py                       # best model, 10 episodes
    python3 launch_test.py --model path/to.zip   # specific checkpoint
    python3 launch_test.py --episodes 20
    python3 launch_test.py --fast 2              # 2× speed / 20 Hz
    python3 launch_test.py --stage 2             # test at dense stage

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
        cmd, shell=True,
        executable="/bin/bash",
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
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
    parser = argparse.ArgumentParser(description="Launch v7 test pipeline")
    parser.add_argument("--model",    type=str, default=None,
                        help="Path to model .zip (default: highest win-rate model in latest run)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of test episodes (default: 10)")
    parser.add_argument("--stage",    type=int, default=0, choices=[0, 1, 2],
                        help="Curriculum stage to test at (0=sparse, 1=medium, 2=dense)")
    parser.add_argument("--fast",      type=int,   default=1, choices=[1, 2, 4],
                        help="Speed multiplier: 2=2×/20Hz, 4=4×/40Hz")
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
    parser.add_argument("--ros2",      action="store_true",
                        help="(accepted for consistency — launcher always uses ROS2)")
    args = parser.parse_args()

    print("=" * 60)
    print("PPO-CNN Drone Navigation v7 — GPS-Free Avoidance Test")
    print("=" * 60)
    print()
    print("Step 1: Open UE5 and hit Play (AirSim must be running).")
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

    cmd = f"{PYTHON} {os.path.join(SCRIPT_DIR, 'test.py')} --ros2"
    if args.model:
        cmd += f" --model {args.model}"
    if args.episodes != 10:
        cmd += f" --episodes {args.episodes}"
    if args.stage > 0:
        cmd += f" --stage {args.stage}"
    if args.fast > 1:
        cmd += f" --fast {args.fast}"
    if args.fwd_speed != 1.0:
        cmd += f" --fwd-speed {args.fwd_speed}"
    if args.lat_speed != 0.8:
        cmd += f" --lat-speed {args.lat_speed}"
    if args.frame_stride is not None:
        cmd += f" --frame-stride {args.frame_stride}"
    if args.action_momentum is not None:
        cmd += f" --action-momentum {args.action_momentum}"
    if args.hz is not None:
        cmd += f" --hz {args.hz}"
    if args.max_steps is not None:
        cmd += f" --max-steps {args.max_steps}"

    print(f"[LAUNCHER] Starting test: {cmd}")
    print("[LAUNCHER] Ctrl+C stops everything.")
    print()

    test_proc = subprocess.Popen(
        cmd, shell=True,
        executable="/bin/bash",
        cwd=SCRIPT_DIR,
    )
    procs.append(test_proc)
    test_proc.wait()
    cleanup()


if __name__ == "__main__":
    main()
