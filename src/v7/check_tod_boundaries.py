"""
Check time-of-day boundaries used in training.

Sets AirSim to the lower bound (17:00 UTC = 10:00 PDT),
waits for you to confirm, then sets the upper bound (22:59 UTC = 15:59 PDT).

Usage:
    python3 check_tod_boundaries.py

AirSim must be running (UE5 Play active).
"""

import sys
import time
import cosysairsim as airsim

TIMES = [
    ("lower bound", "2025-06-15 17:00:00", "10:00 PDT — earliest training time"),
    ("upper bound", "2025-06-15 22:59:00", "15:59 PDT — latest training time"),
]


def set_tod(client, datetime_str, label, description):
    client.simSetTimeOfDay(
        True,
        start_datetime=datetime_str,
        is_start_datetime_dst=False,
        celestial_clock_speed=1,
        update_interval_secs=0.1,
        move_sun=True,
    )
    print(f"\n  Set to {label}: {datetime_str} UTC  ({description})")
    print("  Check the UE5 viewport — it should be daytime.")


def main():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected.\n")

    for i, (label, dt_str, description) in enumerate(TIMES):
        set_tod(client, dt_str, label, description)

        if i < len(TIMES) - 1:
            input("\n  Press Enter to advance to next boundary...")

    print("\nDone. Both boundaries checked.")
    print("To disable time-of-day control, restart AirSim (stop/play UE5).")


if __name__ == "__main__":
    main()
