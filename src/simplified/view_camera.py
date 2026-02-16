"""
Live camera feed viewer.
Run alongside training to see what the drone's camera sees.

Usage:
    python view_camera.py
    python view_camera.py --fps 10

Press Q to quit.
"""

import argparse
import time
import airsim
import numpy as np
import cv2


def main(fps=5):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
    print(f"Showing depth camera feed at ~{fps} FPS. Press Q to quit.")

    delay = 1.0 / fps

    while True:
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.DepthPerspective, True, False)
        ])

        if responses and responses[0].width > 0:
            depth = airsim.list_to_2d_float_array(responses[0].image_data_float,
                                                  responses[0].width, responses[0].height)
            depth = np.clip(depth, 0, 100)
            depth_vis = (depth / 100.0 * 255).astype(np.uint8)
            cv2.imshow("Drone Depth Camera", depth_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(delay)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live camera feed viewer')
    parser.add_argument('--fps', type=int, default=5,
                        help='Target frames per second (default: 5)')
    args = parser.parse_args()

    main(fps=args.fps)
