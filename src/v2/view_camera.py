"""
Live camera feed viewer — shows what the model actually sees during training.

Displays the grayscale RGB feed (same source as the model observation) plus,
optionally, the 4-frame stack side-by-side for debugging temporal context.

Usage:
    python view_camera.py
    python view_camera.py --fps 10
    python view_camera.py --stack     # show all 4 stacked frames

Press Q to quit.
"""

import argparse
import time
import airsim
import numpy as np
import cv2

STACK_FRAMES = 4
IMG_SIZE = 84


def main(fps=5, show_stack=False):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected to AirSim!")
    print(f"Showing grayscale RGB feed at ~{fps} FPS. Press Q to quit.")

    delay = 1.0 / fps
    frame_stack = np.zeros((STACK_FRAMES, IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    while True:
        responses = client.simGetImages([
            airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)
        ])

        if responses and responses[0].width > 0:
            r = responses[0]
            raw = np.frombuffer(r.image_data_uint8, dtype=np.uint8)
            n_ch = len(raw) // (r.width * r.height)
            img = raw.reshape(r.height, r.width, n_ch)
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY if n_ch == 4 else cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))

            # Update stack
            frame_stack = np.roll(frame_stack, shift=-1, axis=0)
            frame_stack[-1] = gray

            if show_stack:
                # Show all 4 frames side-by-side (upscaled for visibility)
                frames = [cv2.resize(frame_stack[i], (200, 200)) for i in range(STACK_FRAMES)]
                display = np.concatenate(frames, axis=1)
                cv2.imshow("Frame Stack (oldest → newest)", display)
            else:
                cv2.imshow("Drone Camera (grayscale RGB)", cv2.resize(gray, (400, 400)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(delay)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Live camera feed viewer')
    parser.add_argument('--fps', type=int, default=5,
                        help='Target frames per second (default: 5)')
    parser.add_argument('--stack', action='store_true',
                        help='Show all 4 stacked frames side-by-side')
    args = parser.parse_args()

    main(fps=args.fps, show_stack=args.stack)
