"""
View live camera feed from the drone
Run this while AirSim is running to see what the drone's camera sees
"""

import airsim
import cv2
import numpy as np
import time

# Connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
print("Connected to AirSim!")

camera_name = "front_center"
print(f"Showing live feed from '{camera_name}' camera")
print("Press 'q' to quit")

while True:
    # Get camera image
    responses = client.simGetImages([
        airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)
    ])

    if responses and len(responses[0].image_data_uint8) > 0:
        # Convert to numpy array
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(responses[0].height, responses[0].width, 3)

        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # Show what the RL agent sees (84x84 downsampled version)
        img_small = cv2.resize(img_rgb, (84, 84))
        img_small_bgr = cv2.cvtColor(img_small, cv2.COLOR_RGB2BGR)

        # Stack images side by side
        img_small_upscaled = cv2.resize(img_small_bgr, (img_bgr.shape[1], img_bgr.shape[0]))
        combined = np.hstack([img_bgr, img_small_upscaled])

        # Add labels
        cv2.putText(combined, "Full Resolution", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "What RL Agent Sees (84x84)",
                   (img_bgr.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display
        cv2.imshow("Drone Camera Feed", combined)

    else:
        print("No image data received")

    # Check for quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Small delay to not overwhelm the system
    time.sleep(0.033)  # ~30 FPS

cv2.destroyAllWindows()
print("Camera viewer closed")
