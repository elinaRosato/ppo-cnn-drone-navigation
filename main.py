"""
Basic AirSim drone control demonstration
"""

import airsim
import time
import numpy as np


def main():
    print("Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("Connected!")

    client.enableApiControl(True)
    print("API control enabled")

    client.armDisarm(True)
    print("Drone armed")

    print("Taking off...")
    client.takeoffAsync().join()
    print("Takeoff complete")

    print("Hovering...")
    time.sleep(2)

    print("Moving forward...")
    client.moveToPositionAsync(10, 0, -5, 5).join()
    time.sleep(1)

    print("Moving right...")
    client.moveToPositionAsync(10, 5, -5, 5).join()
    time.sleep(1)

    print("Returning to start...")
    client.moveToPositionAsync(0, 0, -5, 5).join()
    time.sleep(1)

    print("Capturing images...")
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
        airsim.ImageRequest("front_center", airsim.ImageType.DepthPlanar, True)
    ])

    if responses:
        print(f"Captured {len(responses)} images")
        img_rgb = responses[0]
        if img_rgb.height > 0 and img_rgb.width > 0:
            img1d = np.frombuffer(img_rgb.image_data_uint8, dtype=np.uint8)
            img_rgb_array = img1d.reshape(img_rgb.height, img_rgb.width, 3)
            print(f"RGB Image shape: {img_rgb_array.shape}")

    print("Rotating 360 degrees...")
    client.rotateToYawAsync(90, 2).join()
    time.sleep(0.5)
    client.rotateToYawAsync(180, 2).join()
    time.sleep(0.5)
    client.rotateToYawAsync(270, 2).join()
    time.sleep(0.5)
    client.rotateToYawAsync(0, 2).join()
    time.sleep(0.5)

    state = client.getMultirotorState()
    print(f"\nDrone State:")
    print(f"  Position: {state.kinematics_estimated.position}")
    print(f"  Velocity: {state.kinematics_estimated.linear_velocity}")
    print(f"  Orientation: {state.kinematics_estimated.orientation}")

    print("\nLanding...")
    client.landAsync().join()
    print("Landed")

    client.armDisarm(False)
    print("Drone disarmed")

    client.enableApiControl(False)
    print("API control disabled")

    print("\nMission complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nScript interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        try:
            client = airsim.MultirotorClient()
            client.landAsync().join()
            client.armDisarm(False)
            client.enableApiControl(False)
        except:
            pass
