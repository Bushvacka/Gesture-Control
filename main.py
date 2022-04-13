# Data manipulation
import math
from cv2 import FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_PLAIN
import numpy as np
# Image processing
import cv2
# Hand Tracking
import handDetector
# Volume Control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
# Brightness Control
import screen_brightness_control as sbc

INTERNAL_DISPLAY = 0
EXIT_KEY = 27
GREEN_COLOR = (0, 224, 0)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
BLUE_COLOR = (237, 162, 0)
MIN_DIST = 15
MAX_DIST = 115

def main():    
    # Initialize volume controller
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_control = cast(interface, POINTER(IAudioEndpointVolume))

    # Set initial volume to previous system volume
    volume = volume_control.GetMasterVolumeLevelScalar()*100

    # Set initial brightness to previous system brightness
    brightness = sbc.get_brightness()[INTERNAL_DISPLAY]

    # Initialize hand detection module
    detector = handDetector.handDetector(detection_confidence=.6, tracking_confidence=.6, model_complexity=1)

    # Initialize video feed
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while (cap.isOpened()):
        success, img = cap.read()

        if not success:
            print("Frame not loaded")
            break
        
        # Flip the image horizontally
        img = cv2.flip(img, 1)

        # Get hand landmarks
        results = detector.detectHands(img, True)

        if results.multi_hand_landmarks:
            #Iterate through all detected hands
            for i, hand_lmks in enumerate(results.multi_hand_landmarks):
                thumb = detector.getLandmarkPos(img, hand_lmks, "THUMB_TIP")
                index = detector.getLandmarkPos(img, hand_lmks, "INDEX_FINGER_TIP")
                cv2.line(img, thumb, index, BLACK_COLOR, 3)
                # multi_handedness -> {'classification': [{'index': 1, 'score': 0.92601216, 'label': 'Right'}]}
                handedness = results.multi_handedness[i].classification[0].label
                if (handedness == "Right"):
                    if thumb and index:
                        old_volume = volume # Save previous volume for noise filtering
                        # Calculate new volume based on thumb-index distance
                        dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
                        volume = np.interp(dist, [MIN_DIST, MAX_DIST], [0, 100])
                        
                        # Update volume if a significant change has occured
                        if (abs(volume - old_volume) > 1):
                            volume_control.SetMasterVolumeLevelScalar(volume/100.0, None)
                        else:
                            volume = old_volume
                else:
                    if thumb and index:
                        old_brightness = brightness # Save previous brightness for noise filtering
                        # Calculate new brightness based on thumb-index distance
                        dist = math.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
                        brightness = np.interp(dist, [MIN_DIST, MAX_DIST], [0, 100])

                        # Update brightness if a significant change has occured
                        if (abs(brightness - old_brightness) > 1):
                            sbc.set_brightness(brightness, INTERNAL_DISPLAY)
                        else:
                            brightness = old_brightness
        else:
            # Reflect any changes made by the system
            volume = volume_control.GetMasterVolumeLevelScalar() * 100
            brightness = sbc.get_brightness()[INTERNAL_DISPLAY]
        
        # Brightness
        cv2.rectangle(img, (10, 10), (50, 110), BLACK_COLOR, 2)
        cv2.rectangle(img, (11, int(110 - brightness)), (49, 109), BLUE_COLOR, cv2.FILLED)
        cv2.putText(img, "Light", (10, 130), FONT_HERSHEY_PLAIN, 1, BLUE_COLOR, 2)
        # Volume
        cv2.rectangle(img, (60, 10), (110, 110), BLACK_COLOR, 2)
        cv2.rectangle(img, (61, int(110 - volume)), (109, 109), RED_COLOR, cv2.FILLED)
        cv2.putText(img, "Volume", (60, 130), FONT_HERSHEY_PLAIN, 1, RED_COLOR, 2)
                    
        
        cv2.imshow("Image", img)
        if (cv2.waitKey(1) & 0xFF == EXIT_KEY):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()