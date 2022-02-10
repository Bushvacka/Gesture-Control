# Data manipulation
import math
import numpy as np
# Image processing
import cv2
# Hand Tracking
import handDetector
# Volume Control
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume


EXIT_KEY = 27
GREEN_COLOR = (0, 224, 0)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)
MIN = 30
MAX = 150

def main():
    detector = handDetector.handDetector(model_complexity=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    vol_control = cast(interface, POINTER(IAudioEndpointVolume))
    vol_range = vol_control.GetVolumeRange()
    vol_rect = 0
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

                # multi_handedness -> {'classification': [{'index': 1, 'score': 0.92601216, 'label': 'Right'}]}
                handedness = results.multi_handedness[i].classification[0].label
                if (handedness == "Right"):
                    if thumb and index:
                        dist = math.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
                        vol = np.interp(dist, [MIN, MAX], [vol_range[0], vol_range[1]])
                        vol_rect = np.interp(vol, [vol_range[0], vol_range[1]], [0, 99])
                        vol_control.SetMasterVolumeLevel(vol, None)
                        cv2.line(img, thumb, index, BLACK_COLOR, 3)

        cv2.rectangle(img, (10, 10), (50, 110), BLACK_COLOR, 2)
        cv2.rectangle(img, (11, int(109 - vol_rect)), (49, 109), RED_COLOR, cv2.FILLED)
                    
        
        cv2.imshow("Image", img)
        if (cv2.waitKey(1) & 0xFF == EXIT_KEY):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()