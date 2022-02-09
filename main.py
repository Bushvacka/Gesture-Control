import math # Data manipulation
import numpy as np
import cv2 # Image procesing
import mediapipe as mp # Hand tracking model
import handDetector # Hand tracking module
from google.protobuf.json_format import MessageToDict

EXIT_KEY = 27
GREEN_COLOR = (0, 224, 0)
BLACK_COLOR = (0, 0, 0)
RED_COLOR = (0, 0, 255)

detector = handDetector.handDetector(model_complexity=1)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while (cap.isOpened()):
    success, img = cap.read()

    if not success:
        print("Frame not loaded")
        break
    
    # Flip the image horizontally
    img = cv2.flip(img, 1)

    # Otherwise, the img is valid

    results = detector.detectHands(img, True)
    if results.multi_hand_landmarks:
        #Iterate through all detected hands
        for i, hand_lmks in enumerate(results.multi_hand_landmarks):
            thumb = detector.getLandmarkPos(img, hand_lmks, "THUMB_TIP")
            index = detector.getLandmarkPos(img, hand_lmks, "INDEX_FINGER_TIP")

            # Convert "multi_handedness" into usable values.
            # multi_handedness -> {'classification': [{'index': 1, 'score': 0.92601216, 'label': 'Right'}]}
            handedness = results.multi_handedness[i].classification[0].label
            if (handedness == "Right"):
                if thumb and index:
                    dist = math.sqrt((thumb[0] - index[0])**2 + (thumb[1] - index[1])**2)
                    volume = np.interp(dist, [15, 100], [0, 1])
                    print(volume)
                    cv2.line(img, thumb, index, BLACK_COLOR, 3)
                
    
    cv2.imshow("Image", img)
    if (cv2.waitKey(1) & 0xFF == EXIT_KEY):
        cv2.destroyAllWindows()
        break

