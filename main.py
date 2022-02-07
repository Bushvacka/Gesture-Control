import cv2 # Image procesing
import mediapipe as mp # Hand tracking model


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
while (cap.isOpened()):
    img = cap.read()