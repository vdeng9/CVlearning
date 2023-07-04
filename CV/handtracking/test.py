import cv2
import mediapipe as mp
import time
import handtrackmodule

previoustime = 0
currenttime = 0

cap = cv2.VideoCapture(0)

detector = handtrackmodule.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist[0])

    currenttime = time.time()
    fps = 1/(currenttime-previoustime)
    previoustime = currenttime

    cv2.putText(img, str(int(fps)), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow('image', img)
    cv2.waitKey(1)