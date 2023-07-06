import cv2
import mediapipe as mp
import math
import handtracking
import pyautogui

print(pyautogui.size())

previoustime = 0
currenttime = 0

cap = cv2.VideoCapture(0)

detector = handtracking.handDetector()

while True:
    success, img = cap.read()

    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        print(lmlist[12],lmlist[8])
        x1,y1 = lmlist[12][1], lmlist[12][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]
        cx,cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 10, (0,0,0), cv2.FILLED)
        cv2.circle(img, (x2,y2), 10, (0,0,0), cv2.FILLED)
        cv2.line(img, (x1,y1), (x2,y2), (0,0,0), 3)
        cv2.circle(img, (cx,cy), 10, (0,0,255), cv2.FILLED)

        length = math.hypot(x2-x1, y2-y1)
        #print(length)
        if length < 40:
            cv2.circle(img, (cx,cy), 10, (0,255,0), cv2.FILLED)


    cv2.imshow('image', img)
    cv2.waitKey(1)