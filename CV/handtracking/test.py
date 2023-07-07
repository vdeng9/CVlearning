import cv2
import mediapipe as mp
import numpy as np
import handtracking
import pyautogui

#print(pyautogui.size())
pyautogui.FAILSAFE = False

px, py = 0,0
curx, cury = 0,0
camw = 640
camh = 480
scale = 1

cap = cv2.VideoCapture(0)
cap.set(3, camw)
cap.set(4, camh)

detector = handtracking.handDetector(maxHands=1)

screenw, screenh = pyautogui.size()

while True:
    _, img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findPosition(img)
    if len(lmlist) != 0:
        #print(lmlist[12],lmlist[8])
        x1,y1 = lmlist[12][1], lmlist[12][2]
        x2,y2 = lmlist[8][1], lmlist[8][2]
        fingers = detector.fingersUp()
        if fingers[1] == 1 and fingers[2] == 0:
            cv2.circle(img, (x1, y1), 10, (0,0,255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0,255,0), cv2.FILLED)
            x3 = np.interp(x1, (100, camw-100), (0, screenw))
            y3 = np.interp(y1, (100, camh-100), (0, screenh))
            curx = px + (x3 - px)/scale
            cury = py + (y3 - py)/scale
            pyautogui.moveTo(screenw-curx, cury)
            px, py = curx, cury
        elif fingers[1] == 1 and fingers[2] == 1:
            length, img, distinfo = detector.findDistance(8, 12, img)
            if length < 35:
                cv2.circle(img, (distinfo[4], distinfo[5]), 10, (0,255,0), cv2.FILLED)
                pyautogui.leftClick()

    cv2.imshow('test image', img)
    cv2.waitKey(1)