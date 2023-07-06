import cv2
import mediapipe as mp
import time
import math
import pyautogui

class handDetector():
    def __init__(self, mode=False, maxHands=2, modelCom=1,detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.modelCom = modelCom
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxHands, self.modelCom, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.tipID = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for handLMS in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLMS, self.mphands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):
        self.lmlist = []
        xList = []
        yList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                h,w,c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h) # lm is ratios from img size so convert to pixels
                #print(f"id: {id}, x: {cx}, y: {cy}")
                xList.append(cx)
                yList.append(cy)
                self.lmlist.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 5, (255,0,0), cv2.FILLED)
            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            if draw:
                cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), (0,255,255), 2)
        return self.lmlist
    
    def fingersUp(self):
        fingers = []
        if self.lmlist[self.tipID[0]][1] > self.lmlist[self.tipID[0]-1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lmlist[self.tipID[id]][2] < self.lmlist[self.tipID[id]-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        x1,y1 = self.lmlist[p1][1:]
        x2,y2 = self.lmlist[p2][1:]
        cx,cy = (x1+x2)//2, (y1+y2)//2
        if draw:
            cv2.circle(img, (x1,y1), 10, (0,0,0), cv2.FILLED)
            cv2.circle(img, (x2,y2), 10, (0,0,0), cv2.FILLED)
            cv2.line(img, (x1,y1), (x2,y2), (0,0,0), 3)
            cv2.circle(img, (cx,cy), 10, (0,0,255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
    
def main():
    previoustime = 0
    currenttime = 0

    cap = cv2.VideoCapture(0)

    detector = handDetector()

    while True:
        _, img = cap.read()

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

if __name__ == "__main__":
    main()