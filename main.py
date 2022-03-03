import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
colorR = (255, 0, 255)


class DragRect():
    def __init__(self, posCenter, size=[75, 100]):
        self.posCenter = posCenter
        self.size = size

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size
        if cx - w // 2 < cursor[0] < cx + w // 2 and cy - h // 2 < cursor[1] < cy + h // 2:
            self.posCenter = cursor


rectList = []
for x in range(6):
    rectList.append(DragRect([x * 100 + 50, 50]))

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    _, img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if lmList:
        l, _, _ = detector.findDistance(lmList[8], lmList[12], img)
        print(l)
        if l < 45:
            cursor = lmList[8]
            for rect in rectList:
                rect.update(cursor)

        # print(cursor)
        # call update

    imgNew = np.zeros_like(img, np.uint8)
    for rect in rectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2), colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=5)
    out = img.copy()
    alpha = 0.3
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    cv2.imshow("Image", out)
    cv2.waitKey(1)
