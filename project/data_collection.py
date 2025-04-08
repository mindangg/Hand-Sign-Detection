import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "data/9"
counter = 0

while True:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue
        
        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            # np.uint8 = 0 to 255
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # make sure hand doesnt go out of webcam
            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w

            # check to normalize the img of hand
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap: wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
            
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        
        cv2.imshow('Image', img)
        key = cv2.waitKey(1)
        if key == ord('s'):
            while counter < 1000:
                counter += 1
                # cv2.imwrite(f'{folder}/Image_{counter}.jpg', imgWhite)
                cv2.imwrite(f'{folder}/{str(counter).zfill(3)}.jpg', imgWhite)
                print(counter)

    except Exception as e:
        print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
     

cap.release()
cv2.destroyAllWindows()
