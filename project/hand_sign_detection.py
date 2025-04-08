import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier('Model/keras_model.h5', 'Model/labels.txt')

offset = 20
imgSize = 300

counter = 0

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
            'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


while True:
    try:
        success, img = cap.read()
        imgOutput = img.copy()
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
                prediction, index = classifier.getPrediction(imgWhite, draw = False)
                print(prediction, index)

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap: hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw = False)
                print(prediction, index)

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), 
                            (x - offset + 140, y - offset - 50 + 50), (255, 0 , 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 27), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), 
                            (x + w + offset, y + h + offset), (255, 0 , 255), 4)

            
            cv2.imshow('ImageCrop', imgCrop)
            cv2.imshow('ImageWhite', imgWhite)
        
        cv2.imshow('Image', imgOutput)
        cv2.waitKey(1)

    except Exception as e:
        print(f"Error: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
