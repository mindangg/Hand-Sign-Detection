import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)

detector = HandDetector(maxHands=1)

offset = 20 
imgSize = 300  

folder = "data\\Z"

counter = 0
max_images = 850 

# Tạo thư mục nếu chưa tồn tại
if not os.path.exists(folder):
    os.makedirs(folder)

print("Bắt đầu chụp ảnh")

while counter < max_images:
    try:
        success, img = cap.read()
        if not success:
            print("Failed to capture image")
            continue

        hands, img = detector.findHands(img)
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255


            y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
            x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
            imgCrop = img[y1:y2, x1:x2]

            aspectRatio = h / w

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

            counter += 1
            filename = f"{folder}/Image_{time.time()}.jpg"
            cv2.imwrite(filename, imgWhite)
            print(f"Đã lưu {counter}/{max_images} ảnh: {filename}")

            time.sleep(0.1)

    except Exception as e:
        print(f"❌ Lỗi: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Hoàn thành! Đã lưu đủ ảnh.")
cap.release()
cv2.destroyAllWindows()

