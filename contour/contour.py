import cv2
import numpy as np

img_raw = cv2.imread('img_raw.png')

gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
_, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contour_img = img_raw.copy()

min_area = 300

hexagons = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        continue

    # 多边形逼近
    epsilon = 0.023 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 六边形筛选
    if len(approx) == 6:
        # 计算边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 绘制红色边界框
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('result', contour_img)
cv2.imwrite('result.png', contour_img)
cv2.waitKey(0)