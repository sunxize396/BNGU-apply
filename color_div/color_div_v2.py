import cv2
import numpy as np
img_bgr = cv2.imread('color_div_test.png')
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

lowerb_pink = np.array([160, 30, 30])
upperb_pink = np.array([180, 255, 255])

mask = cv2.inRange(img_hsv, lowerb_pink, upperb_pink)

img_out = cv2.bitwise_and(img_bgr, img_bgr, mask = mask)


cv2.imshow("mask", mask)
cv2.imshow("img", img_out)
cv2.imwrite("img_out2.png", img_out)
cv2.waitKey(0)
