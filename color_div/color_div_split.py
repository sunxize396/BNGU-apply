import cv2

img_bgr = cv2.imread("color_div_test.png")
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
img_h, img_s, img_v = cv2.split(img_hsv)
mask_h = cv2.inRange(img_h, 160, 180)
mask_s = cv2.inRange(img_s, 30, 255)
mask_v = cv2.inRange(img_v, 30, 255)
mask_h_and_s = cv2.bitwise_and(mask_h, mask_s)
mask = cv2.bitwise_and(mask_h_and_s, mask_v)
img_out = cv2.bitwise_and(img_bgr, img_bgr, mask = mask)


# cv2.imshow("h", mask_h)
# cv2.imshow("s", mask_s)
# cv2.imshow("v", mask_v)
cv2.imshow("mask", mask)
cv2.imshow("img", img_out)
cv2.imwrite("img_out1.png", img_out)
cv2.waitKey(0)