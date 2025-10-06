import cv2
import numpy as np


def is_l_shaped_hexagon(approx):

    # 获取所有顶点
    points = [tuple(point[0]) for point in approx]

    # 计算所有边长
    side_lengths = []
    for i in range(6):
        pt1 = np.array(points[i])
        pt2 = np.array(points[(i + 1) % 6])
        side_lengths.append(np.linalg.norm(pt1 - pt2))

    # 计算所有内角
    angles = []
    for i in range(6):
        p1 = np.array(points[i])
        p2 = np.array(points[(i + 1) % 6])
        p3 = np.array(points[(i + 2) % 6])

        v1 = p1 - p2
        v2 = p3 - p2

        # 计算角度
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1, 1)  # 防止浮点误差
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)


    # 90度角
    right_angles = sum(1 for angle in angles if 75 <= angle <= 105)

    # 边长变异
    side_cv = np.std(side_lengths) / np.mean(side_lengths) if np.mean(side_lengths) > 0 else 0

    # 凹形
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(approx)
    solidity = contour_area / hull_area if hull_area > 0 else 0

    # L形六边形判断
    if right_angles >= 2 and side_cv > 0.25 and solidity < 0.85:
        return True, angles, side_lengths

    return False, angles, side_lengths



img_raw = cv2.imread('img_raw.png')

gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
_, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(binary2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

contour_img = img_raw.copy()

min_area = 300

for contour in contours:
    area = cv2.contourArea(contour)
    if area < min_area:
        continue

    # 多边形逼近
    epsilon = 0.023 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 六边形筛选
    if len(approx) == 6:
        # 函数检测L形六边形
        is_l_shape, angles, side_lengths = is_l_shaped_hexagon(approx)

        if is_l_shape:
            # 计算边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 绘制红色边界框
            cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 0, 255), 2)



cv2.imshow('result_v2.png', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('result_v2.png', contour_img)