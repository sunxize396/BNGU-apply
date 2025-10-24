import cv2
import numpy as np
from math import atan2, degrees

PYR_LEVELS = 3
MIN_AREA = 120
EPS_COEFF = 0.018
DILATE_ITERS = 1
SHOW_DEBUG = False

def auto_canny(img, sigma=0.33):
    med = np.median(img)
    lower = int(max(0, (1.0 - sigma) * med))
    upper = int(min(255, (1.0 + sigma) * med))
    edges = cv2.Canny(img, lower, upper, apertureSize=3, L2gradient=True)
    return edges

def preprocess(gray):
    # 对比度受限自适应直方图均衡
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_eq = clahe.apply(gray)
    # 轻度双边滤波
    smooth = cv2.bilateralFilter(gray_eq, d=5, sigmaColor=50, sigmaSpace=50)
    return smooth

def angle_between(v1, v2):
    # 计算两向量夹角
    dot = np.dot(v1, v2)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    cosang = np.clip(dot / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))

def compute_angles(points):
    # points：顺时针/逆时针依次排列的顶点
    n = len(points)
    angles = []
    for i in range(n):
        p_prev = np.array(points[(i - 1) % n], dtype=float)
        p_curr = np.array(points[i], dtype=float)
        p_next = np.array(points[(i + 1) % n], dtype=float)
        v1 = p_prev - p_curr
        v2 = p_next - p_curr
        ang = angle_between(v1, v2)
        angles.append(ang)
    return angles

def is_concave(points):
    # 至少存在一个凹角
    # 通过连续边的叉积符号变化判断是否存在反射角
    pts = np.array(points, dtype=float)
    n = len(pts)
    signs = []
    for i in range(n):
        a = pts[i]
        b = pts[(i+1)%n]
        c = pts[(i+2)%n]
        ab = b - a
        bc = c - b
        cross = ab[0]*bc[1] - ab[1]*bc[0]
        signs.append(np.sign(cross))
    # 同向凸，存在符号变化通常意味着凹
    return np.any(np.array(signs) == 0) or (np.min(signs) < 0 and np.max(signs) > 0)

def is_l_shaped_hexagon_strict(approx):
    # 顶点序列
    pts = [tuple(p[0]) for p in approx]
    n = len(pts)
    if n != 6:
        return False, [], []

    # 边向量和边长
    edges = []
    lengths = []
    perim = 0.0
    for i in range(n):
        a = np.array(pts[i], dtype=float)
        b = np.array(pts[(i+1)%n], dtype=float)
        v = b - a
        edges.append(v)
        d = np.linalg.norm(v)
        lengths.append(d)
        perim += d

    if perim <= 1e-6:
        return False, [], []

    # 角度
    angles = []
    reflex_count = 0
    for i in range(n):
        p_prev = np.array(pts[(i-1)%n], dtype=float)
        p_curr = np.array(pts[i], dtype=float)
        p_next = np.array(pts[(i+1)%n], dtype=float)
        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        # 无符号夹角
        dot = float(np.dot(v1, v2))
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return False, [], []
        cosang = np.clip(dot/(n1*n2), -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))  # 0~180

        # 用叉积判定内角
        cross = v1[0]*v2[1] - v1[1]*v2[0]

        if cross < 0:
            inner = 360.0 - ang
            reflex_count += 1
        else:
            inner = ang
        angles.append(inner)


    # 只有一个凹角，且接近 270°
    if reflex_count != 1:
        return False, angles, lengths
    reflex_angles = [a for a in angles if a > 180]
    big_reflex = reflex_angles[0] if reflex_angles else 0
    if not (230 <= big_reflex <= 310):  # 可把范围调窄/放宽
        return False, angles, lengths

    # 直角数量
    right_angles = sum(1 for a in angles if 80 <= a <= 100)
    if right_angles < 3:
        return False, angles, lengths

    # 边长
    min_len = min(lengths)
    max_len = max(lengths)
    if min_len < 0.05 * perim:   # 任何一边 < 周长 5%：多为噪声/锯齿
        return False, angles, lengths
    if (max_len / max(min_len, 1e-6)) > 4.0:  # 极端长条
        return False, angles, lengths

    # 实心度
    hull = cv2.convexHull(approx)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(approx)
    solidity = contour_area / hull_area if hull_area > 0 else 0
    if solidity >= 0.92:  # 太接近凸六边形/长条
        return False, angles, lengths

    return True, angles, lengths


def nms_boxes(boxes, scores, iou_thresh=0.4):
    # boxes: [x1,y1,x2,y2]
    if len(boxes) == 0:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

print("开始实时检测 L 形六边形... 按 'q' 键退出")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("无法获取视频帧")
        break

    h0, w0 = frame.shape[:2]
    draw = frame.copy()
    gray0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray0 = preprocess(gray0)

    all_boxes = []
    all_scores = []

    # 多尺度金字塔
    scale = 1.0
    img = gray0.copy()
    color_for_debug = frame.copy()

    for lvl in range(PYR_LEVELS):
        # 在当前尺度处理
        edges = auto_canny(img)
        # 形态学闭运算 膨胀
        kernel = np.ones((3,3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        edges = cv2.dilate(edges, kernel, iterations=DILATE_ITERS)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            perim = cv2.arcLength(cnt, True)
            if perim < 12:
                continue

            eps = max(1.0, EPS_COEFF * perim)
            approx = cv2.approxPolyDP(cnt, eps, True)
            if len(approx) != 6:
                continue

            # 先做面积尺度筛
            area_scaled = cv2.contourArea(approx)
            min_area_scaled = (MIN_AREA / (scale * scale))
            if area_scaled < min_area_scaled:
                continue


            rect = cv2.minAreaRect(approx)
            (cx, cy), (w_rect, h_rect), angle_rect = rect
            w_rect, h_rect = max(w_rect, 1e-6), max(h_rect, 1e-6)
            ar = max(w_rect, h_rect) / min(w_rect, h_rect)
            if ar > 3.0:
                continue

            # L 六边形判定
            is_l, angles, side_lengths = is_l_shaped_hexagon_strict(approx)
            if not is_l:
                continue

            # 进 NMS
            x, y, w, h = cv2.boundingRect(approx)
            x1 = int(x * scale)
            y1 = int(y * scale)
            x2 = int((x + w) * scale)
            y2 = int((y + h) * scale)
            score = w * h
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(score)

        # 进入下一层（缩小图像，检测更小目标）
        if lvl < PYR_LEVELS - 1:
            # 下采样一半
            img = cv2.pyrDown(img)
            scale *= 2.0  # 记录当前层到原图的放大倍数

        if SHOW_DEBUG:
            cv2.imshow(f'edges_lvl{lvl}', edges)

    # NMS 去重
    keep_idx = nms_boxes(all_boxes, all_scores, iou_thresh=0.45)
    for i in keep_idx:
        x1, y1, x2, y2 = all_boxes[i]
        cv2.rectangle(draw, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow("L-Shaped Hexagon Detection", draw)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
