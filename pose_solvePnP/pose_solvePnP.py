import cv2
import numpy as np

camera_matrix = np.array([
    [1.340597284330724e+03, 0, 6.536136288960888e+02],
    [0, 1.347810504232473e+03, 8.759024502391557e+02],
    [0, 0, 1]
], dtype=np.float32)

# 畸变系数
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# 黑色标记尺寸
MARKER_SIZE = 0.097

# 世界坐标系
object_points = np.array([
    [-MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, MARKER_SIZE / 2, 0],
    [MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)


def detect_black_marker(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 二值化
    _, binary = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)

    # 去除噪声
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 查找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # 最大的轮廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 多边形
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)

    if len(approx) == 4:
        points = approx.reshape(4, 2)

        # 计算中心点
        center = points.mean(axis=0)

        def sort_point(point):
            x, y = point - center
            if x >= 0 and y <= 0:
                return 0  # 右上
            elif x >= 0 and y >= 0:
                return 1  # 右下
            elif x <= 0 and y >= 0:
                return 2  # 左下
            else:
                return 3  # 左上

        sorted_points = sorted(points, key=sort_point)

        # 重排列：左上、右上、右下、左下
        sorted_points = [sorted_points[3], sorted_points[0], sorted_points[1], sorted_points[2]]

        return np.array(sorted_points, dtype=np.float32)

    return None


def calculate_pose(image_points):

    # solvePnP
    success, rvec, tvec = cv2.solvePnP(object_points, image_points, camera_matrix, dist_coeffs)

    if success:
        return rvec, tvec
    else:
        return None, None


def visualize_axes(image, rvec, tvec, camera_matrix, dist_coeffs):

    axis_length = MARKER_SIZE * 0.8

    axis_points = np.array([
        [0, 0, 0],  # 原点
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)

    projected_points, _ = cv2.projectPoints(axis_points, rvec, tvec, camera_matrix, dist_coeffs)
    projected_points = projected_points.reshape(-1, 2)

    # 绘制坐标轴
    origin = tuple(projected_points[0].astype(int))
    x_axis = tuple(projected_points[1].astype(int))
    y_axis = tuple(projected_points[2].astype(int))
    z_axis = tuple(projected_points[3].astype(int))

    # 绘制坐标轴线
    cv2.arrowedLine(image, origin, x_axis, (0, 0, 255), 3)  # X轴 - 红色
    cv2.arrowedLine(image, origin, y_axis, (0, 255, 0), 3)  # Y轴 - 绿色
    cv2.arrowedLine(image, origin, z_axis, (255, 0, 0), 3)  # Z轴 - 蓝色

    # 添加标签
    cv2.putText(image, 'X', x_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(image, 'Y', y_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, 'Z', z_axis, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    return image


def main():

    video_path = r"C:\Users\47z\Videos\33135ef1125d3c0207235db2e1ef668a.mp4"

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        # 试使用摄像头
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("无法打开摄像头")
            return

    print("按 'q' 退出程序")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("读取失败")
            break

        # 检测黑色标记
        image_points = detect_black_marker(frame)

        if image_points is not None:
            # 计算姿态
            rvec, tvec = calculate_pose(image_points)

            if rvec is not None:
                # 可视化坐标轴
                frame_with_axes = visualize_axes(frame.copy(), rvec, tvec, camera_matrix, dist_coeffs)

                # 绘制检测到的角点
                for i, point in enumerate(image_points):
                    cv2.circle(frame_with_axes, tuple(point.astype(int)), 5, (255, 255, 0), -1)
                    cv2.putText(frame_with_axes, str(i), tuple(point.astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Pose Estimation', frame_with_axes)
            else:
                cv2.imshow('Pose Estimation', frame)
        else:
            # 显示未检测到标记的信息
            info_frame = frame.copy()
            cv2.putText(info_frame, "No marker detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow('Pose Estimation', info_frame)

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()