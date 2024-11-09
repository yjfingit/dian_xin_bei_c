import cv2
import os

# 视频文件路径
video_path = r'G:\ultralytics-main\国外监控视频 (人数很多).mp4'  # 请根据你的实际路径修改

# 输出截图的保存路径
output_dir = 'G:/ultralytics-main/demo_data/'  # 请根据你的实际路径修改

# 截图的时间间隔（单位：秒），例如每隔2秒截取一帧
interval_seconds = 2

# 创建输出文件夹，如果不存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率（FPS）
fps = cap.get(cv2.CAP_PROP_FPS)

# 获取视频的总帧数
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 计算每次截图所需要的帧数
interval_frames = int(fps * interval_seconds)

frame_number = 0  # 当前帧数
saved_count = 0  # 已保存的截图数

while True:
    ret, frame = cap.read()
    if not ret:
        break  # 如果没有更多帧，退出循环

    # 如果当前帧是我们需要的截图时间点（基于帧间隔）
    if frame_number % interval_frames == 0:
        # 保存截图
        screenshot_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
        cv2.imwrite(screenshot_path, frame)
        print(f"Saved {screenshot_path}")
        saved_count += 1

    frame_number += 1

# 释放视频对象
cap.release()
print("Video processing completed.")
