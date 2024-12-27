# 导入所需的库
from sahi import AutoDetectionModel  # 用于自动化目标检测的模型
from sahi.predict import get_sliced_prediction  # 用于进行预测的函数
import cv2  # OpenCV，用于图像和视频处理
import os  # 用于操作文件和环境变量
import time  # 用于计时，计算帧率

# 设置环境变量，防止 OpenCV 报错
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 指定 YOLOv8 模型的路径
yolov8_model_path = r"G:\dian_xing_bei\在kaggle训练的模型\kaggle_best(0.877).pt"

# 加载预训练的目标检测模型，指定模型类型、路径、置信度阈值和使用的设备（GPU 或 CPU）
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8',  # 使用 YOLOv8 模型
    model_path=yolov8_model_path,  # 指定训练好的模型路径
    confidence_threshold=0.20,  # 设置置信度阈值，低于此值的预测将被忽略
    device='cuda:0',  # 设置设备为 GPU（cuda:0 表示使用第一张 GPU），如果是 CPU 设备可改为 'cpu'
)

# 设置视频文件路径
video_path = r"G:\dian_xing_bei\用来测试的视频\影史上关于躺着的镜头.mp4"  # 输入视频的路径
output_video_path = r"G:\dian_xing_bei\用来测试的视频\output_video.avi"  # 输出视频路径

# 确保输出文件夹存在
output_folder = os.path.dirname(output_video_path)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 使用 OpenCV 读取视频
cap = cv2.VideoCapture(video_path)

# 获取视频的帧率（FPS）和分辨率
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的原始宽度
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的原始高度

# 自动调整 slice_height 和 slice_width
slice_width = frame_width // 2  # 设置切片宽度为视频宽度的一半
slice_height = frame_height // 2  # 设置切片高度为视频高度的一半

# 创建视频写入对象，用于保存处理后的视频
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 使用 XVID 编解码器
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width // 2, frame_height // 2))

# 用于计算帧率的变量
prev_time = time.time()

# 按帧读取视频并进行目标检测
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 降低分辨率以加速处理
    # frame_resized = cv2.resize(frame, (frame_width // 2, frame_height // 2))
    frame_resized = cv2.resize(frame, (frame_width // 2, frame_height // 2))

    # 记录每帧的处理时间
    current_time = time.time()
    fps_display = int(1 / (current_time - prev_time))  # 计算FPS
    prev_time = current_time

    # 在帧上显示帧率
    cv2.putText(frame_resized, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 保存当前帧到临时图像文件
    temp_frame_path = "temp_frame.jpg"
    cv2.imwrite(temp_frame_path, frame_resized)

    # 使用 SAHI 进行图像预测
    result = get_sliced_prediction(
        temp_frame_path,  # 当前帧的路径
        detection_model,  # 使用已加载的检测模型
        slice_height=slice_height,  # 自动计算的切片高度
        slice_width=slice_width,  # 自动计算的切片宽度
        overlap_height_ratio=0.3,  # 设置切片垂直方向上的重叠比例，这里是 5%
        overlap_width_ratio=0.3  # 设置切片水平方向上的重叠比例，这里是 5%
    )

    # 导出预测结果的可视化图像，保存为临时图像文件
    result.export_visuals(export_dir="demo_data/", hide_labels=False, hide_conf=False)

    # 读取处理后的帧（可视化图像）
    processed_frame = cv2.imread("demo_data/prediction_visual.png")

    # 将处理后的帧写入输出视频
    out.write(processed_frame)

    # 显示处理后的帧（可选）
    cv2.imshow("Processed Frame", processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按 'q' 键退出
        break

# 释放视频资源并关闭窗口
cap.release()
out.release()
cv2.destroyAllWindows()
