import onnxruntime
import torch
import numpy as np
import cv2
from utils.augmentations import classify_transforms
from utils.general import non_max_suppression
import threading
import time
import signal

# 全局变量
flag = True
frame = None
pause_flag = True
frame_lock = threading.Lock()

# 获取模型的输出信息
# outputs = ort_session.get_outputs()
# 打印每个输出的详细信息
# for output in outputs:
#     print(f"输出名称: {output.name}")
#     print(f"输出类型: {output.type}")
#     print(f"输出形状: {output.shape}")

'''
输入名称: images
输入类型: tensor(float)
输入形状: [1, 3, 640, 640]
'''

def warm_up(model, num=1):
    """
    模型预热函数，用于提高推理速度
    参数:
        model: ONNX模型
        num: 预热次数
    """
    for i in range(num):
        img = torch.empty((1, 3, 640, 640), dtype=torch.float, device='cpu')
        img = img.cpu().numpy()
        model.run(None, {model.get_inputs()[0].name: img})

def from_numpy(x):
    """
    将numpy数组转换为torch张量
    参数:
        x: numpy数组或torch张量
    返回:
        torch张量
    """
    return torch.from_numpy(x).to('cpu') if isinstance(x, np.ndarray) else x

def softmax(x):
    """
    计算softmax函数
    参数:
        x: 输入数组
    返回:
        softmax结果
    """
    exp_x = np.exp(x - np.max(x))  # 减去最大值以防止溢出
    return exp_x / exp_x.sum(axis=-1, keepdims=True)

def preprocess_defalt(img):
    """
    默认的图像预处理函数（用于YOLO模型）
    参数:
        img: 输入图像
    返回:
        预处理后的图像
    """
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)  # contiguous
    img = torch.from_numpy(img).to('cpu')
    img = img.float()  # uint8 to fp16/32
    img /= 255  # 0 - 255 to 0.0 - 1.0
    img = img[None]
    img = img.cpu().numpy()
    return img

def preprocess_cls(img):
    """
    分类模型的图像预处理函数
    参数:
        img: 输入图像
    返回:
        预处理后的图像
    """
    trans = classify_transforms(224)
    img = trans(img)
    img = torch.Tensor(img).to('cpu')
    if len(img.shape) == 3:
        img = img[None]
    img = img.float()
    img = img.cpu().numpy()
    return img

def predict_cls(img, model, fliter=0.5):
    """
    分类模型预测函数
    参数:
        img: 输入图像
        model: ONNX模型
        fliter: 置信度阈值
    返回:
        预测的颜色类别
    """
    # dic = {0: 'a', 1: 'b', 2: 'ba', 3: 'bj', 4: 'c', 5: 'he', 6: 'ho', 7: 'la', 8: 'lv'}
    dic = {0: 'ba', 1: 'bj', 2: 'he', 3: 'ho', 4: 'la', 5: 'lv'}  # 0 白 1 黑 2 红 3 绿 4 蓝 5 紫
    
    img = preprocess_cls(img)
    res = model.run(None, {model.get_inputs()[0].name: img})
    res = from_numpy(res)
    res = np.squeeze(res)
    res = softmax(res)
    ind = np.argmax(res)
    return dic[ind] if res[ind] > fliter else 'None'

def predict(img, model, fliter=0.5):
    """
    YOLO模型预测函数
    参数:
        img: 输入图像
        model: ONNX模型
        fliter: 置信度阈值
    返回:
        预测的颜色类别
    """
    dic = {0: 'he', 1: 'ba', 2: 'ho', 3: 'lv', 4: 'la', 5: 'a', 6: 'b', 7: 'c'}  # 0 黑 1 白 2 红 3 绿 4 蓝 5 a 6 b 7 c
    
    img = preprocess_defalt(img)
    res = model.run(None, {model.get_inputs()[0].name: img})
    res = from_numpy(res[0])
    res = non_max_suppression(res, fliter, 0.45, None, False, max_det=1000)
    res = res[0].cpu().numpy()
    
    if len(res) > 1:
        return 'err'
    if len(res) == 0:
        return 'None'
    return dic[res[0][5]]

def capture_frames(cap):
    """
    图像采集线程函数
    参数:
        cap: 摄像头对象
    """
    global frame
    global flag
    global frame_lock
    
    while flag:
        while pause_flag:
            time.sleep(0.01)
        ret, temp_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = temp_frame.copy()

def inference(model, call, stat, fliter=0.5, method=predict):
    """
    推理线程函数
    参数:
        model: ONNX模型
        call: 回调函数
        stat: 当前状态
        fliter: 置信度阈值
        method: 预测方法
    """
    global frame
    
    while flag:
        while pause_flag:
            time.sleep(0.01)
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
            
            # 开始计时
            start_time = time.time()
            # 进行推理（这里用一个模拟的时间延迟来表示推理过程）
            t = method(current_frame, model, fliter)
            if t != 'bj' and t != 'None':
                print('detect', t)
                call(stat, t)
            # 结束计时
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"推理时间: {processing_time:.4f} 秒")
        else:
            time.sleep(0.1)

def predict_lite(img, model):
    """
    轻量级模型预测函数
    参数:
        img: 输入图像
        model: ONNX模型
    返回:
        预测结果
    """
    img = cv2.resize(img, [320, 320], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)
    # print(blob.shape)
    outs = model.run(None, {model.get_inputs()[0].name: blob})[0].squeeze()
    return outs

def draw_bbx(img, outs):
    """
    在图像上绘制边界框
    参数:
        img: 输入图像
        outs: 模型输出结果
    返回:
        绘制了边界框的图像
    """
    if len(outs) == 0:
        return img
    
    try:
        outs[0][0]
    except:
        outs = [outs]
    
    for out in outs:
        x1, y1, x2, y2, conf, cl = out
        if conf <= 0.3:
            continue
        x1, x2 = x1 * 640 / 320, x2 * 640 / 320
        y1, y2 = y1 * 480 / 320, y2 * 480 / 320
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        cv2.putText(img, str(cl + conf), (int(x1), int(y1)), 0, 0.7, (0, 0, 255), 2)
    
    return img

if __name__ == '__main__':
    def handle_sigtstp(signum, frame):
        """
        信号处理函数
        """
        cam.release()
        exit()
    
    # onnx_model_path = r"D:\yolov5-master\runs\train\exp23\weights\best.onnx"
    onnx_model_path = r"better_cls.onnx"
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'], sess_options=session_options)
    
    # img = cv2.imread(r'img\642.jpg')
    # r = predict(img, ort_session)
    # print(r)
    # cv2.imshow('a', img)
    # cv2.waitKey(0)
    # img2 = cv2.imread(r'D:\Proj\indcmp\img\2024_09_02_19_13_IMG_5617.JPG')  # C
    # img3 = cv2.imread(r'D:\Proj\indcmp\img\2024_09_02_19_12_IMG_5578.JPG')  # 黑
    # cam = cv2.VideoCapture(r"D:\YOLOv5-Lite-master\datasets\color.mp4")
    cam = cv2.VideoCapture('/dev/arm0')
    # cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # cam2 = cv2.VideoCapture(1)
    # cam2.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    # warm_up(ort_session)
    print(cam.isOpened())
    cnt = 0
    signal.signal(signal.SIGTSTP, handle_sigtstp)
    
    while True:
        r, img = cam.read()
        # rr, iimg = cam2.read()
        # cnt += 1
        # if cnt != 5:
        #     continue
        # cnt = 1
        # img = cv2.resize(img, (480, 640))
        if not r:
            break
        # res = predict(img, ort_session, 0.7)
        # print(res)
        print(predict_cls(img, ort_session, 0.7))
        # cv2.putText(img, predict_cls(img, ort_session, 0.7), (100, 100), 0, 1.5, (200, 100, 100), 2, 0)
        # res = predict_lite(img, ort_session)
        # img = draw_bbx(img, res)
        # cv2.imshow('a', img)
        # cv2.imshow('b', iimg)
        cv2.waitKey(1)

# 获取模型的输入信息
# inputs = ort_session.get_inputs()
# 打印每个输入的详细信息
# for input in inputs:
#     print(f"输入名称: {input.name}")
#     print(f"输入类型: {input.type}")
#     print(f"输入形状: {input.shape}")
# 获取模型的输入名称
