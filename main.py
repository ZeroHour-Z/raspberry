import time
# 创建日志文件，记录程序启动时间
log = open(f'/home/pi/Documents/{time.time()}.txt', 'w+')
print(time.ctime(), file=log)
log.close()

try:
    # 导入所需的库
    import cv2          # OpenCV图像处理库
    import numpy as np  # 数值计算库
    import QRread       # QR码识别模块
    import CrossDetect  # 十字线检测模块
    import serial       # 串口通信库
    import QRread       # QR码识别模块（重复导入）
    import threading    # 多线程库
    import ColorDetect  # 颜色检测模块
    import onnxruntime  # ONNX模型推理库
    import time         # 时间处理库
    import sig          # GPIO信号控制模块
except ImportError as e:
    print(e)
    exit()

# 初始化GPIO引脚
sig.init()

# 全局变量定义
model = None  # 提前加载模型变量

# 初始化串口通信
# 连接到/dev/ttyAMA0，波特率14400，超时0.5秒
port = serial.Serial(port='/dev/ttyAMA0', baudrate=14400, timeout=0.5)

# 初始化摄像头
# 身体摄像头：用于QR码识别
cam_body = cv2.VideoCapture('/dev/body0')
# 手臂摄像头：用于颜色检测
cam_arm = cv2.VideoCapture('/dev/arm0')
# 设置摄像头缓冲区大小，减少延迟
cam_arm.set(cv2.CAP_PROP_BUFFERSIZE, 3)
cam_body.set(cv2.CAP_PROP_BUFFERSIZE, 3)

try:
    # 测试摄像头是否正常工作
    r, _ = cam_arm.read()
    r2, _ = cam_body.read()
    if not r or not r2:
        # 如果摄像头初始化失败，重试5次
        n = 0
        while n < 5:
            cam_body = cv2.VideoCapture('/dev/body0')
            cam_arm = cv2.VideoCapture('/dev/arm0')
            r, _ = cam_arm.read()
            r2, _ = cam_body.read()
            if r and r2:
                break

except Exception as e:
    exit()
    
# 配置ONNX运行时选项
session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
model_path = r"better_t2.onnx"  # 模型文件路径

# 加载ONNX模型
model = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'], sess_options=session_options)

def call(stat, data):
    """
    串口通信回调函数
    参数:
        stat: 当前状态 (0=QR识别, 1=颜色检测, 2=QR识别2)
        data: 要发送的数据
    """
    # 颜色映射字典：将颜色名称映射到数字
    dic = {'ho': 17, 'lv': 18, 'la': 19, 'ba': 20, 'he': 21}
    global port
    
    # 检查串口是否打开
    if not port.is_open:
        print('未打开')
        return
        
    if stat == 0 or stat == 2:
        # QR码识别状态：发送字母A-Z（对应数字1-26）
        text = chr(ord('A') + int(data) - 1)
        l = port.write(str(text).encode('utf-8'))
        print(l, text)
    elif stat == 1:
        # 颜色检测状态：发送检测到的颜色
        if data == 'err':
            return
        l = port.write(chr(ord('A') + int(dic[data]) - 1).encode('utf-8'))
        print(l, chr(ord('A') + int(dic[data]) - 1))

def main(stat=0):
    """
    主函数：状态机控制
    参数:
        stat: 初始状态，默认为0（QR码识别）
    """
    color_cnt = 0
    target = []
    
    while True:
        if stat == 0:  # 任务1：QR码识别
            # 创建QR码识别的两个线程
            tqr1 = threading.Thread(target=QRread.capture_frames, args=(cam_body,))  # 图像采集线程
            tqr2 = threading.Thread(target=QRread.inference, args=(call, stat,))      # 推理线程
            
            # 启动线程
            tqr1.start()
            tqr2.start()
            
            # 等待串口确认信号
            while True:
                r = port.read(2)  # 读取2字节
                print('read', r)
                if r.decode('utf-8') == 'ok' or r.decode('utf-8') == 'ko':  # ko是应对一些奇怪的情况
                    # 收到确认信号，点亮LED
                    sig.turn_on()
                    time.sleep(1)
                    sig.turn_off()
                    
                    # 停止QR码识别线程
                    QRread.flag = False
                    tqr1.join()
                    tqr2.join()
                    
                    # 切换到颜色检测状态
                    stat = 1
                    break
                time.sleep(0.01)
                
        if stat == 2:  # 任务2：QR码识别（第二次）
            # 重新启动QR码识别
            QRread.flag = True
            tqr3 = threading.Thread(target=QRread.capture_frames, args=(cam_body,))
            tqr4 = threading.Thread(target=QRread.inference, args=(call, stat,))
            
            tqr3.start()
            tqr4.start()
            
            while True:
                r = port.read(2)
                print('read', r)
                if r.decode('utf-8') == 'ok' or r.decode('utf-8') == 'ko':
                    sig.turn_on()
                    time.sleep(1)
                    sig.turn_off()
                    
                    # 停止线程并退出程序
                    QRread.flag = False
                    tqr3.join()
                    tqr4.join()
                    print('finish')
                    cam_body.release()
                    exit()
                    break
                time.sleep(0.01)
                
        elif stat == 1:  # 颜色检测任务
            # 创建颜色检测的两个线程
            t1 = threading.Thread(target=ColorDetect.capture_frames, args=(cam_arm,))  # 图像采集线程
            t2 = threading.Thread(target=ColorDetect.inference, args=(model, call, stat, 0.7,))  # 推理线程
            
            t1.start()
            t2.start()
            
            det = set()  # 用于存储检测到的颜色集合
            
            while True:
                r = port.read(1)  # 读取1字节
                if r.decode('utf-8') == '':
                    continue
                    
                if r.decode('utf-8') == 'Z':
                    # 收到'Z'信号，继续检测
                    ColorDetect.pause_flag = False
                else:
                    # 收到其他信号，暂停检测并记录
                    ColorDetect.pause_flag = True
                    sig.turn_on()
                    time.sleep(0.5)
                    sig.turn_off()
                    det.add(r.decode('utf-8'))
                    print('copy', r.decode('utf-8'))
                    print(det)
                    
                    if r.decode('utf-8') == 'Y':
                        # 收到'Y'信号，完成颜色检测
                        ColorDetect.flag = False
                        ColorDetect.pause_flag = False
                        t1.join()
                        t2.join()
                        stat = 2  # 切换到QR码识别状态
                        break
                        
                    if det == set(['S', 'Q', 'T', 'U', 'R']):
                        # 检测到所有目标颜色，完成检测
                        ColorDetect.pause_flag = False
                        ColorDetect.flag = False
                        t1.join()
                        t2.join()
                        stat = 3  # 切换到等待状态
                        break
            
            cam_arm.release()
            
        elif stat == 3:  # 等待状态
            while True:
                r = port.read(1)
                if r.decode('utf-8') == 'Y':
                    stat = 2  # 收到'Y'信号，切换到QR码识别状态
                    break

if __name__ == '__main__':
    # 程序启动时的LED指示
    sig.turn_on()
    time.sleep(0.5)
    sig.turn_off()
    time.sleep(0.5)
    sig.turn_on()
    time.sleep(0.5)
    sig.turn_off()
    
    # 启动主程序
    main()
    
    # 清理GPIO资源
    sig.release()
