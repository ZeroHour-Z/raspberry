import cv2
import threading
import time
import ColorDetect
import onnxruntime
# 全局变量
frame = None
frame_lock = threading.Lock()
inference_result = None
cap=None
def capture_frames():
    global frame
    global cap
    if not cap.isOpened():
        print("无法打开摄像头")
        cap = cv2.VideoCapture('/dev/arm0')
        # exit()
    while True:
        # print('hq..')
        ret, temp_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = temp_frame.copy()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

def inference():
    onnx_model_path=r"better.onnx"
    session_options=onnxruntime.SessionOptions()
    session_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    ort_session=onnxruntime.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'],sess_options=session_options)
    global frame, inference_result
    while True:
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
            # 开始计时
            start_time = time.time()
            # 进行推理（这里用一个模拟的时间延迟来表示推理过程）
            
            t=ColorDetect.predict(current_frame,ort_session,0.8)
            # cv2.putText(current_frame,t,(100,100),0,2,(0,255,0))
            # print(t)
            # 您可以在这里调用您的推理函数
            # time.sleep(1)  # 模拟推理时间为0.1秒
            # 结束计时
            end_time = time.time()
            processing_time = end_time - start_time
            if(t=='None'):
                print(f"\r推理时间: {processing_time:.4f} 秒",f'结果：{t}',end='')
            else:
                print(f"\r推理时间: {processing_time:.4f} 秒",f'结果：{t}')
                # print()


            # 显示结果
            # cv2.imshow("摄像头画面", current_frame)

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            time.sleep(0.01)  # 如果没有新帧，稍微等待一下

    # cv2.destroyAllWindows()

# 创建并启动线程
cap = cv2.VideoCapture('/dev/arm0')
capture_thread = threading.Thread(target=capture_frames)
inference_thread = threading.Thread(target=inference)

capture_thread.start()
inference_thread.start()

capture_thread.join()
inference_thread.join()
