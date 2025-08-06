import cv2
import time
import pyzbar
import pyzbar.pyzbar
import numpy as np
import threading
# import matplotlib.pyplot as plt
# import zxing

# 可能需要现场调的参数
# QR码识别二值化阈值——环境越亮，阈值越低
QR_THRESH = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 105, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250]

# 全局变量
flag = True
frame = None
frame_lock = threading.Lock()

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
        ret, temp_frame = cap.read()
        if not ret:
            break
        with frame_lock:
            frame = temp_frame.copy()

def inference(call, stat):
    """
    QR码推理线程函数
    参数:
        call: 回调函数，用于发送识别结果
        stat: 当前状态
    """
    global frame
    
    while flag:
        if frame is not None:
            with frame_lock:
                current_frame = frame.copy()
            
            # 开始计时
            # start_time = time.time()
            # 进行推理（这里用一个模拟的时间延迟来表示推理过程）
            # t = method(current_frame, model, fliter)
            t = main(current_frame)
            if t != '':
                print('qr_read', t)
                while flag:
                    call(stat, t)
                    time.sleep(0.1)
            # 结束计时
            # end_time = time.time()
            # processing_time = end_time - start_time
            # print(f"推理时间: {processing_time:.4f} 秒")
        else:
            time.sleep(0.01)

def getQRCode(img, QR_ind=11):
    """
    传入彩色图片，尝试获取定位点，输出定位点轮廓列表
    若有三个定位点全部被找到，同时定位二维码
    
    参数:
        img: 输入图像
        QR_ind: QR码索引（默认11）
    
    返回:
        res: 定位点轮廓列表
        bound: 边界框
        code: 提取的二维码区域
        center: 中心点坐标
        QRsuccess: 是否成功找到二维码
        debug_img: 调试图像
    """
    global QR_THRESH
    
    # 图像预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (2, 2))  # 3也行，4不行
    gray = cv2.convertScaleAbs(gray)
    # gray = cv2.equalizeHist(gray)
    _, gray = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    
    # 查找轮廓
    con, hie = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    con2 = []
    
    # 轮廓近似
    for c in con:
        con2.append(cv2.approxPolyDP(c, 3, True))  # 越大，越棱角分明
    
    # tmp = img.copy()
    # cv2.drawContours(tmp, con, -1, (0, 0, 255), 1)
    # cvshow(tmp)
    
    have_son = []  # 有子轮廓的索引
    have_son2 = []
    con = con2
    
    # tmp = img.copy()
    # cv2.drawContours(tmp, con, -1, (0, 0, 255), 1)
    # cvshow(tmp)
    
    debug_img = gray
    # pltshow(gray)
    l = len(con)
    # out = open('t.txt', 'w')
    
    # 查找有子轮廓的轮廓
    for i in range(l):
        if (hie[0][i][2] != -1 and abs(len(con[i]) - 4) < 1 and hie[0][hie[0][i][2]][0] == -1):
            # 有子轮廓并且点少（接近正方形）且只有一个
            have_son.append(i)
    
    # print(have_son)
    # tc = [con[i] for i in have_son]
    # tmp = img.copy()
    # cv2.drawContours(tmp, tc, -1, (0, 0, 255), 1)
    # cvshow(tmp)
    # out.close()
    
    # 查找有孙子轮廓的轮廓
    for i in range(len(have_son)):
        if (hie[0][hie[0][have_son[i]][2]][2] != -1 and 
            hie[0][hie[0][hie[0][have_son[i]][2]][2]][2] == -1 and 
            hie[0][hie[0][have_son[i]][2]][0] == -1):
            # 子轮廓还有子轮廓且孙子没有子轮廓且子轮廓只有一个
            have_son2.append(have_son[i])
    
    res = [con[i] for i in have_son2]
    # cv2.drawContours(img, res, -1, (0, 0, 255), 1)
    # cvshow(img)
    
    bound = None
    center = []
    code = None
    QRsuccess = False
    
    if len(res) == 3:
        # 找到三个定位点，计算边界框
        left, right, down, top = 9999, -9999, -9999, 9999
        for i in range(3):
            center.append([(res[i][0][0][0] + res[i][1][0][0] + res[i][2][0][0] + res[i][3][0][0]) / 4,
                         (res[i][0][0][1] + res[i][1][0][1] + res[i][2][0][1] + res[i][3][0][1]) / 4])
            for j in range(4):
                left = min(left, res[i][j][0][0])
                top = min(top, res[i][j][0][1])
                right = max(right, res[i][j][0][0])
                down = max(down, res[i][j][0][1])
        
        # bound = np.array([[[left, down]], [[right, down]], [[right, top]], [[left, top]]])
        bound = np.array([[[left - 10, down + 10]], [[right + 10, down + 10]], 
                         [[right + 10, top - 10]], [[left - 10, top - 10]]])
        code = img[top - 10:down + 10, left - 10:right + 10].copy()
        QRsuccess = True
        # code = cv2.resize(code, (256, 256), interpolation=cv2.INTER_LINEAR)
    
    return res, bound, code, center, QRsuccess, debug_img  # 先不写旋转了

def illum(img):
    """
    图像光照调整函数
    参数:
        img: 输入图像
    返回:
        result: 调整后的图像
    """
    # img = cv2.imread("test2.jpg")
    # img = img[532:768, 0:512]
    img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(img_bw, 180, 255, 0)[1]
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    img_zero = np.zeros(img.shape, dtype=np.uint8)
    # img[thresh == 255] = 150
    
    for cnt in cnts:
        x, y, w, h = cv2.boundingRect(cnt)
        img_zero[y:y + h, x:x + w] = 255
    
    # cv2.imshow("mask", mask)
    mask = img_zero
    # cv2.imshow("mask", mask)
    result = cv2.illuminationChange(img, mask, alpha=1, beta=2)
    # cv2.imshow("result", result)
    # cv2.waitKey(0)
    return result

# qrl = open('/home/pi/Documents/qr.txt', 'w+')

def main(img):
    """
    主要的QR码识别函数
    参数:
        img: 输入彩色图像
    返回:
        str: 二维码内容字符串，如果识别失败返回空字符串
    """
    cv2.imwrite(f'/home/pi/Documents/qr.jpg', img)
    
    # 方法1：直接解码
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = pyzbar.pyzbar.decode(gray)
    if len(res) > 0:
        return str(res[0].data.decode('utf-8'))
    
    # 方法2：图像增强后解码
    gray2 = cv2.convertScaleAbs(gray)
    res = pyzbar.pyzbar.decode(gray2)
    if len(res) > 0:
        return str(res[0].data.decode('utf-8'))
    
    # 方法3：高斯模糊后解码
    gray2 = cv2.GaussianBlur(gray, (7, 7), 1.5)
    res = pyzbar.pyzbar.decode(gray2)
    if len(res) > 0:
        return str(res[0].data.decode('utf-8'))
    
    # 方法4：二值化后解码
    _, gray3 = cv2.threshold(gray2, 105, 255, cv2.THRESH_BINARY)
    res = pyzbar.pyzbar.decode(gray3)
    if len(res) > 0:
        return str(res[0].data.decode('utf-8'))
    
    # 方法5：形态学操作后解码
    k = np.ones((4, 4), np.uint8)
    gray4 = cv2.morphologyEx(gray3, cv2.MORPH_OPEN, k)
    res = pyzbar.pyzbar.decode(gray4)
    if len(res) > 0:
        return str(res[0].data.decode('utf-8'))
    
    # 方法6：基于定位点的精确识别
    con, bound, code, center, QRsuccess, debug_img = getQRCode(img)
    if QRsuccess:
        gray3 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray3 = cv2.convertScaleAbs(gray3)
        gray3 = cv2.resize(gray3, (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
        res = pyzbar.pyzbar.decode(gray3)
        if len(res) > 0:
            return str(res[0].data.decode('utf-8'))
    
    return ''

def main_old(img):
    """
    旧的QR码识别函数（已废弃）
    参数:
        img: 输入彩色图像
    返回:
        str: 二维码内容字符串
    """
    QRsuccess = False
    con, bound, code, center, QRsuccess, debug_img = getQRCode(img)
    
    if QRsuccess:
        cnt2 = 0
        res = {}
        while (len(res) == 0) and (cnt2 < 3):
            cnt2 += 1
            res = {}
            gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            while (len(res) == 0) and (cnt2 < 3):
                res = pyzbar.pyzbar.decode(gray)
                cnt2 += 1
                gray = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray)
                gray2 = gray
                gray2 = cv2.resize(gray2, (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
                res = pyzbar.pyzbar.decode(gray2)
            if cnt2 >= 3:
                return ''
        if cnt2 >= 3:
            cnt2 = 0
        return str(res[0].data.decode('utf-8'))
    else:
        cnt2 = 0
        res = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        while (len(res) == 0) and (cnt2 < 3):
            cnt2 += 1
            res = pyzbar.pyzbar.decode(gray)
        cnt2 = 0
        res = {}
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray)
        while (len(res) == 0) and (cnt2 < 3):
            cnt2 += 1
            res = pyzbar.pyzbar.decode(gray)
        if cnt2 >= 3:
            return ''
        return str(res[0].data.decode('utf-8'))

if __name__ == '__main__':
    # 测试代码
    cam = cv2.VideoCapture(0)
    # t = open('t.txt', 'w')
    # coder = cv2.QRCodeDetector()
    # coder_z = zxing.BarCodeReader()
    defalt_arg = 11  # 二值化默认参数
    err_cnt = 0
    
    while True:
        # time.sleep(1)
        ready, img = cam.read()
        print(main(img))
        # con, bound, code, center, QRsuccess, debug_img = getQRCode(img, defalt_arg)
        # con.append(bound)
        # if QRsuccess:
        #     # 拉普拉斯锐化
        #     img = cv2.blur(img, (3, 3))
        #     laplacian = cv2.Laplacian(img, cv2.CV_64F)
        #     laplacian = np.uint8(np.absolute(laplacian))
        #     img = cv2.addWeighted(img, 1, laplacian, 0.1, 0)
        #     cv2.drawContours(img, con, -1, (0, 0, 255), 1)
        #     gray = cv2.cvtColor(code, cv2.COLOR_BGR2GRAY)
        #     gray = cv2.convertScaleAbs(gray)
        #     gray2 = gray
        #     # gray = cv2.equalizeHist(gray)
        #     # _, gray2 = cv2.threshold(gray, defalt_arg, 255, cv2.THRESH_BINARY)
        #     gray2 = cv2.resize(gray2, (256, 256), interpolation=cv2.INTER_LINEAR_EXACT)
        #     # res, _, _ = coder.detectAndDecode(gray2)
        #     res = pyzbar.pyzbar.decode(gray2)
        #     '上面这个方法经过测试效果最好'
        #     # if len(res) == 0:  # 纯视觉定位到但是扫不出来，说明处理参数有误
        #     #     err_cnt += 1
        #     #     if err_cnt > 30:
        #     #         res, defalt_arg = autoArg(gray)  # 画蛇添足'
        #     #         pass
        #     # else:
        #     #     err_cnt = 0
        #     if len(res) != 0:
        #         print(str(res[0].data.decode('utf-8')))
        #     else:
        #         gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #         res = pyzbar.pyzbar.decode(gray2)
        #         if res:
        #             print(str(res[0].data.decode('utf-8')), 'plb')
        cv2.imshow('cam', img)
        # err_cnt = 0
        # else:
        #     gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #     res = pyzbar.pyzbar.decode(gray2)
        #     if res:
        #         print(str(res[0].data.decode('utf-8')), 'plb')
        #     cv2.imshow('cam', img)
        #     # err_cnt += 1
        #     # if err_cnt > 30:
        #     #     print('c')
        #     #     defalt_arg += 1
        #     # if defalt_arg >= len(QR_THRESH):
        #     #     defalt_arg = 0
        # if cv2.waitKey(1) == ord('p'):
        #     # getQRCode(img)
        #     pass
        if cv2.waitKey(1) == ord('q'):
            break
    # t.close()
    cam.release()
