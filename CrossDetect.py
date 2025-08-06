import cv2
import math
import numpy as np

# ed = cv2.ximgproc.createEdgeDrawing()
# EDParams = cv2.ximgproc_EdgeDrawing_Params()
# EDParams.MinPathLength = 50     # try changing this value between 5 to 1000
# EDParams.PFmode = False         # defaut value try to swich it to True
# EDParams.MinLineLength = 25     # try changing this value between 5 to 100
# EDParams.NFAValidation = False   # defaut value try to swich it to False
# ed.setParams(EDParams)

def getAngle(con, img=None):
    """
    计算轮廓的角度
    参数:
        con: 轮廓
        img: 图像（用于绘制）
    返回:
        img: 绘制了角度信息的图像
    """
    M = cv2.moments(con)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    con2 = np.squeeze(con)
    cv2.circle(img, (cX, cY), 3, (200, 250, 200), -1)
    d1 = (con2[0][0] - con2[1][0]) ** 2 + (con2[0][1] - con2[1][1]) ** 2
    d2 = (con2[0][0] - con2[-1][0]) ** 2 + (con2[0][1] - con2[-1][1]) ** 2
    
    if d1 > d2:  # 0索引点在左上角
        k = -float(con2[-3][1] - con2[2][1]) / (con2[-3][0] - con2[2][0])
        cv2.line(img, (con2[2][0], con2[2][1]), (con2[-3][0], con2[-3][1]), (200, 200, 150), 1)
        cv2.line(img, (con2[2][0], con2[2][1]), (con2[-3][0], con2[2][1]), (200, 200, 150), 1)
        ang = math.atan(k)
        ang = math.degrees(ang)
        cv2.putText(img, str(ang), (cX, cY), 0, 1, (100, 200, 100), 2)
    else:  # 0索引在右上角
        k = -float(con2[-2][1] - con2[3][1]) / (con2[-2][0] - con2[3][0])
        ang = math.atan(k)
        ang = math.degrees(ang)
        cv2.putText(img, str(ang), (cX, cY), 0, 1, (100, 200, 100), 2)
        pass
    # return [(cX,cY),ang]
    # print(ang)
    return img

# def getCrossED(img):
#   gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#   gray=cv2.GaussianBlur(gray,(5,5),1)
#   _, gray = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
#   k = np.ones((10, 10), np.uint8)
#   gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN,k)
#   gray=255-gray
#   cv2.imshow('b',gray)
#   ed.detectEdges(gray)
#   lines = ed.detectLines()
#   if lines is not None: # Check if the lines have been found and only then iterate over these and add them to the image
#     lines = np.uint16(np.around(lines))
#   else:
#     return img
#   for i in range(len(lines)):
#     cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
#   return img

def getCross(img):
    """
    检测十字线标记
    参数:
        img: 输入图像(BGR格式)
    返回:
        img: 绘制了检测结果的图像
    """
    # input:img(BGR)
    # output:[(cX,cY),ang]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.convertScaleAbs(gray)
    gray = cv2.GaussianBlur(gray, (7, 7), 0.7)
    _, gray = cv2.threshold(gray, 105, 255, cv2.THRESH_BINARY)
    k = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, k)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, k)
    gray = 255 - gray
    cv2.imshow('b', gray)
    # ed.detectEdges(gray)
    # lines = ed.detectLines()
    # if lines is not None: # Check if the lines have been found and only then iterate over these and add them to the image
    #   lines = np.uint16(np.around(lines))
    # for i in range(len(lines)):
    #   cv2.line(img, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 1, cv2.LINE_AA)
    con, hie = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    con2 = []
    
    for c in con:
        ct = cv2.approxPolyDP(c, 4, True)
        s = cv2.contourArea(ct)
        if s > 3000:  # 面积去除小干扰
            x, y, w, h = cv2.boundingRect(ct)
            sr = w * h
            if __name__ == '__main__':
                cv2.drawContours(img, [c], 0, (255, 0, 0), 1)
            if ((float(w) / h > 0.8 and float(w) / h < 1.2) and 
                abs(len(ct) - 12) < 7 and float(sr) / s < 3.5):  # 外接矩形宽高比和角点数量
                # img = cv2.rectangle(img, (x+w, y+h), (x, y), (255, 255, 0), 1, 0)
                con2.append(ct)
            else:
                cv2.putText(img, f'{float(w)/h:.3f},{abs(len(ct)-12)},{float(sr)/s:.3f}', 
                           (ct[0][0][0], ct[0][0][1]), 0, 1.5, (255, 0, 0), 2)
            # img = cv2.drawContours(img, [ct], 0, (0, 255, 255), 1, 0)
        # else:
        #   img = cv2.drawContours(img, [ct], 0, (255, 0, 0), 1, 0)
    
    if len(con2) > 1:
        print('multi contour')
        con = sorted(con2, key=lambda c: abs(len(c) - 12))
        con3 = [con[0]]
    elif len(con2) == 0:
        if __name__ == '__main__':
            return img
        else:
            raise FutureWarning('没写好')
    else:
        con3 = con2
    
    if len(con3[0]) > 12:
        con3[0] = cv2.approxPolyDP(con3[0], 4, True)
    
    if __name__ == '__main__':
        cv2.circle(img, (con3[0][0][0][0], con3[0][0][0][1]), 5, (200, 100, 200), -1)
        cv2.circle(img, (con3[0][1][0][0], con3[0][1][0][1]), 5, (100, 200, 200), -1)
    
    img = getAngle(con3[0], img)
    # data = getAngle(con3[0])
    if __name__ == '__main__':
        cv2.drawContours(img, con3, -1, (0, 0, 255), 1, 0)
        return img
    # return data
    raise FutureWarning('没写好')
    # return img
    pass

if __name__ == '__main__':
    # 测试代码
    n = 1
    # img = cv2.imread('cross3.jpg')
    # cam = cv2.VideoCapture(1)
    # cam = cv2.VideoCapture(r"C:\Users\XiYi\Documents\WeChat Files\wxid_d8llyvim5nzj22\FileStorage\Video\2024-08\faf3a8bb0de7c642957d213f3308686f.mp4")
    # cam = cv2.VideoCapture(r"C:\Users\XiYi\Documents\WeChat Files\wxid_d8llyvim5nzj22\FileStorage\Video\2024-09\03ae38d0edd54c875257088073f37bd6.mp4")
    # cam = cv2.VideoCapture(r"C:\Users\XiYi\Documents\WeChat Files\wxid_d8llyvim5nzj22\FileStorage\Video\2024-08\2d88c99a091e03b49f36f1f8408fe02a.mp4")
    # cam = cv2.VideoCapture('/dev/arm0')
    cam = cv2.VideoCapture(0)
    
    while n > 0:
        r, img = cam.read()
        print(r)
        if not r:
            break
        # img = cv2.resize(img, (640, 480))
        # img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
        # n -= 1
        img = getCross(img)
        cv2.imshow('cam', img)
        # cv2.waitKey(0)
        if cv2.waitKey(50) == ord('p'):
            pass
        if cv2.waitKey(20) == ord('q'):
            break
    
    cam.release()