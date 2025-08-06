import cv2

# 初始化身体摄像头
cam = cv2.VideoCapture('/dev/body0')

# 读取一帧图像
r, im = cam.read()

# 保存图像到文件
cv2.imwrite('a.jpg', im)