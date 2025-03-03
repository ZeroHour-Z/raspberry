import cv2
cam=cv2.VideoCapture('/dev/body0')
r,im=cam.read()
cv2.imwrite('a.jpg',im)
# cv2.waitKey(0)