import time
log=open(f'/home/pi/Documents/{time.time()}.txt','w+')
print(time.ctime(),file=log)
log.close()
try:
  import cv2
  import numpy as np
  import QRread
  import CrossDetect
  import serial
  import QRread
  import threading
  import ColorDetect
  import onnxruntime
  import time
  import sig
except ImportError as e:
  print(e)
  exit()
sig.init()
# log=open(f'{time.ctime()}.txt','w+')
model=None#提前加载模型
# print('启动串口',file=log)
port=serial.Serial(port='/dev/ttyAMA0',baudrate=14400,timeout=0.5)
# print(port.is_open,file=log)
# print('启动摄像头',file=log)
cam_body=cv2.VideoCapture('/dev/body0')
cam_arm=cv2.VideoCapture('/dev/arm0')
cam_arm.set(cv2.CAP_PROP_BUFFERSIZE,3)
cam_body.set(cv2.CAP_PROP_BUFFERSIZE,3)
try:
  r,_=cam_arm.read()
  r2,_=cam_body.read()
  # print('arm',r,'body',r2,file=log)
  if(not r or not r2):
    n=0
    while(n<5):
      cam_body=cv2.VideoCapture('/dev/body0')
      cam_arm=cv2.VideoCapture('/dev/arm0')
      r,_=cam_arm.read()
      r2,_=cam_body.read()
      if(r and r2):
        break

except Exception as e:
  # print(e,file=log)
  exit()
  
session_options=onnxruntime.SessionOptions()
session_options.graph_optimization_level=onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
model_path=r"better_t2.onnx"
# print('加载模型',file=log)
model=onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'],sess_options=session_options)
def call(stat,data):
  dic={'ho':17,'lv':18,'la':19,'ba':20,'he':21}
  global port
  if(not port.is_open):
    print('未打开')
    return
  if(stat==0 or stat==2):
    # text=f'c{int(data):0>2d}d'
    text=chr(ord('A')+int(data)-1)
    l=port.write(str(text).encode('utf-8'))
    print(l,text)
  elif(stat==1):
    # text=f'scol{str(data)}e'
    if(data=='err'):
      return
    l=port.write(chr(ord('A')+int(dic[data])-1).encode('utf-8'))
    print(l,chr(ord('A')+int(dic[data])-1))

  pass
def main(stat=0):
  color_cnt=0
  target=[]
  while(True):
    if(stat==0):#task1 QR
      tqr1=threading.Thread(target=QRread.capture_frames,args=(cam_body,))
      tqr2=threading.Thread(target=QRread.inference,args=(call,stat,))
      # ready,img=cam_body.read()
      # qr_data=QRread.main(img)
      # print('reading',qr_data)
      # if(qr_data!=''):
        # call(stat,qr_data)
      tqr1.start()
      tqr2.start()
      while(True):
        r=port.read(2)
        print('read',r)
        if(r.decode('utf-8')=='ok' or r.decode('utf-8')=='ko'):#ko是应对一些奇怪的情况
          sig.turn_on()
          time.sleep(1)
          sig.turn_off()
          QRread.flag=False
          tqr1.join()
          tqr2.join()
          stat=1
          break
        time.sleep(0.01)
    if(stat==2):#task1 QR
      QRread.flag=True
      tqr3=threading.Thread(target=QRread.capture_frames,args=(cam_body,))
      tqr4=threading.Thread(target=QRread.inference,args=(call,stat,))
      # ready,img=cam_body.read()
      # qr_data=QRread.main(img)
      # print('reading',qr_data)
      # if(qr_data!=''):
        # call(stat,qr_data)
      tqr3.start()
      tqr4.start()
      while(True):
        r=port.read(2)
        print('read',r)
        if(r.decode('utf-8')=='ok' or r.decode('utf-8')=='ko'):#ko是应对一些奇怪的情况
          sig.turn_on()
          time.sleep(1)
          sig.turn_off()
          QRread.flag=False
          tqr3.join()
          tqr4.join()
          print('finish')
          cam_body.release()
          exit()
          break
        time.sleep(0.01)
    elif(stat==1):#color
      t1=threading.Thread(target=ColorDetect.capture_frames,args=(cam_arm,))
      t2=threading.Thread(target=ColorDetect.inference,args=(model,call,stat,0.7,))
      t1.start()
      t2.start()
      det=set()
      while(True):
        r=port.read(1)
        if(r.decode('utf-8')==''):
          continue
        if(r.decode('utf-8')=='Z'):
          ColorDetect.pause_flag=False
        else:
          ColorDetect.pause_flag=True
          sig.turn_on()
          time.sleep(0.5)
          sig.turn_off()
          det.add(r.decode('utf-8'))
          print('copy',r.decode('utf-8'))
          print(det)
          if(r.decode('utf-8')=='Y'):
            ColorDetect.flag=False
            ColorDetect.pause_flag=False
            t1.join()
            t2.join()
            stat=2
            break
          if(det==set(['S','Q','T','U','R'])):
            ColorDetect.pause_flag=False
            ColorDetect.flag=False
            t1.join()
            t2.join()
            stat=3
            break
        
      cam_arm.release()
    elif(stat==3):
      while(True):
        r=port.read(1)
        if(r.decode('utf-8')=='Y'):
          stat=2
          break

if __name__ == '__main__':
  sig.turn_on()
  time.sleep(0.5)
  sig.turn_off()
  time.sleep(0.5)
  sig.turn_on()
  time.sleep(0.5)
  sig.turn_off()
  main()
  sig.release()
