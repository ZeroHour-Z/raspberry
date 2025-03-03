import RPi.GPIO as GPIO
import time
 
# 设置编码方式
def init():
  GPIO.setmode(GPIO.BOARD)
  
  # 设置GPIO引脚
  GPIO.setup(16, GPIO.OUT)
def turn_on():
# 用16号引脚输出一个高电平，灯亮
  GPIO.output(16, GPIO.HIGH)
 
def turn_off():
  GPIO.output(16, GPIO.LOW)
def release():
  GPIO.cleanup()
