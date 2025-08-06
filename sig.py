import RPi.GPIO as GPIO
import time

def init():
    """
    初始化GPIO引脚
    设置GPIO模式为BOARD模式，并配置16号引脚为输出模式
    """
    GPIO.setmode(GPIO.BOARD)
    # 设置16号引脚为输出模式
    GPIO.setup(16, GPIO.OUT)

def turn_on():
    """
    点亮LED
    用16号引脚输出一个高电平，使LED亮起
    """
    GPIO.output(16, GPIO.HIGH)

def turn_off():
    """
    熄灭LED
    用16号引脚输出一个低电平，使LED熄灭
    """
    GPIO.output(16, GPIO.LOW)

def release():
    """
    释放GPIO资源
    清理所有GPIO设置，释放引脚资源
    """
    GPIO.cleanup()
