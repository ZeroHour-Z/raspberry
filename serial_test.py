import serial

# 初始化串口通信
# 连接到/dev/ttyAMA0，波特率14400，超时0.5秒
port = serial.Serial(port='/dev/ttyAMA0', baudrate=14400, timeout=0.5)

# 准备发送的测试数据
text = f'c8d'
l = port.write(str(text).encode('utf-8'))
print(l, text)

# 循环发送数据直到收到确认信号
while True:
    l = port.write(str(text).encode('utf-8'))
    r = port.read(2)
    print(r)
    if r.decode('utf-8') == 'ok':
        break