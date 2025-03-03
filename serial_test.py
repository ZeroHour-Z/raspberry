import serial
port=serial.Serial(port='/dev/ttyAMA0',baudrate=14400,timeout=0.5)
text=f'c8d'
l=port.write(str(text).encode('utf-8'))
print(l,text)
while(True):

  l=port.write(str(text).encode('utf-8'))
  r=port.read(2)
  print(r)
  if(r.decode('utf-8')=='ok'):
    break