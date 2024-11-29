import bluetooth
import time

# Arduino的蓝牙设备名称和服务UUID
arduino_name = "myname" #00:24:03:01:08:A6 our device
GESTURE_NUM = 6

# # 连接到Arduino蓝牙设备
# def connect_to_arduino():
#     print("正在搜索 Arduino...")
#     nearby_devices = bluetooth.discover_devices()
#     for addr in nearby_devices:
#         if arduino_name == bluetooth.lookup_name(addr):
#             print(f"找到 Arduino: {arduino_name}, 地址: {addr}")
#             return addr
#     print("未找到 Arduino")
#     return None

# 连接
target_address = "00:24:03:01:08:A6"
port = 1

# 蓝牙套接字、连接
sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
sock.connect((target_address, port))
curr_time = time.time()
command = '0'
sock.send(command.encode())
while True:
    # 读取数据
    command = '1'
    sock.send(command.encode())
    data = sock.recv(10000).decode('utf-8')
    print(data)
    # use_time = time.time()-curr_time
    # if use_time >= 2.5:break

