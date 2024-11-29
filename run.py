import numpy as np
import pandas as pd
from Arduino import Arduino
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import serial
import speech


# set arduino board
board = Arduino('9600',port="COM3")
flexSensorPins = [54,55,56,57,60]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# dictionary:
str_result_dict = {1:"大家好",2:"我们",3:"是",4:"Group",5:"4",6:"0",7:"的",8:"项目",9:"手语识别手套",10:"谢谢",11:"点赞",12:"开",13:"关"}

def getAccDegree(Acc_array):
    """
    calculate accAngleX and accAngleY for a hand gesture
    :param Acc_array:
    :return: res_array
    """
    test_Degree_array = np.zeros((Acc_array.shape[0], Acc_array.shape[1] - 1))
    test_Degree_array[:, 0] = np.arctan(
        Acc_array[:, 1] / np.sqrt(np.power(Acc_array[:, 0], 2) + np.power(Acc_array[:, 2], 2))) * 180 / np.pi - 0.00856
    test_Degree_array[:, 1] = np.arctan(
        -1 * Acc_array[:, 0] / np.sqrt(np.power(Acc_array[:, 1], 2) + np.power(Acc_array[:, 2], 2))) * 180 / np.pi - 0.034
    res_array = test_Degree_array
    return res_array

# running:
while True:
    # flex sensors data
    thumbs = []
    index_fingers = []
    middle_fingers = []
    forth_fingers = []
    little_fingers = []
    # accelerometer data
    accelXs = []
    accelYs = []
    accelZs = []
    GyroXs = []
    GyroYs = []
    GyroZs = []
    # start recording
    board.digitalWrite(8, "HIGH")
    time.sleep(1)
    board.digitalWrite(8, "LOW")
    start_time = time.time()
    # make a flex sensor recording:
    while True:
        thumb = board.analogRead(flexSensorPins[0])  # thumb
        index_finger = board.analogRead(flexSensorPins[1])  # index finger
        middle_finger = board.analogRead(flexSensorPins[2])  # middle finger
        forth_finger = board.analogRead(flexSensorPins[3])  # forth finger
        little_finger = board.analogRead(flexSensorPins[4])  # little finger
        thumbs.append(thumb)
        index_fingers.append(index_finger)
        middle_fingers.append(middle_finger)
        forth_fingers.append(forth_finger)
        little_fingers.append(little_finger)

        # accelerometer recording
        acc_raw_data = board.mpuRead().split(",")
        accelXs.append(acc_raw_data[0])
        accelYs.append(acc_raw_data[1])
        accelZs.append(acc_raw_data[2])
        GyroXs.append(acc_raw_data[3])
        GyroYs.append(acc_raw_data[4])
        GyroZs.append(acc_raw_data[5])
        use_time = time.time() - start_time
        if use_time >= 3.5: break
    # flex_sensor_data:
    flex_data = np.array([thumbs,index_fingers,middle_fingers,forth_fingers,little_fingers])
    # accelerometer data:
    acc_raw_data = np.zeros((len(accelXs), 6))
    acc_raw_data[:, 0] = (np.array(accelXs, dtype=float)) / 8192.0
    acc_raw_data[:, 1] = (np.array(accelYs, dtype=float)) / 8192.0
    acc_raw_data[:, 2] = (np.array(accelZs, dtype=float)) / 8192.0
    acc_raw_data[:, 3] = (np.array(GyroXs, dtype=float)) / 131.0 + 0.0652
    acc_raw_data[:, 4] = (np.array(GyroYs, dtype=float)) / 131.0 - 0.0096
    acc_raw_data[:, 5] = (np.array(GyroZs, dtype=float)) / 131.0 + 0.542
    # accelerometer data:
    Accels = acc_raw_data[:, :3]
    # rolls & pitches
    acc_Angles = getAccDegree(acc_raw_data[:, :3])
    gyro_Angles = acc_raw_data[:, 3:5]
    Angles = 0.04 * gyro_Angles + 0.96 * acc_Angles  # first column: roll, second column pitch
    # final data written in DataFrame
    acc_data = np.concatenate((Accels, Angles), axis=1)
    final_data = np.concatenate((flex_data,acc_data),axis=1)
    # load model
    model_path = "BiLSTM_Attention_weights.pth"
    model = torch.load(model_path)
    model = model.to(device)
    # scaling data:
    scaler = MinMaxScaler(feature_range=(0,1))
    ges_data_scaled = scaler.fit_transform(final_data)

    # prediction:
    result = np.argmax(model(ges_data_scaled))

    # read the prediction result:
    speech.say(str_result_dict[result])
    print(str_result_dict[result])


