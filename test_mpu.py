# find error of our mpu6050 in accel and gyro

import pandas as pd
from Arduino import Arduino
import time
import numpy as np
#set arduino board
board = Arduino('38400',port="COM3") #plugged in via USB, serial com at rate 9600
acc_list = []
# constant
GRESTURE_NUM = 5
aog = 8192.0 # 测量范围是+-4g
gyr = 131.0 # 测量范围是+-250°/s

# final data
acc_final_data = []
def getAccDegree(Acc_array):
    """
    calculate accAngleX and accAngleY for a hand gesture
    :param Acc_array:
    :return: res_array
    """
    test_Degree_array = np.zeros((Acc_array.shape[0], Acc_array.shape[1] - 1))
    test_Degree_array[:, 0] = np.arctan(
        Acc_array[:, 1] / np.sqrt(np.power(Acc_array[:, 0], 2) + np.power(Acc_array[:, 2], 2))) * 180 / np.pi
    test_Degree_array[:, 1] = np.arctan(
        -1 * Acc_array[:, 0] / np.sqrt(np.power(Acc_array[:, 1], 2) + np.power(Acc_array[:, 2], 2))) * 180 / np.pi
    res_array = test_Degree_array
    return res_array


def getAccError(Acc_array,iter_num):
    """
    calculate error of accel data(to calculate roll and pitch), yaws is not included by device limit
    :param Acc_array
    iter_num: how many values are collected
    :return: res_array
    """
    test_Error_array = np.zeros((Acc_array.shape[0],Acc_array.shape[1]-1))
    test_Error_array[:,0] = np.arctan(Acc_array[:,1] / np.sqrt(np.power(Acc_array[:,0],2) + np.power(Acc_array[:,2],2))) *180 / np.pi
    test_Error_array[:,1] = np.arctan(-1 * Acc_array[:,0] / np.sqrt(np.power(Acc_array[:,1],2) + np.power(Acc_array[:,2],2))) *180 / np.pi
    test_Error_array = test_Error_array.T
    res_array = np.zeros((2,))
    res_array[0] = np.sum(test_Error_array[0]) / iter_num
    res_array[1] = np.sum(test_Error_array[1]) / iter_num
    return res_array

def getGyroError(Gyro_array,iter_num):
    """
    calculate error of Gyro data, which is used to complement Gyro for 3 axis
    :param Gyro_array:
    :param iter_num:
    :return:res_array
    """
    Gyro_array = Gyro_array.T
    res_array = np.zeros((3,))
    res_array[0] = np.sum(Gyro_array[0]) / iter_num
    res_array[1] = np.sum(Gyro_array[1]) / iter_num
    res_array[2] = np.sum(Gyro_array[2]) / iter_num
    return res_array


# main
# mpu原始数据测量
for i in range(GRESTURE_NUM):
    # accelerometer data
    accelXs = []
    accelYs = []
    accelZs = []
    GyroXs = []
    GyroYs = []
    GyroZs = []
    elapseTimes = []
    #start recording
    board.digitalWrite(13, "HIGH")  # indicate user can do gesture now
    time.sleep(1)
    board.digitalWrite(13, "LOW")
    # time setting
    start_time = time.time()
    while True:
        acc_raw_data = board.mpuRead().split(",")
        accelXs.append(acc_raw_data[0])
        accelYs.append(acc_raw_data[1])
        accelZs.append(acc_raw_data[2])
        GyroXs.append(acc_raw_data[3])
        GyroYs.append(acc_raw_data[4])
        GyroZs.append(acc_raw_data[5])
        use_time = time.time()-start_time
        if use_time >= 2.5: break;
    # # end recording
    # board.digitalWrite(12, "HIGH")  # indicate user can do gesture now
    # time.sleep(1)
    # board.digitalWrite(12, "LOW")

    # 处理数据:
    acc_raw_data = np.zeros((len(accelXs),6))
    acc_raw_data[:, 0] = (np.array(accelXs,dtype=float)) / 8192.0
    acc_raw_data[:, 1] = (np.array(accelYs,dtype=float)) / 8192.0
    acc_raw_data[:, 2] = (np.array(accelZs,dtype=float)) / 8192.0
    acc_raw_data[:, 3] = (np.array(GyroXs,dtype=float)) / 131.0 + 0.0652
    acc_raw_data[:, 4] = (np.array(GyroYs,dtype=float)) / 131.0 - 0.0096
    acc_raw_data[:, 5] = (np.array(GyroZs,dtype=float)) / 131.0 + 0.542
    # 提取加速度数据
    Accels = acc_raw_data[:,:3]
    # 计算roll 和 pitch
    acc_Angles = getAccDegree(acc_raw_data[:,:3])
    gyro_Angles = acc_raw_data[:,3:5]
    Angles = 0.04 * gyro_Angles + 0.96 * acc_Angles # first column: roll, second column pitch

    # final data written in DataFrame
    acc_data = np.concatenate((Accels,Angles),axis=1)
    columns = ["accelX: g","accelY: g","accelZ: g","roll: °","pitch: °"]
    acc_data_df =pd.DataFrame(acc_data,columns=columns)
    zero_df = pd.DataFrame(np.zeros((1,5)),columns=columns)
    acc_data_df = pd.concat((acc_data_df,zero_df))
    acc_final_data.append(acc_data_df)

    # delay
    time.sleep(1)

acc_final_data = pd.concat(acc_final_data)
path = ".\\acc_data\\acc_output3.csv"
acc_final_data.to_csv(path,index=False)




# # 测量error
# for i in range(500):
#     mpu_data_str = board.mpuRead().split(",")
#     mpu_data = np.array(mpu_data_str,dtype=float)
#     accl_data = mpu_data[:3] / aog
#     gyro_data = mpu_data[3:] / gyr
#     mpu_data = np.concatenate((accl_data,gyro_data))
#     acc_list.append(mpu_data)
#     time.sleep(0.33)
# #500 times data
# mpu_array = np.array(acc_list)
# # Acc error
# AccError = getAccError(mpu_array[:3],500)
# GyroError = getGyroError(mpu_array[3:],500)
# print(AccError,GyroError,sep='\n')





