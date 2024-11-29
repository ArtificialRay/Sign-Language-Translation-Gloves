import pandas as pd
from Arduino import Arduino
import time
import numpy as np
#set arduino board
board = Arduino('38400',port="COM3")
flexSensorPins = [14,15,16,17,18]
# constant to find voltage and resistance
VCC = 5
R_DIV = 22000.0
# constant of flex2.2
flatResistance_22 = 28000.0
bendResistance_22 = 40000.0
# constant of flex4.4
flatResistance_44 = 11000.0
bendResistance_44 = 24000.0
GESTURE_NUM = 5
# final data
flexs_final_data = []

def map_range(value,in_min, in_max, out_min, out_max):
    # define scaling function(same functionality as map() in arduino ide)
    return (value - in_min) / (in_max - in_min) * (out_max-out_min)


for i in range(GESTURE_NUM):
    thumbs = []
    index_fingers = []
    middle_fingers = []
    forth_fingers = []
    little_fingers = []
    # start recording
    board.digitalWrite(13,"HIGH")
    time.sleep(1)
    board.digitalWrite(13,"LOW")
    start_time = time.time()
    while True:
        thumb = board.analogRead(flexSensorPins[0]) # thumb
        index_finger = board.analogRead(flexSensorPins[1]) # index finger
        middle_finger = board.analogRead(flexSensorPins[2]) # middle finger
        forth_finger = board.analogRead(flexSensorPins[3]) # forth finger
        little_finger = board.analogRead(flexSensorPins[4]) # little finger
        thumbs.append(thumb)
        index_fingers.append(index_finger)
        middle_fingers.append(middle_finger)
        forth_fingers.append(forth_finger)
        little_fingers.append(little_finger)
        use_time = time.time()-start_time
        if use_time >= 2.5: break
    Voltage_flexs = np.zeros((len(thumbs), 5))
    Voltage_flexs[:, 0] = np.array(thumbs) * VCC / 1023.0
    Voltage_flexs[:, 1] = np.array(index_fingers) * VCC / 1023.0
    Voltage_flexs[:, 2] = np.array(middle_fingers) * VCC / 1023.0
    Voltage_flexs[:, 3] = np.array(forth_fingers) * VCC / 1023.0
    Voltage_flexs[:, 4] = np.array(little_fingers) * VCC / 1023.0
    Resistance_flexs = R_DIV *(VCC / Voltage_flexs - 1.0)
    Resistance_flexs_22 = np.concatenate(((Resistance_flexs[:,0]).reshape(-1,1),(Resistance_flexs[:,4]).reshape(-1,1)),axis=1)
    Resistance_flexs_44 = Resistance_flexs[:,1:4]
    Angle_flexs_22 = map_range(Resistance_flexs_22,flatResistance_22,bendResistance_22,0,90.0)
    Angle_flexs_44 = map_range(Resistance_flexs_44,flatResistance_44,bendResistance_44,0,90.0)
    Angle_flexs = np.concatenate(((Angle_flexs_22[:,0]).reshape(-1,1),Angle_flexs_44,(Angle_flexs_22[:,1]).reshape(-1,1)),axis=1)
    flexs_data = np.concatenate((Voltage_flexs,Angle_flexs),axis=1)
    columns = ["thumb_Voltage","index_Voltage","middle_Voltage","forth_Voltage","little_Voltage","thumb_Degree","index_Degree","middle_Degree","forth_Degree","little_Degree"]
    flexs_data_df = pd.DataFrame(flexs_data,columns=columns)
    zero_df = pd.DataFrame(np.zeros((1, 10)), columns=columns)
    flexs_data_df = pd.concat((flexs_data_df, zero_df))
    flexs_final_data.append(flexs_data_df)

    #delay
    time.sleep(1)
flexs_final_data = pd.concat(flexs_final_data)
path = ".\\flex_data\\flex_output1.csv"
flexs_final_data.to_csv(path,index=False)