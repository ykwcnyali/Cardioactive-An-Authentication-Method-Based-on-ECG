import serialimport timeimport numpy as np# 设置串口名称（例如 COM3 或 /dev/ttyACM0）# 根据你的操作系统和硬件选择正确的串口port = 'COM4'  # WindowsSample_Time = 100 # 设置采样时间(s)# port = '/dev/ttyACM0'  # Linux with USB-serial converterrawdata = []# 设置波特率baudrate = 9600# 打开串口ser = serial.Serial(port, baudrate)print("连接到串口: " + port)start_time = time.time()try:    while time.time()-start_time < Sample_Time+2:        # 读取串口数据        data = ser.readline()        if data and time.time()>=start_time+1:            # 打印读取到的数据            print(data)            rawdata.append(data)except KeyboardInterrupt:    print("关闭串口连接")    ser.close()export_data = rawdata[88:]print(export_data)print('len:',len(export_data))np.savetxt('array4.txt', rawdata, fmt='%s')