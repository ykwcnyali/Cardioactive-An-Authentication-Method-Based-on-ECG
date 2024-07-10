import numpy as npimport osimport matplotlib.pyplot as pltfrom scipy.signal import butter, lfilter, freqz, find_peaksfrom pdf2image import convert_from_pathfrom PIL import Imageimport cv2import numpy as npimport matplotlib.pyplot as plt#----------------------------------------------------------------------------------------def lowpass_filter(data, Fs, cutoff_freq, order=5):    # 设计低通滤波器    b, a = butter(order, cutoff_freq/(Fs/2), btype='low', analog=False)    # 应用滤波器    y = lfilter(b, a, data)    return y#----------------------------------------------------------------------------------------#--------------------------CHANGE SUBJECT NAME HERE!!!!!!!-------------------------------#Basic infosubject_name = 'lyt'Fs = 195Ts = 1/Fs#----------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------#Image croppingimages = convert_from_path(os.getcwd()+f'\data\{subject_name}.pdf', dpi=200)  # dpi参数决定图片的清晰度for i, image in enumerate(images):    image.save(f'{subject_name}.png', 'PNG')  # 保存图片image_path = subject_name + '.png'image = Image.open(image_path)cropped_image = image.crop((150, 900, 2070, 1125))cropped_image.save(f"cropped_image_{subject_name}.jpg")#----------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------#Extract data from imageimage_path = f'cropped_image_{subject_name}.jpg'image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)image = cv2.bitwise_not(image)# 反转图像颜色（假设心电图是黑色背景白色波形）_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)# 二值化图像height, width = binary_image.shape# 初始化数组来存储波形点r1 = np.zeros(width, dtype=int)for col in range(width):# 遍历每一列，找到波形的最亮点    column_data = binary_image[:, col]    max_index = np.argmax(column_data)  # 找到最亮点的行号    r1[col] = max_indexprint(r1[:100])# 打印数组的一部分以确认lst = 0for i in range(len(r1)):    if(lst ==0):        if(r1[i]!=0 and i-lst>1):           lst=i    elif(r1[i]!=0):        print(i,r1[i],lst,r1[lst],i-lst)        break# 绘制波形plt.plot(r1)plt.title("ECG Waveform")plt.xlabel("Time (pixel index)")plt.ylabel("Amplitude (pixel row)")plt.gca().invert_yaxis()  # 反转Y轴，使得上方的像素值较小plt.show()#----------------------------------------------------------------------------------------#----------------------------------------------------------------------------------------r1 = -r1data_min = np.min(r1)data_max = np.max(r1)normalized_data = (r1 - data_min) / (data_max - data_min)print(r1)# 设定截止频率cutoff_freq = 50threshold = 0.8threshold1 = 0.7y_min = -0.7y_max = 1filtered_data = lowpass_filter(normalized_data, Fs, cutoff_freq)# 应用低通滤波器# 使用find_peaks函数来寻找峰值peaks, _ = find_peaks(filtered_data, height=0.5)  # height参数可以根据实际情况调整t=np.linspace(0,Ts*len(r1),len(r1))plt.subplot(2,1,1)plt.plot(t,r1)plt.xlim(0,5)plt.subplot(2,1,2)plt.plot(t,filtered_data)plt.plot(peaks/Fs, filtered_data[peaks], 'x', color='r', markersize=10)for i in range(len(peaks)):    plt.annotate('R',(peaks[i]/Fs, filtered_data[peaks[i]]+0.05))plt.show()ecg_data = filtered_data[50:]# 设置寻找峰值的距离和高度阈值distance = 0.5*Fsheight_threshold = -65# 寻找R波峰值peaks_r, _ = find_peaks(ecg_data, distance=distance, height=height_threshold)# 寻找S波峰值peaks_s = []for r_peak in peaks_r:    s_search_range = ecg_data[r_peak:]  # 限定在R波之后搜索S波    s_peak, _ = find_peaks(-1*s_search_range)    if len(s_peak) > 0:        peaks_s.append(s_peak[0] + r_peak)# 寻找T波峰值peaks_t = []height_threshold_t = -0.3for i in range(len(peaks_s)-1):    t_search_range = ecg_data[peaks_s[i]:peaks_r[i+1]]  # 限定在R波之后搜索T波    t_peak, _ = find_peaks(t_search_range, distance=peaks_r[i+1]-peaks_r[i]+50)    if len(t_peak) > 0:        peaks_t.append(t_peak[0] + peaks_s[i])# 寻找P波峰值peaks_p = []peaks_findp = []for i in range(len(peaks_t)):    peaks_findp.append(peaks_t[i]+7)height_threshold_p = -0.4for i in range(len(peaks_s)-1):    st = peaks_r[i+1]+(peaks_r[i+1]-peaks_r[i])/1.2    p_search_range = ecg_data[peaks_r[i]+(int)((peaks_r[i+1]-peaks_r[i])*2/3):peaks_r[i+1]]  # 限定在R波之前搜索P波    print('Searching', peaks_findp[i], peaks_r[i+1])    p_peak, _ = find_peaks(p_search_range, distance=peaks_r[i+1]-peaks_r[i]+50)    if len(p_peak) > 0:        peaks_p.append(p_peak[-1] + peaks_r[i]+(int)((peaks_r[i+1]-peaks_r[i])*2/3))        print(peaks_p[len(peaks_p)-1])# 寻找Q波峰值peaks_q = []for r_peak in peaks_r:    q_search_range = ecg_data[:r_peak]  # 限定在R波之前搜索Q波    q_peak, _ = find_peaks(-q_search_range)    if len(q_peak) > 0:        peaks_q.append(q_peak[-1])# 绘制心电图和标记的波峰plt.figure(figsize=(12, 6))plt.plot(ecg_data)plt.plot(peaks_r, ecg_data[peaks_r], 'ro', label='R peaks')plt.plot(peaks_p, ecg_data[peaks_p], 'go', label='P peaks')plt.plot(peaks_q, ecg_data[peaks_q], 'bo', label='Q peaks')plt.plot(peaks_s, ecg_data[peaks_s], 'co', label='S peaks')plt.plot(peaks_t, ecg_data[peaks_t], 'mo', label='T peaks')plt.title("Raw segmented result")plt.show()start_q = -1start_r = -1start_s = -1start_t = -1for i in range(len(peaks_p)):    if(peaks_p[i]<peaks_t[len(peaks_t)-1]):        end_p = ifor i in range(len(peaks_q)):    if(peaks_q[i]>peaks_p[0] and start_q==-1):        start_q = i        print(peaks_q[i])    if(peaks_q[i]<peaks_t[len(peaks_t)-1]):        end_q = ifor i in range(len(peaks_r)):    if(peaks_r[i]>peaks_p[0] and start_r==-1):        start_r = i        print(peaks_r[i])    if(peaks_r[i]<peaks_t[len(peaks_t)-1]):        end_r = ifor i in range(len(peaks_s)):    if(peaks_s[i]>peaks_p[0] and start_s==-1):        start_s = i        print(peaks_s[i])    if(peaks_s[i]<peaks_t[len(peaks_t)-1]):        end_s = ifor i in range(len(peaks_t)):    if(peaks_t[i]>peaks_p[0] and start_t==-1):        start_t = i        print(peaks_t[i])end_t = min([end_s-start_s,end_r-start_r,end_q-start_q,end_p+1,len(peaks_t)-1])+start_t#end_t = len(peaks_t)print(end_s)print(end_r)print(end_q)print(end_p)print(end_t)print(start_s)print(start_r)print(start_q)print(start_t)# 绘制心电图和标记的波峰plt.figure(figsize=(12, 6))plt.plot(ecg_data)plt.plot(peaks_r[start_r:end_r+1], ecg_data[peaks_r[start_r:end_r+1]], 'ro', label='R peaks')plt.plot(peaks_p[:end_p+1], ecg_data[peaks_p[:end_p+1]], 'go', label='P peaks')plt.plot(peaks_q[start_q:end_q+1], ecg_data[peaks_q[start_q:end_q+1]], 'bo', label='Q peaks')plt.plot(peaks_s[start_s:end_s+1], ecg_data[peaks_s[start_s:end_s+1]], 'co', label='S peaks')plt.plot(peaks_t[start_t:end_t+1], ecg_data[peaks_t[start_t:end_t+1]], 'mo', label='T peaks')#保存峰值坐标np.savetxt(f'peak_p_{subject_name}.txt', peaks_p[:end_p+1], fmt='%s')np.savetxt(f'peak_q_{subject_name}.txt', peaks_q[start_q:end_q+1], fmt='%s')np.savetxt(f'peak_r_{subject_name}.txt', peaks_r[start_r:end_r+1], fmt='%s')np.savetxt(f'peak_s_{subject_name}.txt', peaks_s[start_s:end_s+1], fmt='%s')np.savetxt(f'peak_t_{subject_name}.txt', peaks_t[start_t:end_t+1], fmt='%s')plt.legend()plt.grid()plt.title("Selected data")plt.show()#----------------------------------------------------------------------------------------