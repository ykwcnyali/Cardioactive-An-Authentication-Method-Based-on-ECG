# **一种基于心电图的身份认证系统**

本项目利用心电图(ECG)数据，实现身份认证

其中，所有测试数据均利用apple watch ultra采集，以pdf格式存储

通过`\main.py`实现原始数据的处理与特征提取

将生成的`peaks_x_subject-name.txt`文件(其中x为p,q,r,s,t之一)导入`\Classification\`文件夹可作为训练数据与测试数据

通过`\Classification\Ectract_features.py`对数据进行训练，并输出测试数据的测试结果
