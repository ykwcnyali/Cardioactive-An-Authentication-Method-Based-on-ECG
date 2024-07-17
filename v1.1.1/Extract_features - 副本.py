import numpy as np


# 读取文件数据的函数
def read_peaks(file_path):
    with open(file_path, 'r') as file:
        peaks = [float(line.strip()) for line in file.readlines()]
    return np.array(peaks)


# 提取特征的函数
def extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks):
    features = {}

    # Calculate distances
    features['R–P distance'] = np.abs(r_peaks - p_peaks)
    features['R–T distance'] = np.abs(r_peaks - t_peaks)
    features['R–Q distance'] = np.abs(r_peaks - q_peaks)
    features['R–S distance'] = np.abs(r_peaks - s_peaks)

    # Calculate widths
    features['P width'] = np.diff(p_peaks)
    features['T width'] = np.diff(t_peaks)

    features['S–T distance'] = np.abs(s_peaks - t_peaks)
    features['P–Q distance'] = np.abs(p_peaks - q_peaks)
    features['P–T distance'] = np.abs(p_peaks - t_peaks)

    return features


# 读取各个波峰位置文件
r_peaks = read_peaks('peak_r_zzy.txt')
p_peaks = read_peaks('peak_p_zzy.txt')
q_peaks = read_peaks('peak_q_zzy.txt')
s_peaks = read_peaks('peak_s_zzy.txt')
t_peaks = read_peaks('peak_t_zzy.txt')

# 提取特征
features = extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks)

# 打印每个特征的12组值
for key, value in features.items():
    print(f'{key}: {value}')


#----------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 读取文件数据的函数
def read_peaks(file_path):
 with open(file_path, 'r') as file:
  peaks = [float(line.strip()) for line in file.readlines()]
 return np.array(peaks)


# 提取特征的函数（不取平均值）
def extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks):
 features = {}

 # Calculate distances and widths (keeping all values)
 features['R_P'] = np.abs(r_peaks - p_peaks)
 features['R_T'] = np.abs(r_peaks - t_peaks)
 features['R_Q'] = np.abs(r_peaks - q_peaks)
 features['R_S'] = np.abs(r_peaks - s_peaks)
 features['P_width'] = np.diff(p_peaks) if len(p_peaks) > 1 else np.array([0])
 features['T_width'] = np.diff(t_peaks) if len(t_peaks) > 1 else np.array([0])
 features['S_T'] = np.abs(s_peaks - t_peaks)
 features['P_Q'] = np.abs(p_peaks - q_peaks)
 features['P_T'] = np.abs(p_peaks - t_peaks)

 return features


# 读取和提取每个人的ECG特征
def load_person_data(person):
 r_peaks = read_peaks(f'peak_r_{person}.txt')
 p_peaks = read_peaks(f'peak_p_{person}.txt')
 q_peaks = read_peaks(f'peak_q_{person}.txt')
 s_peaks = read_peaks(f'peak_s_{person}.txt')
 t_peaks = read_peaks(f'peak_t_{person}.txt')

 features = extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks)
 return features


# 定义人的ID和对应的标签
persons = ['hyf', 'ykw', 'zzy']
labels = {person: i for i, person in enumerate(persons)}

# 加载所有人的数据
data = []

for person in persons:
 features = load_person_data(person)
 max_len = max(len(feature) for feature in features.values())
 for key in features:
  features[key] = np.pad(features[key], (0, max_len - len(features[key])), 'constant')
 df = pd.DataFrame(features)
 df['label'] = labels[person]
 data.append(df)

# 合并所有人的数据
df = pd.concat(data, ignore_index=True)

# 打印特征数据
print("所有特征数据:")
print(df)

# 按标签分组，并移除每组的最后一行
df = df.groupby('label').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

# 打印处理后的特征数据
print("\n处理后的特征数据:")
print(df)

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n训练集特征 (X_train):")
print(X_train)
print("\n训练集标签 (y_train):")
print(y_train)
print("\n测试集特征 (X_test):")
print(X_test)
print("\n测试集标签 (y_test):")
print(y_test)

# 创建KNN分类器，设置k值
k = 2
knn = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn.fit(X_train, y_train)
print("\n模型训练完成。")

# 在测试集上进行预测
y_pred = knn.predict(X_test)
print("\n测试集预测结果 (y_pred):")
print(y_pred)

# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'\n模型准确率: {accuracy * 100:.2f}%')


#----------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# 读取文件数据的函数
def read_peaks(file_path):
 with open(file_path, 'r') as file:
  peaks = [float(line.strip()) for line in file.readlines()]
 return np.array(peaks)


# 提取特征的函数（不取平均值）
def extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks):
 features = {}
 features['R_P'] = np.abs(r_peaks - p_peaks)
 features['R_T'] = np.abs(r_peaks - t_peaks)
 features['R_Q'] = np.abs(r_peaks - q_peaks)
 features['R_S'] = np.abs(r_peaks - s_peaks)
 features['P_width'] = np.diff(p_peaks) if len(p_peaks) > 1 else np.array([0])
 features['T_width'] = np.diff(t_peaks) if len(t_peaks) > 1 else np.array([0])
 features['S_T'] = np.abs(s_peaks - t_peaks)
 features['P_Q'] = np.abs(p_peaks - q_peaks)
 features['P_T'] = np.abs(p_peaks - t_peaks)
 return features


# 移除异常值的函数
def remove_outliers(df, columns, threshold=2.5):
 for col in columns:
  mean = df[col].mean()
  std = df[col].std()
  print(f"Processing column {col}: mean={mean}, std={std}")  # Debug output
  outliers = df[(df[col] < mean - threshold * std) | (df[col] > mean + threshold * std)]
  num_outliers = len(outliers)
  print(f"Outliers in {col} ({num_outliers}):\n{outliers}")  # Debug output for outliers
  df = df[(df[col] >= mean - threshold * std) & (df[col] <= mean + threshold * std)]
 return df


# 读取和提取每个人的ECG特征
def load_person_data(person):
 r_peaks = read_peaks(f'peak_r_{person}.txt')
 p_peaks = read_peaks(f'peak_p_{person}.txt')
 q_peaks = read_peaks(f'peak_q_{person}.txt')
 s_peaks = read_peaks(f'peak_s_{person}.txt')
 t_peaks = read_peaks(f'peak_t_{person}.txt')
 features = extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks)
 return features


# 定义人的ID和对应的标签
persons = ['hyf', 'ykw', 'zzy']
labels = {person: i for i, person in enumerate(persons)}

# 加载所有人的数据
data = []

for person in persons:
 features = load_person_data(person)
 max_len = max(len(feature) for feature in features.values())
 for key in features:
  features[key] = np.pad(features[key], (0, max_len - len(features[key])), 'constant')
 df = pd.DataFrame(features)
 df['label'] = labels[person]
 data.append(df)

# 合并所有人的数据
df = pd.concat(data, ignore_index=True)

# 打印特征数据
print("所有特征数据:")
print(df)

# 按标签分组，并移除每组的最后一行
df = df.groupby('label').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

# 打印处理后的特征数据
print("\n处理后的特征数据:")
print(df)

# 去除异常值
columns = ['R_P', 'R_T', 'R_Q', 'R_S', 'P_width', 'T_width', 'S_T', 'P_Q', 'P_T']
df = remove_outliers(df, columns)

# 打印去除异常值后的特征数据
print("\n去除异常值后的特征数据:")
print(df)

# 分离特征和标签
X = df.drop('label', axis=1)
y = df['label']

# 将数据分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\n训练集特征 (X_train):")
print(X_train)
print("\n训练集标签 (y_train):")
print(y_train)
print("\n测试集特征 (X_test):")
print(X_test)
print("\n测试集标签 (y_test):")
print(y_test)

# 创建KNN分类器，设置k值
k = 3
knn = KNeighborsClassifier(n_neighbors=k)

# 训练模型
knn.fit(X_train, y_train)
print("\n模型训练完成。")

# 在测试集上进行预测
y_pred = knn.predict(X_test)
print("\n测试集预测结果 (y_pred):")
print(y_pred)

# 计算模型准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'\n模型准确率: {accuracy * 100:.2f}%')


# 加载测试数据并进行预测
def load_test_data():
 r_peaks = read_peaks('peak_r_test.txt')
 p_peaks = read_peaks('peak_p_test.txt')
 q_peaks = read_peaks('peak_q_test.txt')
 s_peaks = read_peaks('peak_s_test.txt')
 t_peaks = read_peaks('peak_t_test.txt')

 features = extract_ecg_features(r_peaks, p_peaks, q_peaks, s_peaks, t_peaks)
 max_len = max(len(feature) for feature in features.values())
 for key in features:
  features[key] = np.pad(features[key], (0, max_len - len(features[key])), 'constant')
 df = pd.DataFrame(features)

 return df


# 加载测试数据
test_df = load_test_data()

# 打印测试数据之前
print("\n去除异常值前的测试数据:")
print(test_df)

# 去除测试数据中的异常值
test_df = remove_outliers(test_df, columns)

test_df = test_df.iloc[:-1]

print("\n去除异常值后的测试数据:")
print(test_df)

# 使用模型进行预测
test_pred = knn.predict(test_df)

# 映射标签到人员
test_df['predicted_label'] = test_pred
test_df['predicted_person'] = test_df['predicted_label'].apply(
 lambda x: list(labels.keys())[list(labels.values()).index(x)])

# 打印每一条测试数据的预测结果
print("\n测试数据的分类结果:")
print(test_df)
test_df = test_df.iloc[:-1]