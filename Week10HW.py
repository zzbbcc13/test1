import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 匯入資料
file_path = r"C:/Users/USER/OneDrive/文件/大二上學期作業/程式作業/Week10作業/train.xlsx"
data = pd.read_excel(file_path)  # 使用 pandas 讀取 Excel 檔案

# 檢查缺失值
print("Missing values check:")  # 打印缺失值檢查訊息
print(data.isnull().sum())  # 打印每個欄位的缺失值數量

# 打印資料欄位名稱
print(data.columns)  # 打印資料框的欄位名稱

# 繪製長條圖
features = ['age', 'job', 'marital', 'education', 'loan']  # 要分析的特徵列表
for i, feature in enumerate(features):
    plt.figure(figsize=(10, 5))  # 設置圖形大小
    data[feature].value_counts().plot(kind='bar')  # 繪製長條圖
    plt.title(f'{feature} vs Subscription Count')  # 設置圖形標題
    plt.xlabel(feature)  # 設置 x 軸標籤
    plt.ylabel('Subscription Count')  # 設置 y 軸標籤
    if feature == 'age':  # 只調整第一張圖的 x 軸標籤間距
        plt.xticks(ticks=range(0, len(data[feature].value_counts()) * 2, 2), labels=data[feature].value_counts().index)
    plt.show()  # 顯示圖形

# 保留 age、balance、loan 三個特徵來訓練羅吉斯回歸的模型
X = data[['age', 'balance', 'loan']]  # 特徵變數
y = data['y']  # 目標變數

# 將 loan 特徵轉換為數值型
X['loan'] = X['loan'].apply(lambda x: 1 if x == 'yes' else 0)  # 將 'yes' 轉換為 1，'no' 轉換為 0

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 分割訓練集和測試集

# 訓練羅吉斯回歸模型
model = LogisticRegression()  # 建立羅吉斯回歸模型
model.fit(X_train, y_train)  # 訓練模型

# 預測
y_pred = model.predict(X_test)  # 使用測試集進行預測

# 輸出測試準確度
accuracy = accuracy_score(y_test, y_pred)  # 計算準確度
print(f'Test Accuracy: {accuracy}')  # 打印測試準確度