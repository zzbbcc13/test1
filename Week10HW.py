import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 匯入資料
file_path = r"C:/Users/USER/OneDrive/文件/大二上學期作業/程式作業/Week10作業/train.xlsx"
data = pd.read_excel(file_path)

# 檢查缺失值
print("Missing values check:")
print(data.isnull().sum())

# 打印資料欄位名稱
print(data.columns)

# 繪製長條圖
features = ['age', 'job', 'marital', 'education', 'loan']
for i, feature in enumerate(features):
    plt.figure(figsize=(10, 5))
    data[feature].value_counts().plot(kind='bar')
    plt.title(f'{feature} vs Subscription Count')
    plt.xlabel(feature)
    plt.ylabel('Subscription Count')
    if feature == 'age':  # 只調整第一張圖的 x 軸標籤間距
        plt.xticks(ticks=range(0, len(data[feature].value_counts()) * 2, 2), labels=data[feature].value_counts().index)
    plt.show()

# 保留age、balance、loan三個特徵來訓練羅吉斯回歸的模型
X = data[['age', 'balance', 'loan']]
y = data['y']

# 將loan特徵轉換為數值型
X['loan'] = X['loan'].apply(lambda x: 1 if x == 'yes' else 0)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練羅吉斯回歸模型
model = LogisticRegression()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 輸出測試準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy}')