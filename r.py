import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt

# 讀取 train.csv 文件
train_df = pd.read_csv('train.csv')

# 打印原始列名
print("Original columns:")
print(train_df.columns)

# 修改列名
new_columns = {
    'PassengerId': 'ID',
    'Survived': 'Target',
    'Pclass': 'Class',
    'Name': 'FullName',
    'Sex': 'Gender',
    'Age': 'AgeYears',
    'SibSp': 'SiblingsSpouses',
    'Parch': 'ParentsChildren',
    'Ticket': 'TicketNumber',
    'Fare': 'FareAmount',
    'Cabin': 'CabinNumber',
    'Embarked': 'PortEmbarked'
}
train_df.rename(columns=new_columns, inplace=True)

# 打印修改後的列名
print("Modified columns:")
print(train_df.columns)

# 打印前10筆資料
print("First 10 rows of the dataset:")
print(train_df.head(10))

# 處理缺失值
# 只對數值型列進行填補
numeric_columns = train_df.select_dtypes(include=['number']).columns
train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].mean())

# 類別標籤轉換
label_encoders = {}
for column in train_df.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    train_df[column] = label_encoders[column].fit_transform(train_df[column])

# 確保所有數值列都是數字類型
for column in train_df.select_dtypes(include=['object']).columns:
    train_df[column] = pd.to_numeric(train_df[column], errors='coerce')

# 打印前10筆資料以檢查轉換後的結果
print("First 10 rows of the dataset after preprocessing:")
print(train_df.head(10))

# 分割數據集
X = train_df.drop('Target', axis=1)
y = train_df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 訓練羅吉斯回歸模型
logistic_model = LogisticRegression()
logistic_model.fit(X_train, y_train)

# 訓練決策樹模型
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# 預測
logistic_y_pred = logistic_model.predict(X_test)
tree_y_pred = tree_model.predict(X_test)

# 評估羅吉斯回歸模型
logistic_accuracy = accuracy_score(y_test, logistic_y_pred)
logistic_precision = precision_score(y_test, logistic_y_pred)
logistic_recall = recall_score(y_test, logistic_y_pred)
logistic_f1 = f1_score(y_test, logistic_y_pred)

print("Logistic Regression Model:")
print(f"Accuracy: {logistic_accuracy}")
print(f"Precision: {logistic_precision}")
print(f"Recall: {logistic_recall}")
print(f"F1 Score: {logistic_f1}")

# 評估決策樹模型
tree_accuracy = accuracy_score(y_test, tree_y_pred)
tree_precision = precision_score(y_test, tree_y_pred)
tree_recall = recall_score(y_test, tree_y_pred)
tree_f1 = f1_score(y_test, tree_y_pred)

print("\nDecision Tree Model:")
print(f"Accuracy: {tree_accuracy}")
print(f"Precision: {tree_precision}")
print(f"Recall: {tree_recall}")
print(f"F1 Score: {tree_f1}")

# 繪製羅吉斯回歸模型的混淆矩陣
logistic_cm = confusion_matrix(y_test, logistic_y_pred)
logistic_disp = ConfusionMatrixDisplay(confusion_matrix=logistic_cm)
logistic_disp.plot(cmap='Blues')
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# 繪製決策樹模型的混淆矩陣
tree_cm = confusion_matrix(y_test, tree_y_pred)
tree_disp = ConfusionMatrixDisplay(confusion_matrix=tree_cm)
tree_disp.plot(cmap='Greens')
plt.title("Decision Tree Confusion Matrix")
plt.show()