# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt

# =========================================================
# 使用繁體中文註解
# =========================================================
# 載入資料集 (假設 hotel_bookings.csv 在工作目錄下)
data = pd.read_csv("C://hotel_bookings.csv")

# 範例假設:
# 'hotel' 欄位中有 "City Hotel" 與 "Resort Hotel"
# 'is_canceled' 欄位表示預訂是否被取消 (0 表示未取消，1 表示已取消)

# 選擇特徵和目標變數
features = ['hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type', 'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests']
X = data[features]
y = data['is_canceled']

# 資料前處理
# 將類別型特徵轉換為數值型特徵
X = pd.get_dummies(X)

# 處理缺失值
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 將資料集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 預測測試集
y_pred = clf.predict(X_test)

# 計算準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'準確度: {accuracy}')

# 顯示混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# 保存模型
joblib.dump(clf, 'hotel_booking_model.pkl')

# =========================================================
# 非遷移學習部分
# =========================================================

# 篩選 City Hotel 的資料
city_hotel_data = data[data['hotel'] == 'City Hotel']
X_c = city_hotel_data[features]
y_c = city_hotel_data['is_canceled']

# 資料前處理
# 將類別型特徵轉換為數值型特徵
X_c = pd.get_dummies(X_c)

# 處理缺失值
X_c = imputer.transform(X_c)

# 將 City Hotel 資料集分割為訓練集和測試集
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

# 建立隨機森林模型 (非遷移學習)
clf_non_transfer = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型 (非遷移學習)
clf_non_transfer.fit(X_train_c, y_train_c)

# 預測 (非遷移學習)
y_pred_c = clf_non_transfer.predict(X_test_c)
acc_non_transfer = accuracy_score(y_test_c, y_pred_c)

print("非遷移學習 - City Hotel 資料預測準確度:", acc_non_transfer)

# 顯示混淆矩陣 (非遷移學習)
cm_non_transfer = confusion_matrix(y_test_c, y_pred_c)
disp_non_transfer = ConfusionMatrixDisplay(confusion_matrix=cm_non_transfer)
disp_non_transfer.plot()
plt.show()

# =========================================================
# 遷移學習區塊
# =========================================================

# 使用原始模型的參數進行微調
initial_params = clf.get_params()
clf_transfer = RandomForestClassifier(**initial_params)
clf_transfer.fit(X_train, y_train)

# 微調模型
initial_params = clf_transfer.get_params()
clf_transfer_tuned = RandomForestClassifier(**initial_params)  # 使用同樣參數初始化一個新模型
clf_transfer_tuned.fit(X_train_c, y_train_c)

# 對 City Hotel 測試集預測 (遷移學習)
y_pred_transfer = clf_transfer_tuned.predict(X_test_c)
acc_transfer = accuracy_score(y_test_c, y_pred_transfer)

print("遷移學習 - City Hotel 微調後預測準確度:", acc_transfer)

# 顯示混淆矩陣 (遷移學習)
cm_transfer = confusion_matrix(y_test_c, y_pred_transfer)
disp_transfer = ConfusionMatrixDisplay(confusion_matrix=cm_transfer)
disp_transfer.plot()
plt.show()

# =========================================================
# 結果比較
# =========================================================
print("非遷移學習準確度:", acc_non_transfer)
print("遷移學習準確度:", acc_transfer)

# 顯示混淆矩陣比較
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay(confusion_matrix=cm_non_transfer).plot(ax=axes[0], cmap=plt.cm.Blues)
axes[0].set_title('非遷移學習混淆矩陣')
ConfusionMatrixDisplay(confusion_matrix=cm_transfer).plot(ax=axes[1], cmap=plt.cm.Blues)
axes[1].set_title('遷移學習混淆矩陣')
plt.show()