import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
import joblib
import matplotlib.pyplot as plt

# 載入資料集
file_path = "C://hotel_bookings.csv"
data = pd.read_csv(file_path)

# 選擇特徵和目標變數
features = [
    'hotel', 'lead_time', 'arrival_date_month', 'arrival_date_week_number', 'arrival_date_day_of_month',
    'stays_in_weekend_nights', 'stays_in_week_nights', 'adults', 'children', 'babies', 'meal',
    'country', 'market_segment', 'distribution_channel', 'is_repeated_guest', 'previous_cancellations',
    'previous_bookings_not_canceled', 'reserved_room_type', 'assigned_room_type', 'booking_changes',
    'deposit_type', 'agent', 'company', 'days_in_waiting_list', 'customer_type', 'adr',
    'required_car_parking_spaces', 'total_of_special_requests'
]
X = data[features]
y = data['is_canceled']

# 資料前處理
X = pd.get_dummies(X)  # 將類別型資料轉為數值型
imputer = SimpleImputer(strategy='mean')  # 建立填補缺失值的工具
X = imputer.fit_transform(X)  # 使用均值填補缺失值

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 預測測試集
y_pred = clf.predict(X_test)

# 計算總體準確度
accuracy = accuracy_score(y_test, y_pred)
print(f'總體準確度: {accuracy}')

# 顯示總體混淆矩陣
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title('confusion_matrix')
plt.show()

# 保存模型
joblib.dump(clf, 'hotel_booking_model.pkl')

# =========================================================
# 各酒店類型的預測
# =========================================================
results = {}
for hotel_type in data['hotel'].unique():
    # 篩選特定酒店類型的資料
    hotel_data = data[data['hotel'] == hotel_type]
    X_h = hotel_data[features]
    y_h = hotel_data['is_canceled']

    # 資料前處理
    X_h = pd.get_dummies(X_h)
    X_h = X_h.reindex(columns=pd.get_dummies(data[features]).columns, fill_value=0)  # 對齊欄位
    X_h = imputer.transform(X_h)

    # 分割資料集為訓練集和測試集
    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

    # 訓練模型
    clf_h = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_h.fit(X_train_h, y_train_h)

    # 預測
    y_pred_h = clf_h.predict(X_test_h)
    acc_h = accuracy_score(y_test_h, y_pred_h)

    # 儲存混淆矩陣和準確度
    cm_h = confusion_matrix(y_test_h, y_pred_h)
    results[hotel_type] = {
        'accuracy': acc_h,
        'confusion_matrix': cm_h
    }

    print(f'{hotel_type} 準確度: {acc_h}')

    # 顯示混淆矩陣
    disp_h = ConfusionMatrixDisplay(confusion_matrix=cm_h)
    disp_h.plot()
    plt.title(f'{hotel_type} confusion_matrix')
    plt.show()

# =========================================================
# 結果比較
# =========================================================
print("\n結果比較:")
for hotel_type, result in results.items():
    print(f"{hotel_type} 準確度: {result['accuracy']}")

# 顯示各酒店類型的混淆矩陣比較
fig, axes = plt.subplots(1, len(results), figsize=(12, 5))
for ax, (hotel_type, result) in zip(axes, results.items()):
    ConfusionMatrixDisplay(confusion_matrix=result['confusion_matrix']).plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f'{hotel_type} confusion_matrix')
plt.tight_layout()
plt.show()