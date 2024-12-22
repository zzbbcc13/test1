import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 載入資料集
file_path = 'hotel_bookings.csv'
df = pd.read_csv(file_path)

# 顯示初始資料集大小
print("初始資料集大小:", df.shape)

# 顯示處理前的資料資訊
print("\n處理前的資料資訊:")
print(df.info())

# 顯示有缺少資料的欄位和數值
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
print("\n有缺少資料的欄位和數值:")
print(missing_values)

# 處理空值：用欄位的平均值填補數值型欄位的空值
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# 輸出處理空值後的資料
df.to_csv('processed_data.csv', index=False)
print("\n處理空值後的資料已輸出至 'processed_data.csv'")

# 標籤編碼處理類別型欄位
label_encoder = LabelEncoder()
categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    df[column] = label_encoder.fit_transform(df[column])

# 顯示處理後的資料資訊
print("\n處理後的資料資訊:")
print(df.info())

# 顯示資料集大小
print("\n資料集大小:", df.shape)

# 資料標籤化：將資料集中的數據附加特定的標籤
# 假設我們有一個新的標籤欄位 'label'，這裡簡單地將所有資料標記為 1
df['label'] = 1

# 創建資料摘要
summary = pd.DataFrame({
    'Column Name': df.columns,
    'Data Type': df.dtypes,
    'Category': ['Categorical' if col in categorical_columns else 'Numerical' for col in df.columns]
})

# 輸出資料摘要到不同的檔案名稱
summary.to_csv('data_summary_new.csv', index=False)
print("\n資料摘要已輸出至 'data_summary_new.csv'")