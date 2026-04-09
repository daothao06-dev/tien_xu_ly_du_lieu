import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'gia_nha': [2500, 3100, -500, 4200, np.nan, 3100, 15000, 4200], 
    'dien_tich': [50, 65, 70, np.nan, 85, 65, 500, np.nan],        
    'so_phong': [2, 3, 0, 4, 2, 3, 10, 4],                        
    'quan': ['Quận 1', 'Q1', 'Quận 3', 'Quận 1', 'Quận 2', 'Q1', 'Quận 7', 'Quận 1'], 
    'mo_ta': [
        'Nhà đẹp trung tâm', 'Nhà nát Q1', 'Gần chợ', 
        'View sông', 'Chính chủ', 'Nhà nát Q1', 'Biệt thự siêu sang', 'View sông'
    ]
}

df = pd.DataFrame(data)

# 1. GIAI ĐOẠN 1.1: KHÁM PHÁ DỮ LIỆU
print("--- 1.1 THỐNG KÊ MÔ TẢ ---")
print(df.describe())

print("\nGiá trị thiếu:\n", df.isnull().sum())
print("\nSố lượng trùng lặp:", df.duplicated().sum())

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 2)
sns.histplot(df['gia_nha'].dropna(), kde=True)
plt.title('Histogram Gia Nha')

plt.subplot(1, 2, 1)
sns.boxplot(x=df['gia_nha'])
plt.title('Boxplot Gia Nha')
plt.show()

# 2. GIAI ĐOẠN 1.2: XỬ LÝ DỮ LIỆU BẨN
print("\n--- 1.2 LÀM SẠCH DỮ LIỆU ---")

df = df[df['so_phong'] > 0]
df.loc[df['gia_nha'] < 0, 'gia_nha'] = np.nan

df['quan'] = df['quan'].replace({'Q1': 'Quận 1'})

df['dien_tich'] = df['dien_tich'].fillna(df['dien_tich'].median())
df['gia_nha'] = df['gia_nha'].fillna(df['gia_nha'].mean())

df = df.drop_duplicates().reset_index(drop=True)

print("\nKết quả sau khi làm sạch:")
print(df)
# 3. GIAI ĐOẠN 3: XỬ LÝ OUTLIERS
print("\n--- 3. XỬ LÝ OUTLIERS (IQR) ---")

Q1 = df['gia_nha'].quantile(0.25)
Q3 = df['gia_nha'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Capping: Thay thế các giá trị vượt ngưỡng bằng giá trị ngưỡng để không mất dữ liệu
df['gia_nha'] = np.where(df['gia_nha'] > upper_bound, upper_bound, 
                         np.where(df['gia_nha'] < lower_bound, lower_bound, df['gia_nha']))

print(f"Ngưỡng trên: {upper_bound}, Ngưỡng dưới: {lower_bound}")
print("Đã xử lý Outliers bằng phương pháp Capping.")

# 4. GIAI ĐOẠN 4: CHUẨN HÓA & ENCODING
from sklearn.preprocessing import MinMaxScaler

print("\n--- 4. CHUẨN HÓA & BIẾN ĐỔI CATEGORICAL ---")

# 4.1. Scaling (Đưa diện tích và giá nhà về khoảng 0-1)
scaler = MinMaxScaler()
df[['gia_nha', 'dien_tich']] = scaler.fit_transform(df[['gia_nha', 'dien_tich']])

# 4.2. One-hot Encoding cho cột 'quan'
# Thao lưu ý: Cột của bạn tên là 'quan' nên ở đây phải để là columns=['quan']
df = pd.get_dummies(df, columns=['quan'])

print("Dữ liệu sau khi Scaling và Encoding:")
print(df.head())

# 5. GIAI ĐOẠN 5: TEXT SIMILARITY (Cơ bản)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("\n--- 5. PHÁT HIỆN TRÙNG LẶP DỰA TRÊN MÔ TẢ ---")

# Chuyển mô tả thành ma trận số TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(df['mo_ta'])

# Tính độ tương đồng giữa các dòng
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# In ra các cặp có độ giống nhau cao (> 0.8) nhưng không phải là chính nó
for i in range(len(cosine_sim)):
    for j in range(i + 1, len(cosine_sim)):
        if cosine_sim[i][j] > 0.8:
            print(f"Phát hiện trùng lặp tiềm năng: Dòng {i} và Dòng {j}")
            print(f"Mô tả: '{df.iloc[i]['mo_ta']}' VS '{df.iloc[j]['mo_ta']}'")



