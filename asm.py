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