import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set_theme(style="whitegrid")

def bai1():
    df1 = pd.read_csv('ITA105_Lab_3_Sports.csv')
    print("--- BÀI 1: VẬN ĐỘNG VIÊN ---")
    print("Missing values:\n", df1.isnull().sum())
    print("\nThống kê mô tả:\n", df1.describe())

    fig, axes = plt.subplots(2, len(df1.columns), figsize=(18, 8))
    fig.suptitle('Bài 1: Phân phối và Scale trước chuẩn hóa', fontsize=16)
    for i, col in enumerate(df1.columns):
        sns.histplot(df1[col], kde=True, ax=axes[0, i])
        sns.boxplot(y=df1[col], ax=axes[1, i])
    plt.tight_layout()
    plt.show()

    scaler_minmax = MinMaxScaler()
    scaler_zscore = StandardScaler()

    df1_minmax = pd.DataFrame(scaler_minmax.fit_transform(df1), columns=df1.columns)
    df1_zscore = pd.DataFrame(scaler_zscore.fit_transform(df1), columns=df1.columns)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(df1['chieu_cao_cm'], kde=True, ax=axes[0]).set_title('Gốc')
    sns.histplot(df1_minmax['chieu_cao_cm'], kde=True, ax=axes[1]).set_title('Min-Max [0,1]')
    sns.histplot(df1_zscore['chieu_cao_cm'], kde=True, ax=axes[2]).set_title('Z-Score (mean=0, std=1)')
    plt.tight_layout()
    plt.show()

def bai2():
    df2 = pd.read_csv('ITA105_Lab_3_Health.csv')
    print("\n--- BÀI 2: CHỈ SỐ BỆNH NHÂN ---")
    print("Thống kê mô tả (tìm ngoại lệ):\n", df2.describe())

    df2_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df2), columns=df2.columns)
    df2_zscore = pd.DataFrame(StandardScaler().fit_transform(df2), columns=df2.columns)

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df2)
    plt.title("Bài 2: Boxplot phát hiện ngoại lệ (Gốc)")
    plt.show()

def bai3():
    df3 = pd.read_csv('ITA105_Lab_3_Finance.csv')
    print("\n--- BÀI 3: CHỈ SỐ CÔNG TY ---")

    df3_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df3), columns=df3.columns)
    df3_zscore = pd.DataFrame(StandardScaler().fit_transform(df3), columns=df3.columns)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df3, ax=axes[0]).set_title('Gốc')
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df3_minmax, ax=axes[1]).set_title('Min-Max')
    sns.scatterplot(x='doanh_thu_musd', y='loi_nhuan_musd', data=df3_zscore, ax=axes[2]).set_title('Z-Score')
    plt.tight_layout()
    plt.show()

def bai4():
    df4 = pd.read_csv('ITA105_Lab_3_Gaming.csv')
    print("\n--- BÀI 4: NGƯỜI CHƠI TRỰC TUYẾN ---")
    print("Missing values:\n", df4.isnull().sum())

    df4_minmax = pd.DataFrame(MinMaxScaler().fit_transform(df4), columns=df4.columns)
    df4_zscore = pd.DataFrame(StandardScaler().fit_transform(df4), columns=df4.columns)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    sns.histplot(df4['gio_choi'], bins=20, ax=axes[0]).set_title('Giờ chơi (Gốc)')
    sns.histplot(df4_minmax['gio_choi'], bins=20, ax=axes[1]).set_title('Min-Max')
    sns.histplot(df4_zscore['gio_choi'], bins=20, ax=axes[2]).set_title('Z-Score')
    plt.tight_layout()
    plt.show() 

if __name__ == "__main__":
    bai1()
    bai2()
    bai3()
    bai4()