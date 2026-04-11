import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 5)

# --- BÀI 1: Dữ liệu Chứng khoán (Stock) - Biểu đồ đường ---
def bai1_stock():
    file_name = 'ITA105_Lab_5_Stock.csv'
    if not os.path.exists(file_name):
        print(f"Lỗi: Không tìm thấy file {file_name}")
        return
    
    df = pd.read_csv(file_name)
    num_cols = df.select_dtypes(include='number').columns
    
    if len(num_cols) > 0:
        plt.figure()
        for col in num_cols:
            plt.plot(df.index, df[col], label=col, marker='o', markersize=3)
        plt.title('Bài 1: Biểu đồ biến động Chứng khoán (Stock)', fontsize=14, fontweight='bold')
        plt.xlabel('Chỉ mục / Thời gian')
        plt.ylabel('Giá trị')
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Không có dữ liệu số để vẽ biểu đồ.")

def bai2_supermarket():
    file_name = 'ITA105_Lab_5_Supermarket.csv'
    if not os.path.exists(file_name):
        print(f"Lỗi: Không tìm thấy file {file_name}")
        return
    
    df = pd.read_csv(file_name)
    num_cols = df.select_dtypes(include='number').columns
    cat_cols = df.select_dtypes(exclude='number').columns
    
    plt.figure()
    if len(cat_cols) > 0 and len(num_cols) > 0:
        summary = df.groupby(cat_cols[0])[num_cols[0]].sum().sort_values(ascending=False).head(10)
        summary.plot(kind='bar', color='skyblue', edgecolor='black')
        plt.title(f'Bài 2: Tổng {num_cols[0]} theo {cat_cols[0]} (Top 10)', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    elif len(num_cols) > 0:
        df[num_cols[0]].head(15).plot(kind='bar', color='coral')
        plt.title('Bài 2: Biểu đồ cột Siêu thị (Supermarket)', fontsize=14, fontweight='bold')
    
    plt.ylabel('Giá trị')
    plt.tight_layout()
    plt.show()

# --- BÀI 3: Dữ liệu Sản xuất (Production) - Biểu đồ phân phối / Boxplot ---
def bai3_production():
    file_name = 'ITA105_Lab_5_Production.csv'
    if not os.path.exists(file_name):
        print(f"Lỗi: Không tìm thấy file {file_name}")
        return
    
    df = pd.read_csv(file_name)
    num_cols = df.select_dtypes(include='number').columns
    
    if len(num_cols) > 0:
        plt.figure()
        sns.boxplot(data=df[num_cols], orient="h", palette="Set2")
        plt.title('Bài 3: Phân phối dữ liệu Sản xuất (Boxplot)', fontsize=14, fontweight='bold')
        plt.xlabel('Giá trị')
        plt.tight_layout()
        plt.show()

def bai4_webtraffic():
    file_name = 'ITA105_Lab_5_Web_traffic.csv'
    if not os.path.exists(file_name):
        print(f"Lỗi: Không tìm thấy file {file_name}")
        return
    
    df = pd.read_csv(file_name)
    num_cols = df.select_dtypes(include='number').columns
    
    if len(num_cols) >= 2:
        plt.figure()
        sns.scatterplot(x=df[num_cols[0]], y=df[num_cols[1]], alpha=0.7, color='purple')
        plt.title(f'Bài 4: Tương quan giữa {num_cols[0]} và {num_cols[1]}', fontsize=14, fontweight='bold')
        plt.xlabel(num_cols[0])
        plt.ylabel(num_cols[1])
        plt.tight_layout()
        plt.show()
    elif len(num_cols) == 1:
        plt.figure()
        sns.histplot(df[num_cols[0]], kde=True, color='purple')
        plt.title(f'Bài 4: Phân phối của {num_cols[0]} (Web Traffic)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    bai1_stock()
    bai2_supermarket()
    bai3_production()
    bai4_webtraffic()
