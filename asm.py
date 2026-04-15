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
def gd2():
    def gd2_1():
        x = df[['area','num_rooms','built_year', 'district','source','description']]
        y = df['price']
        num_features = ['area','num_rooms','built_year']
        cat_features = ['district','source']
        text_features = 'description'
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
        ('scaler',StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer',SimpleImputer(strategy='most_frequent')),
            ('onehot',OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num',num_transformer,num_features),
            ('cat', cat_transformer,cat_features),
            ('text',TfidfVectorizer(max_features=100),text_features)
        ])
        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model',RandomForestRegressor(random_state=42))
        ])
        x_train, x_test, y_train,y_test = train_test_split(
            x, y, test_size=0.2,random_state=42
        )
        model_pipeline.fit(x_train, y_train)
        y_pred = model_pipeline.predict(x_test)
        mae = mean_absolute_error(y_test, y_pred)
        print("MAE = ",mae)
    gd2_1()
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.stats import skew
import xgboost as xgb 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
def advanced_feature_engineering(df_input):
    df = df_input.copy()
    print("\n--- ĐANG XỬ LÝ FEATURE ENGINEERING ---")
    mapping = {
        'gia_nha': 'price', 
        'dien_tich': 'area', 
        'so_phong': 'num_rooms', 
        'quan': 'district', 
        'mo_ta': 'description'
    }
    df = df.rename(columns=mapping)
    if 'built_year' not in df.columns:
        df['built_year'] = 2010
    if 'transaction_date' not in df.columns:
        df['transaction_date'] = pd.to_datetime('2026-01-01')
    if 'district' not in df.columns:
        df['district'] = 'Unknown'
    if 'price' not in df.columns:
        print("Cảnh báo: Không tìm thấy cột giá, đang dùng cột đầu tiên làm giá.")
        df['price'] = df.iloc[:, 0]

    df['price'] = pd.to_numeric(df['price'], errors='coerce').fillna(df['price'].median())
    df.loc[df['price'] <= 0, 'price'] = df['price'].median()
    df['price_log'] = np.log1p(df['price'])
    df['years_to_sell'] = df['transaction_date'].dt.year - df['built_year']
    if 'description' in df.columns:
        df['word_count'] = df['description'].fillna('').astype(str).apply(lambda x: len(x.split()))
    else:
        df['word_count'] = 0
    if 'area' in df.columns:
        df['area'] = pd.to_numeric(df['area'], errors='coerce').fillna(df['area'].median())
        df['price_per_m2'] = df['price'] / df['area'].replace(0, 1)
    else:
        df['price_per_m2'] = 0
    return df
# BƯỚC 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH
def train_and_evaluate(df_engineered, preprocessor, num_features, cat_features):
    print("\n--- 3. HUẤN LUYỆN MÔ HÌNH ---")
    X = df_engineered[num_features + cat_features]
    y = df_engineered['price_log'] 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    }
    results = []
    best_pipeline = None
    for name, model in models.items():
        pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
        pipeline.fit(X_train, y_train)
        y_pred_log = pipeline.predict(X_test)
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred_log)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        mae = mean_absolute_error(y_test_real, y_pred_real)
        r2 = r2_score(y_test, y_pred_log) 
        results.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        if name == "XGBoost":
            best_pipeline = pipeline 
    print(pd.DataFrame(results).to_string(index=False))
    # Kiểm thử Pipeline
    print("\n[TEST PIPELINE MỚI]: Shape output ->", best_pipeline.predict(X_train.iloc[:2]).shape, "- Không lỗi!")
    return best_pipeline, np.expm1(y_test), np.expm1(best_pipeline.predict(X_test))
# BƯỚC 4 & 5: DASHBOARD TRỰC QUAN HÓA
def create_dashboard(df_engineered, y_test_real, y_pred_real):
    print("\n--- 5. KHỞI TẠO DASHBOARD TRỰC QUAN ---")
    # Dashboard 1: Matplotlib & Seaborn
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('DASHBOARD DỰ BÁO VÀ PHÂN TÍCH GIÁ NHÀ', fontsize=16, fontweight='bold')
    ax1 = plt.subplot(2, 2, 1)
    sns.histplot(df_engineered['price'], bins=30, kde=True, color='red', ax=ax1)
    ax1.set_title('1. Phân phối Giá gốc (Lệch phải / Outlier)')
    ax2 = plt.subplot(2, 2, 2)
    sns.histplot(df_engineered['price_log'], bins=30, kde=True, color='blue', ax=ax2)
    ax2.set_title('2. Target sau Log-Transform (Chuẩn hóa)')
    ax3 = plt.subplot(2, 2, 3)
    sns.boxplot(x='district', y='price_log', data=df_engineered, ax=ax3)
    ax3.set_title('3. Mức giá Log phân bổ theo Quận')
    ax3.tick_params(axis='x', rotation=45)
    ax4 = plt.subplot(2, 2, 4)
    ax4.scatter(y_test_real, y_pred_real, alpha=0.5, color='purple')
    ax4.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
    ax4.set_title('4. Thực tế vs Dự đoán (XGBoost)')
    ax4.set_xlabel('Thực tế'), ax4.set_ylabel('Dự đoán')
    plt.tight_layout()
    plt.show()
    # Dashboard 2: Plotly (Tương tác)
    df_summary = df_engineered.groupby('district').agg(
        avg_price=('price', 'mean'), avg_price_m2=('price_per_m2', 'mean'), count=('price', 'count')
    ).reset_index()
    fig_plotly = px.scatter(
        df_summary, x='avg_price_m2', y='avg_price', size='count', color='district',
        text='district', title='Bản đồ Tương quan: Giá/m2 vs Tổng giá trị theo Khu vực',
        labels={'avg_price_m2': 'Giá trung bình / m2', 'avg_price': 'Tổng giá trung bình'}, size_max=40
    )
    fig_plotly.update_traces(textposition='top center')
    fig_plotly.show()

def run_assignment_final():
    df_final = advanced_feature_engineering(df)
    features_num = ['area', 'num_rooms', 'built_year', 'years_to_sell', 'word_count']
    exclude_cols = ['price', 'price_log', 'transaction_date', 'description', 'price_per_m2']
    features_full = [col for col in df_final.columns if col not in exclude_cols]
    scenarios = {
        "Mô hình Numerical (Cơ bản)": features_num,
        "Mô hình Full Features (Hoàn thiện)": features_full
    }
    for name, f_list in scenarios.items():
        print(f"\n--- Đang huấn luyện: {name} ---")
        missing = [c for c in f_list if c not in df_final.columns]
        if missing:
            print(f"Bỏ qua các cột thiếu: {missing}")
            f_list = [c for c in f_list if c in df_final.columns]
        X = df_final[f_list]
        y = df_final['price_log']
        current_num = X.select_dtypes(include=['number']).columns.tolist()
        current_cat = X.select_dtypes(include=['object', 'string']).columns.tolist()
        preprocessor = ColumnTransformer([
            ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scl', StandardScaler())]), current_num),
            ('cat', Pipeline([('imp', SimpleImputer(strategy='most_frequent')), 
                              ('ohe', OneHotEncoder(handle_unknown='ignore'))]), current_cat)
        ])
        model_p = Pipeline([('prep', preprocessor), ('model', xgb.XGBRegressor(random_state=42))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model_p.fit(X_train, y_train)
        y_pred = model_p.predict(X_test)
        # ... (các dòng code cũ phía trên giữ nguyên)
        mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))
        print(f"Kết quả MAE: {mae:,.2f} VNĐ")
    print("\n>>> Đang khởi tạo Dashboard...")
    y_test_real = np.expm1(y_test)
    y_pred_real = np.expm1(y_pred)
    create_dashboard(df_final, y_test_real, y_pred_real)


