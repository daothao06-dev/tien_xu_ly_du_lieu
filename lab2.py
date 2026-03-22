##Bai1
import pandas as pd
#1
df = pd.read_csv("ITA105_Lab_2_Housing.csv")
print(df)
print("Shape:",df.shape)
missing_values = df.isnull().sum()
print(missing_values)
#2
print("Thong ke mo ta:", df.describe())
print("-->Nhận xét về dữ liệu: max rất lớn => có khả năng Outlier.")
#3
import matplotlib.pyplot as plt
numeric_column =  df.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_column:
    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.show()
print("-->Từ boxplot => Điểm nằm ngoài Boxplot là Outlier.")
#4
plt.figure()
plt.scatter(df['dien_tich'], df['gia'])
plt.xlabel("dien_tich")
plt.ylabel("gia")
plt.title("dien_tich vs gia")
plt.show()
print("-->Từ scatterplot => Điểm tách xa là Outlier.")
#5
Q1 = df[numeric_column].quantile(0.25)
Q2 = df[numeric_column].quantile(0.5)
Q3 = df[numeric_column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_IQR = ((df[numeric_column] < lower_bound) | (df[numeric_column] > upper_bound))
print("Số outlier theo IQR:", outlier_IQR.sum())
#6
from scipy.stats import zscore
import numpy as np
z_scores = np.abs(zscore(df[numeric_column]))
outlier_z = (z_scores > 3)
print("Số outlier theo Z-score:", outlier_z.sum())
#7
print("-->IQR, Boxplot gần giống nhau và phát hiện số lượng outlier nhiều hơn Z-score.")
#8
print("-->Nguyên nhân: Do lỗi nhập liệu => min max bất thường.")
#9
df_clip = df.copy()
for col in numeric_column:
    lower = Q1[col] - 1.5 * IQR[col]
    upper = Q3[col] + 1.5 * IQR[col]
    df_clip[col] = df_clip[col].clip(lower, upper)
#10
for col in numeric_column:
    plt.figure()
    df_clip.boxplot(column=col)
    plt.title(f"After cleaning - {col}")
    plt.show()
print("-->Boxplot sau khi xử lí => ít outlier hơn.")
df.to_csv("clean_housing.csv", index=False)

##Bai2
import pandas as pd
#1
df = pd.read_csv("ITA105_Lab_2_Iot.csv")
print(df)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
missing_values = df.isnull().sum()
print(missing_values)
#2
import matplotlib.pyplot as plt
sensors = df['sensor_id'].unique()
for s in sensors:
    plt.figure()
    df[df['sensor_id'] == s]['temperature'].plot()
    plt.title(f"Temperature - Sensor {s}")
    plt.show()
window = 10
rolling_mean = df['temperature'].rolling(window).mean()
rolling_std= df['temperature'].rolling(window).std()
lower_bound = rolling_mean + 3 * rolling_std
upper_bound = rolling_mean - 3 * rolling_std
outlier_roll = ((df['temperature'] < lower_bound) | (df['temperature'] > upper_bound))
print("Số outlier theo Rolling:", outlier_roll.sum())
#4
from scipy.stats import zscore
import numpy as np
z_scores = np.abs(zscore(df['temperature']))
outlier_z = (z_scores > 3)
print("Số outlier theo Z-score:", outlier_z.sum())
#5
import matplotlib.pyplot as plt
plt.figure()
df[['temperature', 'pressure', 'humidity']].boxplot()
plt.title("Boxplot")
plt.show()

plt.figure()
plt.scatter(df['temperature'], df['pressure'], alpha=0.5)

plt.scatter(df[outlier_z]['temperature'], df[outlier_z]['pressure'])
plt.title("Temp vs Pressure")
plt.show()

plt.figure()
plt.scatter(df['pressure'], df['humidity'], alpha=0.5)
plt.scatter(df[outlier_z]['pressure'], df[outlier_z]['humidity'])
plt.title("Pressure vs Humidity")
plt.show()
#6
print("-->Rolling mean phát hiện ngoại lệ theo thời gian, " \
"Z-score phát hiện theo toàn bộ phân phối," \
"Boxplot (IQR) phát hiện dựa trên khoảng tứ phân vị," \
"Scatter plot giúp trực quan hóa mối quan hệ và nhận diện điểm bất thường " \
"=> Số lượng outlier khác nhau do mỗi phương pháp có một đặc thù riêng.")
#7
df_clean = df.copy()
df_clean['temperature'] = df_clean['temperature'].interpolate()
df.to_csv("clean_Iot.csv", index=False)

##Bai3
import pandas as pd
#1
df = pd.read_csv("ITA105_Lab_2_Ecommerce.csv")
print(df)
missing_values = df.isnull().sum()
print(missing_values)
print("Thong ke mo ta:", df.describe())
#2
import matplotlib.pyplot as plt
numeric_column = ['price', 'quantity', 'rating']
for col in numeric_column:
    plt.figure()
    df.boxplot(column=col)
    plt.title(f"Boxplot of {col}")
    plt.show()
#3
Q1 = df[numeric_column].quantile(0.25)
Q2 = df[numeric_column].quantile(0.5)
Q3 = df[numeric_column].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outlier_IQR = ((df[numeric_column] < lower_bound) | (df[numeric_column] > upper_bound))
print("Số outlier theo IQR:", outlier_IQR.sum())

from scipy.stats import zscore
import numpy as np
z_scores = (df[numeric_column] - df[numeric_column].mean()) / df[numeric_column].std()
outlier_z = (z_scores > 3)
print("Số outlier theo Z-score:", outlier_z.sum())
#4
plt.figure()
plt.scatter(df['price'], df['quantity'], alpha=0.5)

plt.scatter(df[outlier_z['price']]['price'],
            df[outlier_z['price']]['quantity'])
plt.xlabel("Price")
plt.ylabel("Quantity")
plt.title("Price vs Quantity")
plt.show()
#5
print("Giá trị price = 0 → lỗi nhập liệu, " \
"rating > 5 → sai hệ thống (rating tối đa là 5), " \
"quantity quá lớn → có thể là đơn hàng bulk hoặc lỗi, " \
"category hiếm → dữ liệu ít → dễ thành outlier")
#6
df_clean = df.copy()
df_clean = df_clean[df_clean['price'] > 0]
df_clean = df_clean[df_clean['rating'] <= 5]

df_clean['price'] = np.log1p(df_clean['price'])
df_clean['quantity'] = np.log1p(df_clean['quantity'])
#7
import matplotlib.pyplot as plt
for col in numeric_column:
    plt.figure()
    df_clean.boxplot(column=col)
    plt.title(f"After cleaning - {col}")
    plt.show()

plt.figure()
plt.scatter(df_clean['price'], df_clean['quantity'], alpha=0.5)
plt.title("After cleaning")
plt.show()
print("-->Nhận xét:Sau khi xử lý, số lượng outlier giảm, " \
"boxplot ít điểm chênh lệch, mối quan hệ scatter rõ hơn.")
df.to_csv("clean_Ecommerce.csv", index=False)

##Bai4
#1
import pandas as pd
df = pd.read_csv("ITA105_Lab_2_Housing.csv")
cols_housing = ['dien_tich', 'gia']
z_housing = (df[cols_housing] - df[cols_housing].mean()) / df[cols_housing].std()
outlier_multi_housing = (abs(z_housing) > 3).any(axis=1)
print("So luong outlier theo Housing:", outlier_multi_housing.sum())

df = pd.read_csv("ITA105_Lab_2_Iot.csv")
cols_iot = ['temperature', 'pressure']
z_iot = (df[cols_iot] - df[cols_iot].mean()) / df[cols_iot].std()
outlier_multi_iot = (abs(z_iot) > 3).any(axis=1)
print("So luong outlier theo Iot:", outlier_multi_iot.sum())

df = pd.read_csv("ITA105_Lab_2_Ecommerce.csv")
cols_ecom = ['price', 'quantity', 'rating']
z_ecom = (df[cols_ecom] - df[cols_ecom].mean()) / df[cols_ecom].std()
outlier_multi_ecom = (abs(z_ecom) > 3).any(axis=1)
print("So luong outlier theo Ecommerce:", outlier_multi_ecom.sum())
#2
Q1 = df[cols_ecom].quantile(0.25)
Q2 = df[cols_ecom].quantile(0.5)
Q3 = df[cols_ecom].quantile(0.75)
IQR = Q3 - Q1
outlier_iqr_multi = ((df[cols_ecom] < (Q1 - 1.5 * IQR)) |
                     (df[cols_ecom] > (Q3 + 1.5 * IQR))).any(axis=1)
print("IQR multivariate:", outlier_iqr_multi.sum())
#3
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
scatter_matrix(df[['price', 'quantity', 'rating']], figsize=(8,8))
plt.show()
#4
outlier_uni = (abs(z_ecom) > 3)
print("Univariate:", outlier_uni.sum())
print("Multivariate:", outlier_multi_ecom.sum())

print("-->Multivariate phát hiện chính xác hơn trong dữ liệu thực tế, " \
"Univariate đơn giản nhưng có thể bỏ sót hoặc phát hiện sai")