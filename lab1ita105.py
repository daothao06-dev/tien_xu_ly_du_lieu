import pandas as pd
# bai1-Kham pha du lieu
df = pd.read_csv("ITA105_Lab_1.csv")
print(df)
print("Kich thuoc du lieu:",df.shape)
print("Mo ta:",df.describe())
print("Mo ta:",df.describe())
missing = df.isnull().sum()
print(missing)
#bai2-Xu li du lieu thieu
missing = df.isnull().sum()
print(missing)
df["Price"].fillna(df["Price"].mean(), inplace=True)
df["Rating"].fillna(df["Rating"].median(), inplace=True)
df["StockQuantity"].fillna(df["StockQuantity"].mode(), inplace=True)
print(df)
df_dropna = df.dropna()
print("--> Phuong phap dropna:", df_dropna.shape)
#bai3-Xu li du lieu loi
print("--> Gia tri loi:", df[df["Price"] < 2])
df = df[df["Price"] >=5]
print("--> Gia tri loi:", df[df["StockQuantity"] < 0])
df = df[df["StockQuantity"] >=0]
print("-->Gia tri khong hop le:", df[df["Rating"]>5])
df = df[df["Rating"]<=5] 
df = df[df["Rating"]>=0]
#bai4-lam muot du lieu nhieu
import matplotlib.pyplot as plt
df["Price_smooth"] = df["Price"].rolling(window=3).mean()
plt.plot(df["Price"])
plt.plot(df["Price_smooth"])
plt.title("Price before and after smoothing")
plt.show()
#bai5-Chuan hoa du lieu
df["Category"] = df["Category"].str.lower()
df["Description"] = df["Description"].str.strip()
df["Price_VND"] = df["Price"]*23000
print("--> USD->VND:", df["Price_VND"])
print(df)
df.to_csv("clean_data.csv", index=False)