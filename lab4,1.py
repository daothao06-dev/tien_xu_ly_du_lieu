#bai1: review khach san
import pandas as pd
import numpy as np
import re          
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
stop_words=["là","và","có", "rất", "thì", "một", "những", "các", "ở"]
def clean_text(text):
    text=text.lower()
    text=re.sub(r"[^\w\s]", "", text)
    text=text.replace('sạch sẽ','sạch_sẽ')
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return words
df=pd.read_csv('ITA105_Lab_4_Hotel_reviews.csv')
print(df)
df=df.dropna()
print(df)
missing_values=df.isnull().sum()
print(missing_values)
#label encoding( chuyen chu thanh so)
le=LabelEncoder()
df['hotel_name_encoded']=le.fit_transform(df['hotel_name'])
df['customer_type_encoded']=le.fit_transform(df['customer_type'])
print(df)
#xu li dang text
df['tokens']=df['review_text'].apply(clean_text)
print(df)
#/TF=IDF
tfidf=TfidfVectorizer()
tfidf_matrix=tfidf.fit_transform(df['review_text'])
print('\nTF-IDF matrix:')
print(tfidf_matrix.toarray())
#word2Vec
model = Word2Vec(sentences=df['tokens'], vector_size=50, window=4, min_count=1)
#tim tu giong 'sạch sẽ'
print("\ntu gan 'sạch_sẽ':")
print(model.wv.most_similar('sạch_sẽ',topn=5))
#giai thich
print('\nkhi nao dung TF-IDF?')
print('---khi dem cac tu quan trong---')
print('khi nao dung word2Vec?')
print('khi can hieu nghia cua tu')


print('---bài 2 bình luận trấn đấu---')
stop_words=['là', 'và', 'có', 'rất', 'thì', 'một', 'những', 'các', 'ở']
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]', '',text )
    text=text.replace('xuất sắc', 'xuất_sắc')
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return words
df1=pd.read_csv('ITA105_Lab_4_Match_comments.csv')
print(df1)
df1=df1.dropna()
print(df1)
missing_values=df.isnull().sum()
print(missing_values)
#ma hoa cac cot bang Label Ecoding
le=LabelEncoder()
df1['team_encoded']=le.fit_transform(df1['team'].astype(str))
df1['author_encoded']=le.fit_transform(df1['author'].astype(str))
print(df1)
#xu li dang text
df1['tokens']=df1['comment_text'].apply(clean_text)
print(df1)
#TF=IDF
tfidf1=TfidfVectorizer()
tfidf_matrix1=tfidf1.fit_transform(df1['comment_text'])
print('\nTF-IDF matrix:')
print(tfidf_matrix1.toarray())
#word2Vec
model1=Word2Vec(sentences=df1['tokens'],vector_size=50 ,window=4, min_count=1 )
#tim tu giong'xuat sac'
print('\n từ gần' 'xuất_sắc:')
print(model1.wv.most_similar('xuất_sắc',topn=5))
#giải thích
print('\n so sánh:')
print('TF-IDF: chỉ đếm từ')
print('Word2Vec: hiểu nghĩa tốt hơn')
print('---bài3 feedback người chơi ---')
stop_words=['là', 'và', 'có', 'rất', 'thì', 'một', 'những', 'các', 'ở']
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]', '',text )
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return words
df2=pd.read_csv('ITA105_Lab_4_Player_feedback.csv')
print(df2)
df2=df2.dropna()
print(df2)
missing_values=df.isnull().sum()
print(missing_values)
#ma hoa cac cot Label Encoding
le=LabelEncoder()
df2['player_type_encoded']=le.fit_transform(df2['player_type'])
df2['device_encoded']=le.fit_transform(df2['device'])
print(df2)
#xu li dang text
df2['tokens']=df2['feedback_text'].apply(clean_text)
print(df2)
#TF-IDF
tfidf2=TfidfVectorizer()
tfidf_matrix2=tfidf2.fit_transform(df2['feedback_text'])
print('\nTF-IDF matrix:')
print(tfidf_matrix2.toarray())
#word2Vec
model2=Word2Vec(sentences=df2['tokens'],vector_size=50, window=4, min_count=1 )
#tìm từ giống "đẹp"
print("\n từ gần 'đẹp':")
print(model2.wv.most_similar('đẹp',topn=5))
print('\n chọn gì để phân loại cảm xúc?')
print('-Word2Vec: hiểu nghĩa tốt hơn')
print('---bài 4 review album')
stop_words=['là', 'và', 'có', 'rất', 'thì', 'một', 'những', 'các', 'ở']
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^\w\s]', '',text )
    text=text.replace('sáng tạo', 'sáng_tạo')
    words=text.split()
    words=[w for w in words if w not in stop_words]
    return words
df3=pd.read_csv('ITA105_Lab_4_Album_reviews.csv')
print(df3)
df3=df3.dropna()
print(df3)
missing_values=df.isnull().sum()
print(missing_values)
#mã hóa các cột Label Encoding
le=LabelEncoder()
df3['genre_encoded']=le.fit_transform(df3['genre'])
df3['platform_encoded']=le.fit_transform(df3['platform'])
print(df3)
#xu li dang text
df3['tokens']=df3['review_text'].apply(clean_text)
print(df3)
#TF-IDF
tfidf3=TfidfVectorizer()
tfidf_matrix3=tfidf3.fit_transform(df3['review_text'])
print('\nTF-IDF matrix:')
print(tfidf_matrix3.toarray())
#word2Vec
model3=Word2Vec(sentences=df3['tokens'],vector_size=50, window=4, min_count=1 )
#tìm từ giống "đẹp"
print("\n từ gần 'sáng_tạo':")
print(model3.wv.most_similar('sáng_tạo',topn=5))