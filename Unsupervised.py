
import numpy as np
import pandas as pd
import random
import warnings
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

## Adım 1: flo_data_20K.csv verisini okutunuz.

df = pd.read_csv("/Users/ulukan/PycharmProjects/pythonProject/osman/machine_learning/odev/Flo gözetimsiz ogrenme/flo_data_20k.csv")

df.head()

eda.check_df(df)

## Adım 2: Müşterileri segmentlerken kullanacağınız değişkenleri seçiniz.

df['last_order_date'] = pd.to_datetime(df['last_order_date'])

df['first_order_date'] = pd.to_datetime(df['first_order_date'])

df['last_order_date_online'] = pd.to_datetime(df['last_order_date_online'])

df['last_order_date_offline'] = pd.to_datetime(df['last_order_date_offline'])

df["last_order_date"].max()

today_date = dt.datetime(2021, 6, 2)

df['Tenure'] = (today_date - df['first_order_date']).dt.days

df['Recency'] = (today_date - df['last_order_date']).dt.days

df['Last_online'] = (today_date - df['last_order_date_online']).dt.days

df['Last_offline'] = (today_date - df['last_order_date_offline']).dt.days

df["interested_in_categories_12"].unique()

df.shape

df.head()
## Görev 2: K-Means ile Müşteri Segmentasyonu

## Adım 1: Değişkenleri standartlaştırınız.

cat_cols,cat_but_car,num_cols,num_but_cat = eda.grab_col_names(df)

cat_cols = cat_cols + [col for col in cat_but_car if col not in "master_id"]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)

df.head()

df.shape

df.drop(["first_order_date", "last_order_date", "last_order_date_online", "last_order_date_offline", "master_id"], axis=1,inplace=True)


sc = MinMaxScaler()
df = sc.fit_transform(df)

dfh = df.copy()


## Adım 2: Optimum küme sayısını belirleyiniz.

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

kmeans = KMeans(n_clusters=8).fit(df)

kmeans.n_clusters           # K-Means modelinin kaç küme oluşturduğu
kmeans.cluster_centers_     # her bir küme merkezinin koordinatları
kmeans.labels_              # her bir veri noktasının hangi kümeye ait olduğu

clusters_kmeans = kmeans.labels_

df = pd.read_csv("/Users/ulukan/PycharmProjects/pythonProject/osman/machine_learning/odev/Flo gözetimsiz ogrenme/flo_data_20k.csv")

df["CLUSTER"] = clusters_kmeans

df.loc[df["CLUSTER"] == 0]

df["CLUSTER"] = df["CLUSTER"] + 1

df.head()

df[df["CLUSTER"]==5]

df.groupby("CLUSTER").agg(["count","mean","median"])

df.groupby("CLUSTER").describe().T

# df.to_csv("CLUSTER.csv")

hc_average = linkage(dfh, "average")

#  linkage fonksiyonu, veri noktalarını birbirine bağlayan birleşme veya uzaklık ölçülerini hesaplamak için kullanılır.

# "ward": Küme içi varyansı minimize eden şekilde birleştirme yapar.
# "single": Her iki küme içindeki en yakın noktaları birleştirir.
# "complete": Her iki küme içindeki en uzak noktaları birleştirir.
# "average": Küme merkezleri arasındaki ortalama uzaklığı kullanır.


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.show()


plt.figure(figsize=(7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=2, color='r', linestyle='--')
plt.axhline(y=2.1, color='b', linestyle='--')
plt.show()


cluster = AgglomerativeClustering(n_clusters=7, linkage="average")

clusters = cluster.fit_predict(dfh)


df.head()

df = pd.read_csv("/Users/ulukan/PycharmProjects/pythonProject/osman/machine_learning/odev/Flo gözetimsiz ogrenme/flo_data_20k.csv")

df["HI_CLUSTER"] = clusters
df["HI_CLUSTER"] = df["HI_CLUSTER"] + 1

df["CLUSTER"] = clusters_kmeans
df["CLUSTER"] = df["CLUSTER"] + 1


df.groupby("HI_CLUSTER").agg(["count","mean","median"])

df.groupby("HI_CLUSTER").describe().T

df.groupby("CLUSTER").agg(["count","mean","median"])

df.groupby("CLUSTER").describe().T





