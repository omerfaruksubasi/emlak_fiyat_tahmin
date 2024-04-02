import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

df = pd.read_csv("evler.csv",sep=";")#veri setini import ettik
reg = linear_model.LinearRegression()
reg.fit(df[['alan','odasayisi','binayasi']], df['fiyat'])
alan = float(input("alan:"))
odasayisi = int(input("oda sayısı:"))
binayasi = int(input("bina yaşı:"))

tahmin = reg.predict([[alan,odasayisi,binayasi]])
print("tahmini fiyat:",tahmin[0])