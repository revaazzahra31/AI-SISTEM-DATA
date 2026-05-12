import pandas as pd

#membaca file dataset
df = pd.read_csv('Data_koma.csv')

#menampilkan 5 baris pertama untuk verifikasi atribut
print("Daftar Atribut dalam Dataset:")
print(df.columns.tolist()) #menampilkan nama-nama kolom
print(df.head())    

