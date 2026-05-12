import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

#menghilangkan semua pesan peringatan (warning)
warnings.filterwarnings('ignore')

#membaca data
df = pd.read_csv('Data_kosong.csv')

#proses pembersihan & imputasi (Sesuai langkah sebelumnya)
df_bersih = df.copy()
#daftar atribut numerik yang perlu diperiksa (Predikat dan Minat)
atribut_nilai = ['Predikat IPA', 'Predikat IPS', 'Predikat Bahasa', 'Nilai Minat']

for kolom in atribut_nilai:
    # mengubah spasi kosong (" ") menjadi NaN agar terbaca sebagai data kosong
    df_bersih[kolom] = pd.to_numeric(df_bersih[kolom], errors='coerce') 
    #mengisi data yang kosong (NaN) dengan nilai rata-rata tersebut
    df_bersih[kolom] = df_bersih[kolom].fillna(df_bersih[kolom].mean())

#grafuk heatmap: memastikan tidak ada lagi "lubang" data kosong setelah pembersihan
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(df.replace(r'^\s*$', np.nan, regex=True).isnull(), cbar=False, cmap='viridis')
plt.title('Sebelum: Ada Data Kosong (Garis Kuning)')

plt.subplot(1, 2, 2)
sns.heatmap(df_bersih.isnull(), cbar=False, cmap='viridis')
plt.title('Sesudah: Data Bersih (Ungu Solid)')
plt.show()

#grafik ba: melihat distribusi nilai rata-rata
plt.figure(figsize=(10, 6))
df_plot = df_bersih.melt(id_vars=('Nama Siswa'), value_vars=atribut_nilai)
sns.barplot(data=df_plot, x='variable', y='value', palette='viridis')
plt.title('Validasi Distribusi Nilai Setelah Imputasi')
plt.show