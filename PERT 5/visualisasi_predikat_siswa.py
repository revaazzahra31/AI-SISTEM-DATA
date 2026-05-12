import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# 1. Load Data yang sudah dibersihkan 
df = pd.read_csv('Data_Penjurusan_Cleaned.csv') 

# Mengatur tema visualisasi 
sns.set_theme(style="whitegrid") 

# ========================================================== 
# VISUALISASI 1: Distribusi Predikat (IPA, IPS, Bahasa) 
# ========================================================== 
# Kita buat 3 subplots dalam satu baris 
fig, axes = plt.subplots(1, 3, figsize=(18, 5)) 
fig.suptitle('Distribusi Predikat Siswa per Mata Pelajaran', fontsize=16) 

cols = ['Predikat IPA', 'Predikat IPS', 'Predikat Bahasa'] 
colors = ['#4C72B0', '#DD8452', '#55A868'] # Warna Biru, Orange, Hijau

for i, col in enumerate(cols): 
    sns.countplot(x=col, data=df, ax=axes[i], color=colors[i]) 
    axes[i].set_title(f'Distribusi {col}') 
    axes[i].set_xlabel('Nilai Predikat') 
    axes[i].set_ylabel('Jumlah Siswa') 

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.show() # Jika di lingkungan lokal 
# plt.savefig('distribusi_predikat.png') # Untuk menyimpan gambar 

# ========================================================== 
# VISUALISASI 2: Perbandingan Minat vs Jurusan Aktual 
# ========================================================== 
plt.figure(figsize=(14, 6)) 

# Grafik Minat Siswa (Input) 
plt.subplot(1, 2, 1) 
sns.countplot(x='Nilai Minat', data=df, palette='viridis',  
    order=df['Nilai Minat'].value_counts().index) 
plt.title('Minat Awal Siswa (Feature)', fontsize=14) 
plt.xlabel('Bidang Minat') 
plt.ylabel('Jumlah Siswa') 

# Grafik Jurusan Akhir (Target/Label) 
plt.subplot(1, 2, 2) 
sns.countplot(x='Jurusan', data=df, palette='magma',  
    order=df['Jurusan'].value_counts().index) 
plt.title('Hasil Penjurusan Aktual (Target)', fontsize=14) 
plt.xlabel('Jurusan') 
plt.ylabel('Jumlah Siswa') 

plt.tight_layout() 
plt.show() 
# plt.savefig('minat_vs_jurusan.png')

# ========================================================== 
# ANALISIS TAMBAHAN: Korelasi Sederhana 
# ========================================================== 
print("\n--- Ringkasan Data Penjurusan ---") 
print(f"Total Data: {len(df)} siswa") 
print("\nPersentase Jurusan:") 
print(df['Jurusan'].value_counts(normalize=True) * 100)