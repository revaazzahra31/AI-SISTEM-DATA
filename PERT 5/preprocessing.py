import pandas as pd 
import numpy as np 

# 1. Load Dataset 
# Membaca file dataset asli 
df = pd.read_csv('DataSet Full.csv', sep=";") 
print("=== STATUS DATA AWAL ===") 
print("Jumlah missing values per kolom:") 
print(df.isnull().sum())

# 2. Mendeteksi dan Mengganti Spasi Kosong 
# Terkadang data kosong diisi dengan spasi ' ' yang dianggap teks oleh pandas. 
# Kita ubah spasi tersebut menjadi NaN agar bisa diproses sebagai missing values. 
df.replace(r'^\s*$', np.nan, regex=True, inplace=True) 
print("\n=== STATUS SETELAH CEK SPASI ===") 
print("Jumlah missing values (termasuk bekas spasi):") 
print(df.isnull().sum())

# 3. Proses Imputasi (Pengisian Data Kosong) 
# Untuk fitur numerik seperti Predikat, kita gunakan Median  
# agar tidak merusak distribusi data penjurusan. 
cols_predikat = ['Predikat IPA', 'Predikat IPS', 'Predikat Bahasa'] 

for col in cols_predikat: 
    # Mengubah tipe data ke numerik (memastikan tidak ada teks tersisa) 
    df[col] = pd.to_numeric(df[col], errors='coerce') 

    # Menghitung median 
    median_val = df[col].median() 

    # Mengisi nilai NaN dengan median 
    df[col] = df[col].fillna(median_val) 
    print(f"Selesai membersihkan {col}. Nilai kosong diisi dengan median: {median_val}")

# 4. Simpan Data Bersih 
# Data ini yang nantinya akan digunakan untuk tahap EDA dan Modeling 
cleaned_file_name = 'Data_Penjurusan_Cleaned.csv' 
df.to_csv(cleaned_file_name, index=False) 
print("\n=== PROSES PEMBERSIHAN SELESAI ===") 
print("Jumlah missing values akhir:") 
print(df.isnull().sum()) 
print(f"Data bersih disimpan dengan nama: {cleaned_file_name}") 