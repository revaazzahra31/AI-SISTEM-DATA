import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 

# ========================================================== 
# 1. LOAD DATA & INITIAL PREPARATION 
# ========================================================== 
# Pastikan file CSV menggunakan pemisah titik koma (;) 
df_train = pd.read_csv('Data Training Kelulusan.csv', sep=';') 
df_test = pd.read_csv('Data Testing Kelulusan.csv', sep=';')

def clean_data(df): 
    # Mengubah format desimal (koma ke titik) pada kolom IP 
    ip_cols = ['IP Semester 1', 'IP Semester 2', 'IP Semester 3', 'IP Semester 4'] 
    for col in ip_cols: 
        df[col] = df[col].astype(str).str.replace(',', '.').astype(float)

    # Label Encoding untuk Jenis Kelamin (Kategorikal -> Numerik) 
    le = LabelEncoder() 
    df['Jenis Kelamin'] = le.fit_transform(df['Jenis Kelamin']) 
    return df

# Jalankan pembersihan data 
df_train = clean_data(df_train)
df_test = clean_data(df_test)

# Label Encoding untuk Target 'Status Kelulusan' 
le_target = LabelEncoder() 
df_train['Status Kelulusan'] = le_target.fit_transform(df_train['Status Kelulusan']) 

# ========================================================== 
# 2. SELEKSI FITUR (Sesuai dengan Parameter Akademik) 
# ========================================================== 
features = ['Jenis Kelamin', 'Jenis Seleksi', 'Pendapatan Ayah', 'Pendidikan Ibu',  
    'IP Semester 1', 'IP Semester 2', 'IP Semester 3', 'IP Semester 4', 
    'SKS Semester 1', 'SKS Semester 2', 'SKS Semester 3', 'SKS Semester 4']

X_train = df_train[features] 
y_train = df_train['Status Kelulusan'] 
X_test = df_test[features]

# ========================================================== 
# 3. TRAINING MODEL (Tiga Algoritma Sekaligus) 
# ========================================================== 
# Model 1: Decision Tree (Referensi: Tom Mitchell) 
model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42) 
model_dt.fit(X_train, y_train) 

# Model 2: Naive Bayes (Referensi: Tom Mitchell) 
model_nb = GaussianNB() 
model_nb.fit(X_train, y_train)

# Model 3: Support Vector Machine (Referensi: Deng et al. 2013) 
# Kita gunakan kernel 'linear' untuk mencari optimal hyperplane 
model_svm = SVC(kernel='linear', random_state=42) 
model_svm.fit(X_train, y_train)

# ========================================================== 
# 4. PREDIKSI & PERBANDINGAN HASIL 
# ========================================================== 
# Melakukan prediksi pada data testing 
pred_dt = model_dt.predict(X_test) 
pred_nb = model_nb.predict(X_test) 
pred_svm = model_svm.predict(X_test) 

# Mengembalikan nilai angka ke teks asli (Tepat/Terlambat) 
df_test['Prediksi_DT'] = le_target.inverse_transform(pred_dt) 
df_test['Prediksi_NB'] = le_target.inverse_transform(pred_nb) 
df_test['Prediksi_SVM'] = le_target.inverse_transform(pred_svm)

# Menampilkan Tabel Perbandingan 
print("=== PERBANDINGAN HASIL PREDIKSI KELULUSAN ===") 
print("-------------------------------------------------------------") 
cols_to_show = ['IP Semester 4', 'SKS Semester 4', 'Prediksi_DT', 'Prediksi_NB', 
'Prediksi_SVM'] 
print(df_test[cols_to_show]) 
print("-------------------------------------------------------------")