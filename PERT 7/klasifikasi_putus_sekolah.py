import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


# ==========================================================
# 1. LOAD DATA & INITIAL PREPARATION
# ==========================================================
# Pastikan file CSV menggunakan pemisah titik koma (;)
df = pd.read_csv('Dataset Pendidikan SD Indonesia Tahun 2023-2024.csv', sep=';')

def clean_data(df):
    # Merapikan nama kolom yang berisi enter/newline
    df.columns = [c.replace('\n', ' ').replace('  ', ' ').strip().replace('>- ', '>=') for c in df.columns]
    
    # Ubah semua kolom selain Provinsi menjadi numerik
    num_cols = df.columns.drop('Provinsi')
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Hapus baris kosong jika ada
    df = df.dropna().copy()
    return df

df = clean_data(df)

# ==========================================================
# 2. MEMBUAT TARGET KLASIFIKASI
# ==========================================================
# Karena dataset asli tidak punya label kelas,
# kita buat target berdasarkan Rasio Putus Sekolah
df['Rasio Putus Sekolah'] = df['Putus Sekolah'] / df['Siswa']

# Membagi menjadi 3 kategori: Rendah, Sedang, Tinggi
df['Kategori Putus Sekolah'] = pd.qcut(
    df['Rasio Putus Sekolah'],
    q=3,
    labels=['Rendah', 'Sedang', 'Tinggi']
)

# ==========================================================
# 3. FEATURE ENGINEERING
# ==========================================================
# Membuat fitur rasio agar data lebih stabil dan lebih cocok untuk klasifikasi
df['Siswa per Sekolah'] = df['Siswa'] / df['Sekolah']
df['Rasio Mengulang'] = df['Mengulang'] / df['Siswa']
df['Guru <S1 per Sekolah'] = df['Kepala Sekolah dan Guru(<S1)'] / df['Sekolah']
df['Guru >=S1 per Sekolah'] = df['Kepala Sekolah dan Guru(>=S1)'] / df['Sekolah']
df['Tendik SM per Sekolah'] = df['Tenaga Kependidikan(SM)'] / df['Sekolah']
df['Tendik >SM per Sekolah'] = df['Tenaga Kependidikan(>SM)'] / df['Sekolah']
df['Rombel per Sekolah'] = df['Rombongan Belajar'] / df['Sekolah']

# Total ruang kelas
df['Total Ruang Kelas'] = (
    df['Ruang kelas(baik)'] +
    df['Ruang kelas (rusak ringan)'] +
    df['Ruang kelas (rusak sedang)'] +
    df['Ruang kelas (rusak berat)']
)

# Persentase kondisi ruang kelas
df['Persen Ruang Baik'] = df['Ruang kelas(baik)'] / df['Total Ruang Kelas']
df['Persen Ruang Rusak Ringan'] = df['Ruang kelas (rusak ringan)'] / df['Total Ruang Kelas']
df['Persen Ruang Rusak Sedang'] = df['Ruang kelas (rusak sedang)'] / df['Total Ruang Kelas']
df['Persen Ruang Rusak Berat'] = df['Ruang kelas (rusak berat)'] / df['Total Ruang Kelas']

# ==========================================================
# 4. SELEKSI FITUR
# ==========================================================
features = [
    'Siswa per Sekolah',
    'Rasio Mengulang',
    'Guru <S1 per Sekolah',
    'Guru >=S1 per Sekolah',
    'Tendik SM per Sekolah',
    'Tendik >SM per Sekolah',
    'Rombel per Sekolah',
    'Persen Ruang Baik',
    'Persen Ruang Rusak Ringan',
    'Persen Ruang Rusak Sedang',
    'Persen Ruang Rusak Berat'
]

X = df[features]
y_text = df['Kategori Putus Sekolah']

# Encode target
le_target = LabelEncoder()
y = le_target.fit_transform(y_text)

# ==========================================================
# 5. MEMBAGI DATA TRAINING DAN TESTING
# ==========================================================
X_train, X_test, y_train, y_test, prov_train, prov_test = train_test_split(
    X, y, df['Provinsi'],
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================================================
# 6. SCALING KHUSUS UNTUK SVM
# ==========================================================
scaler = StandardScaler()
X_train_svm = scaler.fit_transform(X_train)
X_test_svm = scaler.transform(X_test)

# ==========================================================
# 7. TRAINING MODEL (Tiga Algoritma Sekaligus)
# ==========================================================
# Model 1: Decision Tree
model_dt = DecisionTreeClassifier(criterion='entropy', random_state=42)
model_dt.fit(X_train, y_train)

# Model 2: Naive Bayes
model_nb = GaussianNB()
model_nb.fit(X_train, y_train)

# Model 3: Support Vector Machine
model_svm = SVC(kernel='linear', random_state=42)
model_svm.fit(X_train_svm, y_train)

# ==========================================================
# 8. PREDIKSI
# ==========================================================
pred_dt = model_dt.predict(X_test)
pred_nb = model_nb.predict(X_test)
pred_svm = model_svm.predict(X_test_svm)

# Mengembalikan angka ke label teks
pred_dt_text = le_target.inverse_transform(pred_dt)
pred_nb_text = le_target.inverse_transform(pred_nb)
pred_svm_text = le_target.inverse_transform(pred_svm)
y_test_text = le_target.inverse_transform(y_test)

# ==========================================================
# 9. HASIL PREDIKSI PER PROVINSI
# ==========================================================
hasil = pd.DataFrame({
    'Provinsi': prov_test.values,
    'Aktual': y_test_text,
    'Prediksi_DT': pred_dt_text,
    'Prediksi_NB': pred_nb_text,
    'Prediksi_SVM': pred_svm_text
})

print("=== PERBANDINGAN HASIL PREDIKSI ===")
print(hasil)