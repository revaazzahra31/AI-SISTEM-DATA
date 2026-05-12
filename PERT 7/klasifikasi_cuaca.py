# ============================================
# KLASIFIKASI CUACA MENGGUNAKAN DECISION TREE
# ============================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# --------------------------------------------
# 1. MEMBACA DATASET
# --------------------------------------------
# Ganti nama file jika perlu
df = pd.read_csv("Dataset Cuaca.csv", sep=';')

print("5 data pertama:")
print(df.head())
print("\nInfo data:")
print(df.info())
print("\nJumlah data tiap kelas:")
print(df["Cuaca"].value_counts())


# --------------------------------------------
# 2. MEMBERSIHKAN DATA
# --------------------------------------------
# Pada file ini angka tersimpan dengan banyak titik,
# jadi kita bersihkan dulu lalu ubah menjadi numerik

def clean_number(x):
    """
    Menghapus semua karakter selain angka,
    lalu mengubahnya menjadi float.
    Dibagi 1e15 agar nilainya kembali ke skala yang lebih masuk akal.
    """
    x = str(x)
    digits_only = ''.join(ch for ch in x if ch.isdigit())
    if digits_only == "":
        return 0.0
    return int(digits_only) / 1e15

numeric_cols = ["Suhu", "Kelembaban", "Angin"]

for col in numeric_cols:
    df[col] = df[col].apply(clean_number)

print("\nData setelah dibersihkan:")
print(df.head())


# --------------------------------------------
# 3. MENENTUKAN FITUR DAN TARGET
# --------------------------------------------
X = df[["Suhu", "Kelembaban", "Angin"]]   # fitur / variabel input
y = df["Cuaca"]                           # target / label kelas


# --------------------------------------------
# 4. MEMBAGI DATA TRAINING DAN TESTING
# --------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        # 20% data testing
    random_state=42,      # agar hasil konsisten
    stratify=y            # menjaga proporsi kelas
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))


# --------------------------------------------
# 5. MEMBANGUN MODEL DECISION TREE
# --------------------------------------------
model = DecisionTreeClassifier(
    criterion="entropy",   # bisa juga "gini"
    max_depth=4,           # agar pohon tidak terlalu kompleks
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------
# 6. MELAKUKAN PREDIKSI
# --------------------------------------------
y_pred = model.predict(X_test)


# --------------------------------------------
# 7. EVALUASI MODEL
# --------------------------------------------
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n=== HASIL EVALUASI ===")
print("Akurasi Model:", round(acc * 100, 2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)


# --------------------------------------------
# 8. MENAMPILKAN POHON KEPUTUSAN
# --------------------------------------------
plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)
plt.title("Decision Tree untuk Klasifikasi Cuaca")
plt.show()


# --------------------------------------------
# 9. CONTOH PREDIKSI DATA BARU
# --------------------------------------------
# Contoh: Suhu=28, Kelembaban=7.5, Angin=10
data_baru = pd.DataFrame({
    "Suhu": [28],
    "Kelembaban": [7.5],
    "Angin": [10]
})

hasil_prediksi = model.predict(data_baru)
print("\nPrediksi data baru:", hasil_prediksi[0])