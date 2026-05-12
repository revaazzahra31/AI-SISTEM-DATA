import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# ===============================
# 1. MEMBACA DATASET
# ===============================
data = pd.read_csv("dataset_lulus_tepat_waktu_300.csv", sep=";")

print("=== 5 DATA TERATAS ===")
print(data.head())

# ===============================
# 2. PREPROCESSING DATA
# ===============================
data["Status_Skripsi"] = data["Status_Skripsi"].map({
    "Belum": 0,
    "Proses": 1,
    "Selesai": 2
})

data["Lulus_Tepat_Waktu"] = data["Lulus_Tepat_Waktu"].map({
    "Tidak": 0,
    "Ya": 1
})

# ===============================
# 3. MENENTUKAN FITUR DAN TARGET
# ===============================
X = data[[
    "IPK",
    "SKS",
    "Jumlah_Mengulang",
    "Semester_Cuti",
    "Kehadiran_%",
    "Status_Skripsi",
    "Masa_Studi_Semester"
]]

y = data["Lulus_Tepat_Waktu"]

# ===============================
# 4. MEMBAGI DATA TRAINING DAN TESTING
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ===============================
# 5. MEMBUAT MODEL DECISION TREE
# ===============================
model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=3,
    random_state=42
)

# Training model
model.fit(X_train, y_train)

# ===============================
# 6. PREDIKSI DATA TESTING
# ===============================
y_pred = model.predict(X_test)

# ===============================
# 7. EVALUASI MODEL
# ===============================
print("\n=== HASIL EVALUASI MODEL ===")
print("Akurasi Model:", round(accuracy_score(y_test, y_pred) * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ===============================
# 8. VISUALISASI POHON KEPUTUSAN
# ===============================
plt.figure(figsize=(28, 16), dpi=100)

plot_tree(
    model,
    feature_names=X.columns,
    class_names=["Tidak", "Ya"],
    filled=True,
    rounded=True,
    fontsize=10,
    proportion=True,
    impurity=False
)

plt.title("Pohon Keputusan Prediksi Lulus Tepat Waktu", fontsize=18)
plt.tight_layout()
plt.show()

# ===============================
# 9. CONTOH PREDIKSI MAHASISWA BARU
# ===============================
mahasiswa_baru = pd.DataFrame({
    "IPK": [3.45],
    "SKS": [144],
    "Jumlah_Mengulang": [0],
    "Semester_Cuti": [0],
    "Kehadiran_%": [90],
    "Status_Skripsi": [2],
    "Masa_Studi_Semester": [8]
})

hasil = model.predict(mahasiswa_baru)

print("\n=== PREDIKSI MAHASISWA BARU ===")
if hasil[0] == 1:
    print("Prediksi: Lulus Tepat Waktu")
else:
    print("Prediksi: Tidak Lulus Tepat Waktu")