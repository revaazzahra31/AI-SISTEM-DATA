import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# =========================
# 1. Load Dataset
# =========================
df = pd.read_csv("dataset_kelulusan_mahasiswa_1000.csv", sep=";")

print("Preview Dataset:")
print(df.head())

# =========================
# 2. Preprocessing
# =========================

# Hapus kolom yang tidak dipakai untuk prediksi
df = df.drop(columns=["NIM", "Nama"])

# Encoding data kategorikal
le_jk = LabelEncoder()
le_org = LabelEncoder()

df["Jenis_Kelamin"] = le_jk.fit_transform(df["Jenis_Kelamin"])
df["Keaktifan_Organisasi"] = le_org.fit_transform(df["Keaktifan_Organisasi"])

# =========================
# 3. Menentukan Fitur dan Target
# =========================
X = df.drop(columns=["Ontime"])
y = df["Ontime"]

# =========================
# 4. Split Data
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# =========================
# 5. Modeling Decision Tree
# =========================
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    random_state=42
)

model.fit(X_train, y_train)

# =========================
# 6. Prediksi
# =========================
y_pred = model.predict(X_test)

# =========================
# 7. Evaluasi Model
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\n=== HASIL EVALUASI MODEL ===")
print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# =========================
# 8. Simpan Model
# =========================
joblib.dump(model, "model_decision_tree_kelulusan.pkl")
joblib.dump(le_jk, "encoder_jenis_kelamin.pkl")
joblib.dump(le_org, "encoder_organisasi.pkl")

print("\nModel berhasil disimpan.")