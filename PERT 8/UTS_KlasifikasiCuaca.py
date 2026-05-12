import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Dataset Cuaca.csv", sep=';')

print("5 data pertama:")
print(df.head())
print("\nInfo data:")
print(df.info())
print("\nJumlah data tiap kelas:")
print(df["Cuaca"].value_counts())

def clean_number(x):
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

X = df[["Suhu", "Kelembaban", "Angin"]]   
y = df["Cuaca"]                           

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,        
    random_state=42,      
    stratify=y            
)

print("\nJumlah data training:", len(X_train))
print("Jumlah data testing :", len(X_test))

model = DecisionTreeClassifier(
    criterion="entropy",  
    max_depth=4,          
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

hasil = X_test.copy()
hasil["Asli"] = y_test.values
hasil["Prediksi"] = y_pred

print("\n=== Hasil Prediksi Data testing ===")
print(hasil.head(10))

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("\n=== HASIL EVALUASI ===")
print("Akurasi Model:", round(acc * 100,2), "%")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(report)

plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=model.classes_,
    filled=True
)
plt.title("Decision Tree untuk Klasifikasi Cuaca")
plt.show()

data_baru = pd.DataFrame({
    "Suhu": [28],
    "Kelembaban": [7.5],
    "Angin": [10]
})
 
hasil_prediksi = model.predict(data_baru)
print("\nPrediksi data baru:", hasil_prediksi[0])
