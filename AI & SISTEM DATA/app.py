import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model_decision_tree_kelulusan.pkl")

st.set_page_config(
    page_title="Prediksi Kelulusan Tepat Waktu",
    layout="wide"
)

st.title("🎓 Sistem Prediksi Kelulusan Tepat Waktu Mahasiswa")
st.write("Menggunakan algoritma **Decision Tree**")

st.sidebar.title("Role Pengguna")

role = st.sidebar.selectbox(
    "Pilih Role",
    [
        "Admin",
        "Kaprodi",
        "Dosen Wali",
        "Mahasiswa",
        "Pimpinan Fakultas"
    ]
)

st.sidebar.write(f"Anda login sebagai: **{role}**")

if role == "Admin":
    menu = st.sidebar.radio(
        "Menu",
        ["Prediksi", "Upload Dataset", "Informasi Model"]
    )

elif role == "Kaprodi":
    menu = st.sidebar.radio(
        "Menu",
        ["Prediksi", "Dashboard Monitoring", "Daftar Mahasiswa Berisiko"]
    )

elif role == "Dosen Wali":
    menu = st.sidebar.radio(
        "Menu",
        ["Prediksi", "Monitoring Mahasiswa"]
    )

elif role == "Mahasiswa":
    menu = st.sidebar.radio(
        "Menu",
        ["Prediksi Pribadi"]
    )

else:
    menu = st.sidebar.radio(
        "Menu",
        ["Dashboard Fakultas", "Prediksi"]
    )

def form_prediksi():

    st.subheader("Form Prediksi Kelulusan Mahasiswa")

    kiri, kanan = st.columns(2)
    with kiri:

        tahun_masuk = st.selectbox(
            "Tahun Masuk",
            [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
        )

        jenis_kelamin = st.selectbox(
            "Jenis Kelamin",
            ["L", "P"]
        )

        ips1 = st.number_input(
            "IPS Semester 1",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )

        ips2 = st.number_input(
            "IPS Semester 2",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )

        ips3 = st.number_input(
            "IPS Semester 3",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )

        ips4 = st.number_input(
            "IPS Semester 4",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )

        ipk = st.number_input(
            "IPK",
            min_value=0.0,
            max_value=4.0,
            value=3.0
        )

    with kanan:

        mk_gagal = st.number_input(
            "Jumlah MK Gagal",
            min_value=0,
            max_value=20,
            value=0
        )

        sks_lulus = st.number_input(
            "SKS Lulus",
            min_value=0,
            max_value=144,
            value=120
        )

        status_skripsi_label = st.selectbox(
            "Status Skripsi",
            [
                "Belum Ambil",
                "Judul ACC",
                "Sempro",
                "Semhas",
                "Sidang Skripsi",
                "Lulus"
            ]
        )

        jumlah_bimbingan = st.number_input(
            "Jumlah Bimbingan",
            min_value=0,
            max_value=50,
            value=10
        )

        kehadiran = st.number_input(
            "Kehadiran (%)",
            min_value=0,
            max_value=100,
            value=80
        )

        organisasi = st.selectbox(
            "Keaktifan Organisasi",
            ["Aktif", "Tidak Aktif"]
        )

        status_cuti_label = st.selectbox(
            "Status Cuti",
            ["Tidak Pernah Cuti", "Pernah Cuti"]
        )

    status_skripsi_dict = {
        "Belum Ambil": 0,
        "Judul ACC": 1,
        "Sempro": 2,
        "Semhas": 3,
        "Sidang Skripsi": 4,
        "Lulus": 5
    }

    status_skripsi = status_skripsi_dict[status_skripsi_label]

    status_cuti = 0 if status_cuti_label == "Tidak Pernah Cuti" else 1

    organisasi_encoded = 0 if organisasi == "Aktif" else 1

    jenis_kelamin_encoded = 1 if jenis_kelamin == "P" else 0

    rata_ips = round((ips1 + ips2 + ips3 + ips4) / 4, 2)
    progress_sks = round(sks_lulus / 144, 2)
    tren_ip = round(ips4 - ips1, 2)

    jenis_kelamin_encoded = 1 if jenis_kelamin == "P" else 0
    organisasi_encoded = 0 if organisasi == "Aktif" else 1

    data_input = pd.DataFrame({
        "Tahun_Masuk": [tahun_masuk],
        "Jenis_Kelamin": [jenis_kelamin_encoded],
        "IPS1": [ips1],
        "IPS2": [ips2],
        "IPS3": [ips3],
        "IPS4": [ips4],
        "IPK": [ipk],
        "MK_Gagal": [mk_gagal],
        "SKS_Lulus": [sks_lulus],
        "Status_Skripsi": [status_skripsi],
        "Jumlah_Bimbingan": [jumlah_bimbingan],
        "Kehadiran": [kehadiran],
        "Keaktifan_Organisasi": [organisasi_encoded],
        "Status_Cuti": [status_cuti],
        "Rata_IPS": [rata_ips],
        "Progress_SKS": [progress_sks],
        "Tren_IP": [tren_ip]
    })

    st.write("Data Input:")
    st.dataframe(data_input)

    if st.button("Prediksi Kelulusan"):
        hasil = model.predict(data_input)[0]
        probabilitas = model.predict_proba(data_input)[0]

        if hasil == 1:
            st.success("Mahasiswa diprediksi: LULUS TEPAT WAKTU")
        else:
            st.error("Mahasiswa diprediksi: TIDAK LULUS TEPAT WAKTU")

        st.write("Probabilitas Tidak Ontime:", round(probabilitas[0] * 100, 2), "%")
        st.write("Probabilitas Ontime:", round(probabilitas[1] * 100, 2), "%")

        st.subheader("Rekomendasi Tindakan")

        if hasil == 0:
            st.warning("""
            Mahasiswa perlu mendapatkan perhatian khusus:
            - Tingkatkan jumlah bimbingan skripsi
            - Evaluasi mata kuliah gagal
            - Monitoring kehadiran
            - Konseling akademik dengan dosen wali
            """)
        else:
            st.info("""
            Mahasiswa berada dalam kondisi baik:
            - Pertahankan performa akademik
            - Lanjutkan progres skripsi
            - Tetap aktif mengikuti bimbingan
            """)

if menu in ["Prediksi", "Prediksi Pribadi"]:
    form_prediksi()

elif menu == "Upload Dataset":
    st.subheader("Upload Dataset Mahasiswa")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, sep=None, engine="python")
        st.dataframe(df_upload)
        st.success("Dataset berhasil diupload.")

elif menu == "Informasi Model":
    st.subheader("Informasi Model")
    st.write("""
    Model yang digunakan adalah Decision Tree Classifier.

    Fitur utama:
    - IPS Semester 1 sampai Semester 6
    - IPK
    - Jumlah MK gagal
    - SKS lulus
    - Status skripsi
    - Jumlah bimbingan
    - Kehadiran
    - Rata-rata IPS
    - Progress SKS
    - Tren IP
    """)

elif menu in ["Dashboard Monitoring", "Dashboard Fakultas"]:
    st.subheader("Dashboard Monitoring Mahasiswa")

    try:
        df = pd.read_csv(
            "dataset_kelulusan_mahasiswa_1000.csv",
            sep=None,
            engine="python"
        )

        total = len(df)
        ontime = df["Ontime"].sum()
        tidak_ontime = total - ontime

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Mahasiswa", total)
        col2.metric("Prediksi Ontime", ontime)
        col3.metric("Tidak Ontime", tidak_ontime)

        st.bar_chart(df["Ontime"].value_counts())

        st.subheader("Data Mahasiswa")
        st.dataframe(df)

    except Exception as e:
        st.error(e)

elif menu in ["Daftar Mahasiswa Berisiko", "Monitoring Mahasiswa"]:
    st.subheader("Daftar Mahasiswa Berisiko")

    try:
        df = pd.read_csv(
            "dataset_kelulusan_mahasiswa_1000.csv",
            sep=None,
            engine="python"
        )

        risiko = df[df["Ontime"] == 0]

        st.write("Jumlah mahasiswa berisiko:", len(risiko))
        st.dataframe(risiko)

    except Exception as e:
        st.error(e)