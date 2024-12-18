import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Load models
rf_model = joblib.load('models/rf_model.pkl')
c45_model = joblib.load('models/c45_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')

# Load data
df = pd.read_csv("dataset.csv")

# Mengatur tab default jika belum ada di session state
if "selected_tab" not in st.session_state:
    st.session_state.selected_tab = "Home"  # Tab pertama sebagai default

# Menambahkan judul di sidebar
st.sidebar.title("Menu Sidebar")
st.sidebar.subheader("Navigasi")
home_button = st.sidebar.button("Home")
tentang_button = st.sidebar.button("Data Asli")
korelasi_button = st.sidebar.button("Korelasi Atribut")
evaluasi_button = st.sidebar.button("Evaluasi Model")

st.sidebar.subheader("Model Prediksi")
model_selected = st.sidebar.radio("Pilih Model untuk Prediksi:", ("Random Forest", "C4.5", "XGBoost"))

# Periksa jika tombol ditekan
if home_button:
    st.session_state.selected_tab = "Home"
elif tentang_button:
    st.session_state.selected_tab = "Data Asli"
elif korelasi_button:
    st.session_state.selected_tab = "Korelasi Atribut"
elif evaluasi_button:
    st.session_state.selected_tab = "Evaluasi Model"

# Menampilkan konten sesuai dengan pilihan tab
if st.session_state.selected_tab == "Home":
    # Judul dan deskripsi aplikasi
    st.title("Prediksi Club Lolos UCL")
    st.markdown("""
        Aplikasi ini memungkinkan Anda untuk memprediksi apakah klub sepak bola akan lolos ke 
        Liga Champions Eropa (UCL) berdasarkan performa mereka.
    """)
    
    # Input form
    with st.form(key='input_form'):
        st.subheader("Masukkan Data Klub untuk Prediksi")
        club_name = st.text_input("Nama Klub:")
        pts_g = st.number_input("Points per Game:", min_value=0.0, format="%.2f")
        xg = st.number_input("Expected Goals:", min_value=0.0, format="%.2f")
        xga = st.number_input("Expected Goals Against:", min_value=0.0, format="%.2f")
        xgd = st.number_input("Expected Goal Difference:", min_value=0.0, format="%.2f")
        xgd_90 = st.number_input("xGD/90:", min_value=0.0, format="%.2f")
        w = st.number_input("WIN:", min_value=0)

        # Tombol untuk melakukan prediksi
        submit_button = st.form_submit_button("Prediksi")

    if submit_button:
        # Periksa apakah semua input ada
        if not club_name:
            st.error("Nama klub harus diisi!")
        else:
            # Buat data untuk prediksi
            input_data = np.array([[pts_g, xg, xga, xgd, xgd_90, w]])

            # Prediksi menggunakan model yang dipilih
            if model_selected == "Random Forest":
                prediction = rf_model.predict(input_data)[0]
                status = "Layak masuk UCL" if prediction == 1 else "Tidak layak masuk UCL"
                st.subheader(f"Hasil Prediksi dengan Random Forest untuk Klub {club_name}")
                st.markdown(f"**Status:** {status}")
            elif model_selected == "C4.5":
                prediction = c45_model.predict(input_data)[0]
                status = "Layak masuk UCL" if prediction == 1 else "Tidak layak masuk UCL"
                st.subheader(f"Hasil Prediksi dengan C4.5 untuk Klub {club_name}")
                st.markdown(f"**Status:** {status}")
            elif model_selected == "XGBoost":
                prediction = xgb_model.predict(input_data)[0]
                status = "Layak masuk UCL" if prediction == 1 else "Tidak layak masuk UCL"
                st.subheader(f"Hasil Prediksi dengan XGBoost untuk Klub {club_name}")
                st.markdown(f"**Status:** {status}")

elif st.session_state.selected_tab == "Data Asli":
    st.title("Data Asli Klub")
    st.markdown("""
        Berikut adalah data asli yang digunakan untuk pelatihan model prediksi.
    """)
    st.dataframe(df)

    st.markdown("""
    **Catatan:**
    - Data ini mencakup statistik performa klub berdasarkan berbagai metrik sepak bola.
    - Anda dapat melihat lebih detail mengenai data yang digunakan untuk membuat prediksi.
    """)

elif st.session_state.selected_tab == "Korelasi Atribut":
    st.title("Korelasi Antar Atribut")
    st.markdown("""
        Di tab ini, Anda dapat melihat hubungan antar atribut dalam dataset dalam bentuk matriks korelasi.
    """)

    # Langkah 3: Fungsi untuk menghitung korelasi pada kolom yang dipilih
    def calculate_selected_correlation(data):
        # Memilih kolom yang relevan untuk korelasi
        selected_columns = ['Pts', 'GF', 'GA', 'GD', 'xG']
        # Memfilter hanya kolom yang dipilih dari dataset
        selected_data = data[selected_columns]
        # Menghitung matriks korelasi
        correlation_matrix = selected_data.corr()
        return correlation_matrix

    # Menghitung korelasi
    try:
        correlation_matrix = calculate_selected_correlation(df)

        # Tampilkan matriks korelasi dalam bentuk tabel
        st.subheader("Matriks Korelasi")
        st.dataframe(correlation_matrix)

        # Visualisasi heatmap menggunakan seaborn
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
        st.pyplot(fig)
    except KeyError as e:
        st.error(f"Kolom yang dipilih tidak ditemukan dalam dataset: {e}")

elif st.session_state.selected_tab == "Evaluasi Model":
    st.title("Evaluasi Model")
    st.markdown("""
        Di tab ini, Anda dapat melihat evaluasi dari berbagai model (Random Forest, C4.5, XGBoost) berdasarkan data pengujian.
    """)

    # Load the dataset again to prepare for model evaluation
    data = pd.read_csv('dataset.csv')  # Gantilah dengan path dataset yang sesuai

    # Membuat kolom target 'UCL_Eligible' berdasarkan kolom 'LgRk'
    data['UCL_Eligible'] = data['LgRk'].apply(lambda x: 1 if x <= 4 else 0)

    # Pilih Fitur yang Digunakan
    features = ['Pts/G', 'xG', 'xGA', 'xGD', 'xGD/90', 'W']  # Fitur yang digunakan untuk prediksi

    # Konversi Kolom Kategorikal menjadi Numerik jika ada
    labelencoder = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = labelencoder.fit_transform(data[col])

    # Pisahkan Fitur (X) dan Label (y)
    X = data[features]  # Fitur yang dipilih
    y = data['UCL_Eligible']  # Kolom 'UCL_Eligible' sebagai label

    # Split Dataset (Training dan Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    #  Train Models
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    c45_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    c45_model.fit(X_train, y_train)

    xgb_model = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    models = [("Random Forest", rf_model), ("C4.5", c45_model), ("XGBoost", xgb_model)]

    for name, model in models:
        predictions = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]  # Probabilitas kelas positif untuk AUC

        # Menghitung dan menampilkan evaluasi
        st.subheader(f"Evaluasi Model {name}:")
        st.write(f"Akurasi: {accuracy_score(y_test, predictions):.2f}")
        st.write(f"F1 Score: {f1_score(y_test, predictions):.2f}")
        st.write(f"Presisi: {precision_score(y_test, predictions):.2f}")
        st.write(f"Recall: {recall_score(y_test, predictions):.2f}")
        st.write(f"AUC: {roc_auc_score(y_test, y_prob):.2f}")

        # Display Confusion Matrix
        cm = confusion_matrix(y_test, predictions)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", linewidths=0.5, ax=ax)
        ax.set_xlabel('Prediksi')
        ax.set_ylabel('Aktual')
        ax.set_title(f"Confusion Matrix - {name}")
        st.pyplot(fig)
