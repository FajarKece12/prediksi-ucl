import streamlit as st
import joblib
import numpy as np
import pandas as pd

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
home_button = st.sidebar.button("Home")
tentang_button = st.sidebar.button("Data Asli")

# Periksa jika tombol "Home" ditekan
if home_button:
    st.session_state.selected_tab = "Home"
# Periksa jika tombol "Data Asli" ditekan
elif tentang_button:
    st.session_state.selected_tab = "Data Asli"

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

            # Prediksi menggunakan setiap model
            rf_pred = rf_model.predict(input_data)[0]
            c45_pred = c45_model.predict(input_data)[0]
            xgb_pred = xgb_model.predict(input_data)[0]

            # Tentukan status untuk setiap model
            rf_status = "Layak masuk UCL" if rf_pred == 1 else "Tidak layak masuk UCL"
            c45_status = "Layak masuk UCL" if c45_pred == 1 else "Tidak layak masuk UCL"
            xgb_status = "Layak masuk UCL" if xgb_pred == 1 else "Tidak layak masuk UCL"

            # Tampilkan hasil prediksi dengan tampilan yang menarik
            st.subheader(f"Hasil Prediksi untuk Klub {club_name}")
            st.markdown(f"**Random Forest Model:** {rf_status}")
            st.markdown(f"**C4.5 Model:** {c45_status}")
            st.markdown(f"**XGBoost Model:** {xgb_status}")

            # Berikan hasil dalam format tabel
            result_data = {
                "Model": ["Random Forest", "C4.5", "XGBoost"],
                "Prediksi": [rf_status, c45_status, xgb_status]
            }
            result_df = pd.DataFrame(result_data)
            st.dataframe(result_df)

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

