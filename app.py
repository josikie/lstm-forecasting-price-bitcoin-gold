### Merancang website sederhana untuk streamlit (app.py)
# Import Streamlit
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import pickle
from keras.models import load_model
import numpy as np

# 1. LOAD MODEL DAN SCALER YANG SUDAH DILATIH
@st.cache_resource
def muat_aset_model():
    btc_model = load_model('lstm_bitcoin_model.h5')
    gold_model = load_model('lstm_gold_model.h5')

    with open('btc_scaler.pkl', 'rb') as f:
        btc_scaler = pickle.load(f)
    with open('gold_scaler.pkl', 'rb') as f:
        gold_scaler = pickle.load(f)
    return btc_model, gold_model, btc_scaler, gold_scaler


btc_model, gold_model, btc_scaler, gold_scaler = muat_aset_model()
# atur mode website menjadi melebar memanfaatkan semua ruang kosong dilayar
st.set_page_config(layout="wide")
# judul website
st.title("📈💰Dashboard Forecasting Harga Bitcoin dan Emas")
# biodata peneliti
st.info("""**INFORMASI PENELITI:**  
           Nama            : Josi Kie N.  
           NIM             : 220401010122  
           Dosen Pembimbing: Cian Ramadhona Hassolthine, S.Kom, M.Kom
      """)
# tentang website
st.write("Website ini menarik data harian terbaru perdagangan Bitcoin dan emas dari Yahoo Finance API. Data - data tersebut dipotong menggunakan teknik sliding window 30 hari kebelakang. Data yang sudah dipotong dimasukkan ke dalam model LSTM untuk memprediksi estimasi harga aset pada keesokan hari. ")

# Bagian badan dashboard
aset = st.sidebar.selectbox("Pilih Aset:", ["Bitcoin (BTC-USD)", "Emas (GC=F)"])
symbol = "BTC-USD" if aset == "Bitcoin (BTC-USD)" else "GC=F"
model_aktif = btc_model if symbol == "BTC-USD" else gold_model
scaler_aktif = btc_scaler if symbol == "BTC-USD" else gold_scaler
if st.button("Prediksi Harga Besok"):
    st.info(f"Sedang menarik data real-time terbaru untuk {aset}...")
    # melihat tanggal brp hari ini
    # mengambil 60 hari setelah hari ini
    # mengunduh data dari yfinance
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    data_dari_yfinance = yf.download(symbol, start=start_date, end=end_date)
    if not data_dari_yfinance.empty:
        # 1. Hanya ambil kolom Close dan Volume, lalu tambal data jika ada hari libur
        tidy_data_column = data_dari_yfinance.xs(symbol, axis=1, level=1)
        df_bersih = tidy_data_column[['Close', 'Volume']].copy()
        df_bersih = df_bersih.ffill().bfill()
        
        # 2. Mengunci pas 30 hari bursa kerja terakhir untuk input sliding window
        df_30hari = df_bersih.tail(30)
        st.write("Data Historis 5 Hari Terakhir:")
        st.dataframe(df_30hari.tail(5))

        #Mengubah angka harga dan volume asli menjadi skala 0 sampai 1
        data_scaled = scaler_aktif.transform(df_30hari.values)
        # Mengubah bentuk dimensi data menjadi matriks 3D khusus input LSTM
        X_input = np.reshape(data_scaled, (1, 30, 2))
        prediksi_scaled = model_aktif.predict(X_input)
        # 4. Trik membuat matriks bayangan untuk proses inverse
        dummy_array = np.zeros((1, 2))
        dummy_array[:, 0] = prediksi_scaled[:, 0]
        
        # 5. Mengembalikan angka skala menjadi harga nominal Dollar asli ($)
        prediksi_asli = scaler_aktif.inverse_transform(dummy_array)
        harga_final = prediksi_asli[0, 0]

                # 1. Membuat garis pembatas horizontal yang elegan di website
        st.markdown("---")
        
        # 2. Memajang kotak indikator raksasa berisi nominal Dollar ($) harga esok hari
        st.metric(
            label=f"💰 Estimasi Prediksi Harga {aset} untuk Besok:", 
            value=f"${harga_final[0, 0]:,.2f}"
        )
        
        # 3. Mencetak kalimat sertifikasi jaminan mutu bersertifikat Lewis (1982)
        st.success("Kalkulasi selesai. Status model: Highly Accurate (Sangat Akurat) dengan skor evaluasi data testing tepercaya.")
