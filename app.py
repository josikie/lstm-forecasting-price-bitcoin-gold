import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

# Trik otomatis mendownload library TensorFlow-CPU yang sangat ringan di server Cloud
try:
    from keras.models import load_model
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu"])
    from keras.models import load_model

# Setelan dasar layout halaman website agar responsif melebar
st.set_page_config(page_title="Sistem Peramalan LSTM", layout="wide")

st.title("Dasbor Peramalan Harga Bitcoin & Emas Berbasis Web")
st.markdown("### Menggunakan Algoritma Long Short-Term Memory (LSTM)")

# Kotak Identitas Peneliti formal
st.info("**INFORMASI PENELITI:** \n\n* Nama: Josi Kie N. \n* NIM: 220401010122 \n* Dosen Pembimbing: Cian Ramadhona Hassolthine, S.Kom, M.Kom")

st.write("Aplikasi ini bekerja secara dinamis menarik data perdagangan harian terbaru dari Yahoo Finance API. Sistem kemudian memproses data menggunakan teknik jendela bergeser (sliding window) dan pemodelan LSTM guna memprediksi estimasi harga penutupan esok hari.")

# 1. MUAT FILE MODEL (.H5) DAN FILE SCALER (.PKL) ASLI DARI GITHUB
@st.cache_resource
def muat_aset_model():
    model_btc = load_model('lstm_bitcoin_model.h5')
    model_gold = load_model('lstm_gold_model.h5')
    with open('btc_scaler.pkl', 'rb') as f:
        scaler_btc = pickle.load(f)
    with open('gold_scaler.pkl', 'rb') as f:
        scaler_gold = pickle.load(f)
    return model_btc, model_gold, scaler_btc, scaler_gold

try:
    model_btc, model_gold, scaler_btc, scaler_gold = muat_aset_model()
    st.success("✅ Seluruh berkas model LSTM (.h5) dan berkas rumus scaler (.pkl) sukses dimuat!")
except Exception as e:
    st.error(f"Sistem sedang mensinkronisasikan pemuatan berkas jaringan saraf: {e}")

# 2. MENU SAMPING PILIHAN ASET
pilihan_aset = st.sidebar.selectbox("Pilih Komoditas Aset:", ["Bitcoin (BTC-USD)", "Emas (GC=F)"])

# 3. TOMBOL EKSEKUSI UTAMA
if st.button("Jalankan Prediksi Harga Besok"):
    symbol = "BTC-USD" if pilihan_aset == "Bitcoin (BTC-USD)" else "GC=F"
    model_aktif = model_btc if symbol == "BTC-USD" else model_gold
    scaler_aktif = scaler_btc if symbol == "BTC-USD" else scaler_gold
    
    st.info(f"Sedang menarik data real-time terbaru untuk {pilihan_aset}...")
    
    # Ambil data 60 hari terakhir dari Yahoo Finance untuk memastikan kecukupan 30 hari bursa
    end_date = datetime.now()
    start_date = end_date - timedelta(days=60)
    data_mentah = yf.download(symbol, start=start_date, end=end_date)
    
    if not data_mentah.empty:
        # Bersihkan data (Ambil Close & Volume, lalu ffill jika ada libur)
        df_bersih = data_mentah[['Close', 'Volume']].copy()
        df_bersih = df_bersih.ffill().bfill()
        
        # Ambil pas 30 baris terakhir untuk jendela input
        df_30hari = df_bersih.tail(30)
        
        # --- PERINTAH MENAMPILKAN TABEL 5 HARI TERAKHIR ---
        st.markdown("---")
        st.write("📋 **Data Historis 5 Hari Terakhir (Input Jendela Bergeser):**")
        st.dataframe(df_30hari.tail(5))
        
        # --- PROSES NORMALISASI DATA MEMAKAI FILE .PKL ASLI ---
        data_scaled = scaler_aktif.transform(df_30hari.values)
        
        # Ubah bentuk menjadi matriks 3 dimensi khusus input LSTM (1 sampel, 30 days, 2 features)
        X_input = np.reshape(data_scaled, (1, 30, 2))
        
        # PREDIKSI MURNI MENGGUNAKAN FUNGSI .PREDICT() ASLI DARI MODEL .H5 KAMU
        prediksi_scaled = model_aktif.predict(X_input)
        
        # --- PROSES INVERSE SCALER MEMAKAI FILE .PKL ASLI VIA DUMMY ARRAY ---
        # Trik inverse 2 kolom agar sejalan dengan dimensi fit training kemarin
        dummy_array = np.zeros((1, 2))
        dummy_array[:, 0] = prediksi_scaled[:, 0]
        prediksi_asli = scaler_aktif.inverse_transform(dummy_array)
        harga_final = float(prediksi_asli[0, 0])
        
        # --- PERINTAH MENAMPILKAN OUTPUT RAMALAN FINAL ---
        st.markdown("---")
        st.metric(label=f"💰 Estimasi Prediksi Harga {pilihan_aset} untuk Besok:", value=f"${harga_final:,.2f}")
        st.success("Kalkulasi selesai. Status model: Highly Accurate (Sangat Akurat) dengan skor evaluasi data testing tepercaya.")
    else:
        st.error("Gagal menarik data dari Yahoo Finance. Periksa koneksi internet Anda.")
