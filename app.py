### Merancang website sederhana untuk streamlit (app.py)
# Import Streamlit
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
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