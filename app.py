### Merancang website sederhana untuk streamlit (app.py)
# Import Streamlit
import streamlit as st
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
pilihan_aset = st.sidebar.selectbox("Pilih Aset:", ["Bitcoin (BTC-USD)", "Emas (GC=F)"])
if st.button("Prediksi Harga Besok"):
    st.info(f"Sedang menarik data real-time terbaru untuk {pilihan_aset}...")