import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# 1. KONFIGURASI HALAMAN
st.set_page_config(page_title="Monitoring Pencegahan DBD", page_icon="🛡️", layout="wide")

# Style CSS Profesional
st.markdown("""
    <style>
    .main { background-color: #f4f7f6; }
    h1, h2, h3 { color: #1b5e20; font-weight: bold; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 10px solid #2e7d32; }
    </style>
    """, unsafe_allow_html=True)

# 2. LOAD DATA OTOMATIS
@st.cache_data
def load_automated_data():
    try:
        model = joblib.load('model_dbd_rf.pkl')
        features = joblib.load('features_list.pkl')
        df = pd.read_csv('df_gabungan.csv')
        return model, features, df
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        return None, None, None

model, features, df_master = load_automated_data()

st.title("🛡️ Sistem Otomatis Strategi Pencegahan DBD")
st.markdown("---")

if df_master is not None:
    st.sidebar.title("📍 Pilih Wilayah")
    list_kota = sorted(df_master['Kabupaten/Kota'].unique())
    selected_kota = st.sidebar.selectbox("Pilih Kabupaten/Kota:", list_kota)

    # Ambil data terbaru otomatis
    data_terbaru = df_master[df_master['Kabupaten/Kota'] == selected_kota].iloc[-1]

    st.subheader(f"📊 Kondisi Ekologis Terkini: {selected_kota} (Tahun {int(data_terbaru['Tahun'])})")

    c1, c2, c3, c4 = st.columns(4)
    hujan_val = data_terbaru['curah_hujan_mm']
    sampah_val = data_terbaru['timbulan_sampah_ton']
    kepadatan_val = data_terbaru['kepadatan_penduduk_km2']
    sanitasi_val = data_terbaru['akses_sanitasi_layak_persen']

    c1.metric("Curah Hujan", f"{hujan_val:.0f} mm")
    c2.metric("Sampah", f"{sampah_val:,.0f} Ton")
    c3.metric("Kepadatan", f"{kepadatan_val:,.0f} Jiwa/km²")
    c4.metric("Sanitasi", f"{sanitasi_val:.1f}%")

    st.markdown("---")
    st.header("🎯 Rekomendasi Langkah Pencegahan Strategis")

    col_l, col_r = st.columns(2)
    with col_l:
        st.success("#### 🌧️ Mitigasi Dampak Curah Hujan")
        # Sesuaikan threshold 2000 dengan skala data asli Anda
        if hujan_val > 2000:
            st.write("**Metode: PSN 3M Plus Intensif**")
            st.write("- **Kuras:** Minimal 2x seminggu.")
            st.write("- **Tutup:** Tandon air harus rapat.")
        else:
            st.write("- Fokus pada pembersihan wadah air indoor.")

    with col_r:
        st.success("#### 👥 Pencegahan Area Padat")
        # Sesuaikan threshold 1200 dengan skala data asli Anda
        if kepadatan_val > 1200:
            st.write("**Metode: Gerakan G1W1J**")
            st.write("- **G1W1J:** 1 Rumah 1 Jumantik mandiri.")
            st.write("- **Fisik:** Pasang kawat nyamuk pada ventilasi.")
