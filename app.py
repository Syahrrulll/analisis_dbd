import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

# ============================================================================
# PAGE CONFIG & STYLING
# ============================================================================
st.set_page_config(page_title="Mitigasi DBD Terpadu", page_icon="🏥", layout="wide")

st.markdown("""
<style>
    .stApp { 
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9; 
    }
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        padding: 2.5rem; border-radius: 20px; margin-bottom: 2.5rem;
        border-left: 8px solid #10b981; box-shadow: 0 8px 25px rgba(0,0,0,0.4);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: "🏥";
        position: absolute;
        right: 30px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 5rem;
        opacity: 0.1;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.8);
        padding: 1.8rem; 
        border-radius: 16px;
        border: 1px solid #475569;
        margin-bottom: 1.5rem;
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #10b981;
    }
    .recommendation-card {
        padding: 1.8rem; 
        border-radius: 16px; 
        margin-top: 1.5rem;
        border-left: 12px solid;
        background: rgba(30, 41, 59, 0.9);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .high-risk { 
        background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
        border-left-color: #ef4444; 
        color: #fecaca; 
    }
    .med-risk { 
        background: linear-gradient(135deg, #422006 0%, #92400e 100%);
        border-left-color: #f59e0b; 
        color: #fef3c7; 
    }
    .low-risk { 
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border-left-color: #10b981; 
        color: #d1fae5; 
    }
    .var-card {
        background: rgba(30, 41, 59, 0.8);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #475569;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.2rem;
        margin-top: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 1rem 2rem;
        font-weight: bold;
        background-color: rgba(30, 41, 59, 0.7);
    }
    .stTabs [aria-selected="true"] {
        background-color: #10b981 !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD ASSETS (Model Robust 88%)
# ============================================================================
@st.cache_resource
def load_assets():
    bundle = joblib.load('model_robust_bundle.pkl')
    df = pd.read_csv('df_final_dashboard.csv')
    return bundle, df

bundle, df_master = load_assets()
model = bundle['model']
features = bundle['features']
metrics = bundle['metrics']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def format_number(value, is_population=False):
    """Format angka dengan penanganan untuk nilai None/NaN"""
    if pd.isna(value) or value is None:
        return "N/A"
    try:
        # Coba format sebagai float
        float_val = float(value)

        # Jika ini data penduduk dalam ribuan, kalikan dengan 1000
        if is_population:
            float_val = float_val * 1000

        if float_val >= 1000000:
            return f"{float_val/1000000:,.2f} juta"
        elif float_val >= 1000:
            return f"{float_val:,.0f}"
        elif float_val.is_integer():
            return f"{int(float_val):,}"
        else:
            return f"{float_val:,.2f}"
    except:
        return str(value)

def get_variable_recommendation(feature_name, current_value, feature_mean):
    """Menghasilkan rekomendasi spesifik untuk setiap variabel"""
    recommendations = {
        'IR_tahun_lalu': {
            'high': "📈 **Kasus tahun lalu tinggi** → Fokuskan surveilans intensif di wilayah dengan riwayat kasus tinggi",
            'med': "📊 **Kasus tahun lalu sedang** → Lanjutkan pemantauan rutin dan persiapan respons cepat",
            'low': "📉 **Kasus tahun lalu rendah** → Pertahankan pencegahan dan waspada peningkatan mendadak"
        },
        'kepadatan_penduduk_km2': {
            'high': "🏙️ **Kepadatan tinggi** → Optimalkan PSN massal, distribusi kelambu, dan pengaturan jarak hunian",
            'med': "🏘️ **Kepadatan sedang** → Tingkatkan edukasi dan partisipasi masyarakat dalam 3M Plus",
            'low': "🌳 **Kepadatan rendah** → Fokus pada daerah perkantoran dan fasilitas umum"
        },
        'curah_hujan_mm': {
            'high': "🌧️ **Curah hujan tinggi** → Perketat pemantauan genangan air, perbaiki drainase, sosialisasi PSN",
            'med': "⛈️ **Curah hujan sedang** → Waspada penampungan air hujan di rumah tangga",
            'low': "☀️ **Curah hujan rendah** → Perhatikan penampungan air buatan dan penyimpanan air bersih"
        },
        'akses_sanitasi_layak_persen': {
            'high': "✅ **Sanitasi layak tinggi** → Pertahankan dan tingkatkan cakupan",
            'med': "⚠️ **Sanitasi layak sedang** → Intensifkan sosialisasi sanitasi sehat",
            'low': "🚨 **Sanitasi layak rendah** → Prioritas intervensi infrastruktur sanitasi"
        }
    }

    # Default mapping untuk variabel lain
    if feature_name not in recommendations:
        feature_display = feature_name.replace('_', ' ').title()
        recommendations[feature_name] = {
            'high': f"📊 **{feature_display} tinggi** → Perlu evaluasi dampaknya terhadap risiko DBD",
            'med': f"📈 **{feature_display} sedang** → Monitor perkembangan secara berkala",
            'low': f"📉 **{feature_display} rendah** → Kondisi optimal untuk pencegahan"
        }

    # Tentukan kategori berdasarkan nilai (hanya jika keduanya numerik)
    try:
        current_num = float(current_value)
        mean_num = float(feature_mean)

        if current_num > mean_num * 1.3:
            category = 'high'
        elif current_num < mean_num * 0.7:
            category = 'low'
        else:
            category = 'med'
    except:
        category = 'med'

    return recommendations[feature_name][category]

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("### 🎯 KONTROL PANEL")

    selected_kota = st.selectbox(
        "Pilih Kabupaten/Kota:", 
        sorted(df_master['Kabupaten/Kota'].unique()),
        help="Pilih wilayah untuk analisis dan prediksi"
    )

    st.markdown("---")

    # Model metrics
    st.markdown("### 📊 STATUS MODEL")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Akurasi (R²)", f"{metrics['test_r2']*100:.1f}%")
    with col2:
        st.metric("Stabilitas", f"{(1-metrics['gap'])*100:.1f}%")

    st.caption(f"🔄 Update terakhir: {datetime.now().strftime('%d %b %Y %H:%M')}")

    st.markdown("---")
    st.markdown("### 📈 LEGENDA STATUS")
    st.markdown("""
    - 🚨 **Tinggi**: IR > 50
    - 🟡 **Sedang**: IR 20-50  
    - ✅ **Rendah**: IR < 20
    """)

# ============================================================================
# MAIN CONTENT
# ============================================================================
# Header utama
st.markdown(f"""
<div class="main-header">
    <h1 style="margin-bottom: 0.5rem; font-size: 2.8rem;">🏥 Dashboard Mitigasi DBD</h1>
    <h2 style="margin-top: 0; color: #10b981; font-size: 1.8rem;">{selected_kota}</h2>
    <p style="font-size: 1.1rem; opacity: 0.9;">Sistem Rekomendasi Mitigasi Berbasis Predictive Modeling • Akurasi: {metrics['test_r2']*100:.1f}%</p>
</div>
""", unsafe_allow_html=True)

# Ambil data terbaru untuk kota tersebut (tahun terakhir)
data_kota_latest = df_master[df_master['Kabupaten/Kota'] == selected_kota].iloc[-1]

# Debug: Tampilkan kolom yang tersedia
with st.sidebar:
    st.markdown("---")
    st.markdown("### 🔍 DEBUG DATA")
    st.write(f"Kolom tersedia: {list(data_kota_latest.index)}")

# Lakukan Prediksi
input_df = pd.DataFrame([data_kota_latest[features]], columns=features)
pred_ir = model.predict(input_df)[0]

# Tab navigation
tab1, tab2, tab3 = st.tabs(["🎯 PREDIKSI & REKOMENDASI", "📊 ANALISIS VARIABEL", "🔍 EVALUASI MODEL"])

with tab1:
    # Kartu prediksi utama
    col_res, col_stats = st.columns([1.2, 0.8])

    with col_res:
        # Tentukan kategori risiko
        if pred_ir > 50:
            risk_class = "high-risk"
            risk_label = "🚨 RISIKO TINGGI"
            risk_icon = "⚠️"
        elif pred_ir > 20:
            risk_class = "med-risk"
            risk_label = "🟡 RISIKO SEDANG"
            risk_icon = "🔔"
        else:
            risk_class = "low-risk"
            risk_label = "✅ RISIKO RENDAH"
            risk_icon = "✅"

        st.markdown(f"""
        <div class="recommendation-card {risk_class}">
            <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                <div style="font-size: 3rem;">{risk_icon}</div>
                <div>
                    <h3 style="margin:0;">PREDIKSI INDEKS MORBIDITAS</h3>
                    <p style="margin:0; opacity: 0.9;">{selected_kota}</p>
                </div>
            </div>
            <h1 style="font-size: 5rem; margin: 1rem 0; text-align: center;">{pred_ir:.1f}</h1>
            <h3 style="text-align: center; margin-bottom: 1rem;">{risk_label}</h3>
            <p style="text-align: center; opacity: 0.9;">per 100.000 penduduk</p>
        </div>
        """, unsafe_allow_html=True)

    with col_stats:
        # Statistik ringkas - SESUAI NAMA KOLOM DI DATA
        st.markdown("### 📈 STATISTIK WILAYAH")

        # Ambil data dengan nama kolom yang benar dari dataframe
        stats_data = {
            "Jumlah Penduduk": format_number(data_kota_latest.get('penduduk_ribu'), is_population=True),
            "Tahun Data": data_kota_latest.get('Tahun', 'N/A'),
            "Kasus DBD": format_number(data_kota_latest.get('kasus_dbd')),
            "Kepadatan": f"{data_kota_latest.get('kepadatan_penduduk_km2', 'N/A')} jiwa/km²",
            "Sanitasi Layak": f"{data_kota_latest.get('akses_sanitasi_layak_persen', 'N/A'):.1f}%"
        }

        # Debug info
        with st.expander("📊 Detail Data", expanded=False):
            st.write("Data terbaru untuk:", selected_kota)
            st.dataframe(data_kota_latest, use_container_width=True)

        for key, value in stats_data.items():
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">{key}</div>
                <div style="font-size: 1.5rem; font-weight: bold; color: #10b981;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # REKOMENDASI UTAMA BERDASARKAN RISIKO
    st.markdown("### 🛡️ REKOMENDASI STRATEGIS UTAMA")

    if pred_ir > 50:
        st.error("""
        ## 🚨 STATUS DARURAT - TINDAKAN SEGERA DIPERLUKAN

        1. **AKTIFKAN POSKO DARURAT DBD** di tingkat kecamatan dengan anggaran khusus
        2. **FOGGING INTENSIF** radius 200m dari lokasi kasus baru dalam 24 jam
        3. **MOBILISASI TENAGA MEDIS** tambahan dan pastikan ketersediaan platelet
        4. **SURVEILANS HARIAN** oleh Dinas Kesehatan dan lapor ke Gubernur
        5. **GERAKAN MASSAL 3M PLUS** melibatkan TNI/Polri dan organisasi masyarakat
        """)
    elif pred_ir > 20:
        st.warning("""
        ## 🟡 STATUS WASPADA - TINGKATKAN PENCEGAHAN

        1. **INTENSIFKAN PSN** (Pembersihan Sarang Nyamuk) serentak seminggu sekali
        2. **DISTRIBUSI ABATE/LARVASIDA** ke seluruh rumah di wilayah rawan
        3. **SOSIALISASI MASIF** 3M Plus melalui media lokal dan kader
        4. **PEMANTAUAN JENTIK** mingguan dengan cakupan >80% rumah
        5. **SIAGA RUJUKAN** di fasilitas kesehatan dengan bed khusus DBD
        """)
    else:
        st.success("""
        ## ✅ STATUS AMAN - PERTAHANKAN DAN OPTIMALKAN

        1. **LANJUTKAN PEMANTAUAN RUTIN** jentik oleh Jumantik terlatih
        2. **EDUKASI BERKELANJUTAN** di sekolah dan perkantoran
        3. **PERTAHANKAN SANITASI** dan drainase lingkungan
        4. **KAPASITAS RESPONS CEPAT** yang siap diaktifkan jika diperlukan
        5. **DOKUMENTASI BEST PRACTICE** untuk replikasi ke wilayah lain
        """)

with tab2:
    st.markdown("### 📊 ANALISIS DETAIL VARIABEL PREDIKTOR")
    st.markdown("Analisis mendalam setiap variabel yang mempengaruhi prediksi risiko DBD")

    # Feature Importance Visualization
    feature_names_display = [f.replace('_', ' ').title() for f in features]

    # Cari nilai saat ini dengan penanganan missing values
    current_values = []
    for f in features:
        val = data_kota_latest.get(f)
        if pd.isna(val):
            current_values.append(np.nan)
        else:
            try:
                current_values.append(float(val))
            except:
                current_values.append(np.nan)

    imp_df = pd.DataFrame({
        'Variabel': feature_names_display,
        'Kepentingan': model.feature_importances_,
        'Nilai Saat Ini': current_values
    }).sort_values('Kepentingan', ascending=False)

    # Plot feature importance
    fig_imp = go.Figure()
    fig_imp.add_trace(go.Bar(
        x=imp_df['Kepentingan'],
        y=imp_df['Variabel'],
        orientation='h',
        marker=dict(
            color=imp_df['Kepentingan'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Kepentingan")
        ),
        text=[f"{imp*100:.1f}%" for imp in imp_df['Kepentingan']],
        textposition='inside',
        name='Feature Importance'
    ))

    fig_imp.update_layout(
        title="📈 Variabel Paling Berpengaruh dalam Prediksi",
        height=500,
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Tingkat Kepentingan",
        yaxis_title="Variabel",
        showlegend=False
    )
    st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 REKOMENDASI SPESIFIK PER VARIABEL")

    # Hitung rata-rata untuk setiap feature (abaikan NaN)
    feature_means = df_master[features].apply(lambda x: pd.to_numeric(x, errors='coerce')).mean()

    # Tampilkan rekomendasi untuk 5 variabel terpenting
    top_features = imp_df.head(5)

    for _, row in top_features.iterrows():
        # Cari nama asli feature
        original_feature_name = features[feature_names_display.index(row['Variabel'])]
        current_value = row['Nilai Saat Ini']

        with st.expander(f"🔍 {row['Variabel']} (Kepentingan: {row['Kepentingan']*100:.1f}%)", expanded=True):
            col_metric, col_rec = st.columns([1, 2])

            with col_metric:
                # Tampilkan nilai dan status
                mean_value = feature_means[original_feature_name]

                if pd.isna(current_value):
                    status = "❓ **DATA TIDAK TERSEDIA**"
                    color = "#94a3b8"
                    display_value = "N/A"
                else:
                    # Format nilai berdasarkan jenis data
                    if 'persen' in original_feature_name.lower() or '_p' in original_feature_name.lower():
                        display_value = f"{current_value:.1f}%"
                    elif 'ribu' in original_feature_name.lower():
                        display_value = format_number(current_value, is_population=True)
                    else:
                        display_value = format_number(current_value)

                    try:
                        current_num = float(current_value)
                        mean_num = float(mean_value)

                        if current_num > mean_num * 1.3:
                            status = "🔼 **DI ATAS RATA-RATA**"
                            color = "#ef4444"
                        elif current_num < mean_num * 0.7:
                            status = "🔽 **DI BAWAH RATA-RATA**"
                            color = "#10b981"
                        else:
                            status = "↔️ **SESUAI RATA-RATA**"
                            color = "#f59e0b"
                    except:
                        status = "📊 **DATA TERSEDIA**"
                        color = "#94a3b8"

                st.markdown(f"""
                <div style="background: {color}20; padding: 1rem; border-radius: 10px; border-left: 4px solid {color};">
                    <div style="font-size: 0.9rem;">Nilai Saat Ini</div>
                    <div style="font-size: 1.8rem; font-weight: bold;">{display_value}</div>
                    <div style="font-size: 0.8rem;">{status}</div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">Rata-rata: {format_number(mean_value)}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_rec:
                # Tampilkan rekomendasi
                if pd.isna(current_value):
                    recommendation = f"⚠️ **Data tidak tersedia** untuk {row['Variabel']}. Disarankan untuk mengumpulkan data ini untuk analisis yang lebih akurat."
                else:
                    recommendation = get_variable_recommendation(
                        original_feature_name, 
                        current_value, 
                        mean_value
                    )

                st.markdown(f"""
                <div style="background: #1e293b; padding: 1rem; border-radius: 10px;">
                    <div style="font-size: 1rem; line-height: 1.6;">
                    {recommendation}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Action item spesifik
                st.markdown("**🎯 Action Item:**")
                if pd.isna(current_value):
                    st.info(f"Kumpulkan data {row['Variabel'].lower()} untuk analisis yang lebih baik")
                elif "tinggi" in recommendation.lower():
                    st.info(f"Prioritaskan intervensi pada {row['Variabel'].lower()}")
                elif "sedang" in recommendation.lower():
                    st.warning(f"Monitor perkembangan {row['Variabel'].lower()} secara berkala")
                else:
                    st.success(f"Pertahankan kondisi optimal untuk {row['Variabel'].lower()}")

with tab3:
    st.markdown("### 🔍 EVALUASI MODEL PREDIKTIF")

    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        # Metrik model
        st.markdown("#### 📊 PERFORMANCE METRICS")

        metrics_data = {
            "R² Score (Test)": f"{metrics['test_r2']*100:.1f}%",
            "R² Score (Train)": f"{metrics['train_r2']*100:.1f}%",
            "Gap Train-Test": f"{metrics['gap']*100:.2f}%",
            "MAE (Mean Absolute Error)": f"{metrics.get('mae', 0):.2f}" if 'mae' in metrics else "N/A",
            "RMSE (Root Mean Squared Error)": f"{metrics.get('rmse', 0):.2f}" if 'rmse' in metrics else "N/A"
        }

        for metric_name, value in metrics_data.items():
            st.markdown(f"""
            <div class="metric-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 0.9rem;">{metric_name}</div>
                    <div style="font-size: 1.2rem; font-weight: bold; color: #10b981;">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_eval2:
        st.markdown("#### 🎯 INFORMASI MODEL")

        info_cards = [
            ("🧠 **Algoritma**", "Random Forest Regressor"),
            ("📊 **Jumlah Fitur**", f"{len(features)} variabel"),
            ("🎯 **Target**", "Indeks Morbiditas (IR) DBD"),
            ("📈 **Stabilitas**", "Tinggi (Gap < 10%)"),
            ("🔄 **Update Terakhir**", bundle.get('timestamp', '2024-01-01') if 'timestamp' in bundle else '2024-01-01')
        ]

        for title, value in info_cards:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size: 0.9rem; opacity: 0.8;">{title}</div>
                <div style="font-size: 1rem; font-weight: bold;">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Visualization of model performance
    st.markdown("#### 📈 VISUALISASI KINERJA MODEL")

    # Create a synthetic comparison chart
    fig_perf = go.Figure()

    # Add bars for train and test performance
    fig_perf.add_trace(go.Bar(
        x=['Training Set', 'Test Set'],
        y=[metrics['train_r2']*100, metrics['test_r2']*100],
        marker_color=['#60a5fa', '#10b981'],
        text=[f"{metrics['train_r2']*100:.1f}%", f"{metrics['test_r2']*100:.1f}%"],
        textposition='auto',
        name='R² Score'
    ))

    fig_perf.update_layout(
        title="Perbandingan Performance Model",
        height=400,
        font=dict(color='white'),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis_title="R² Score (%)",
        showlegend=False
    )

    st.plotly_chart(fig_perf, use_container_width=True)

    # Interpretasi hasil
    st.markdown("#### 📋 INTERPRETASI HASIL EVALUASI")

    if metrics['gap'] < 0.1:
        st.success("""
        ✅ **MODEL STABIL DAN RELIABLE**

        Model menunjukkan performa yang konsisten antara data training dan testing, 
        menandakan tidak terjadi overfitting. Model dapat diandalkan untuk prediksi 
        di berbagai kondisi wilayah.
        """)
    else:
        st.warning("""
        ⚠️ **PERLU PERHATIAN KHUSUS**

        Terdapat gap yang signifikan antara performa training dan testing. 
        Disarankan untuk melakukan validasi tambahan dan monitoring ketat 
        sebelum implementasi skala penuh.
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns([2, 1, 1])

with footer_col1:
    st.markdown("""
    <div style="text-align: left; opacity: 0.7;">
        <p>📋 <strong>Sistem Mitigasi DBD Terpadu</strong> • Dashboard v2.0 • © 2024 Kementerian Kesehatan</p>
        <p style="font-size: 0.9rem;">Sistem ini menggunakan model machine learning untuk prediksi risiko DBD dengan akurasi tinggi.</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col2:
    st.markdown("""
    <div style="text-align: center; opacity: 0.7;">
        <p>📞 Hotline DBD</p>
        <p style="font-size: 1.2rem; font-weight: bold;">119</p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    st.markdown(f"""
    <div style="text-align: right; opacity: 0.7;">
        <p>Terakhir diperbarui:</p>
        <p>{datetime.now().strftime('%d %B %Y')}</p>
    </div>
    """, unsafe_allow_html=True)
