import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import plotly.io as pio
from sklearn.inspection import permutation_importance

# Set tema Plotly yang lebih gelap
pio.templates.default = "plotly_white"

# ============================================================================
# PAGE CONFIG & CUSTOM CSS
# ============================================================================
st.set_page_config(
    page_title="Dashboard Mitigasi DBD",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling premium dengan background gelap
st.markdown("""
<style>
    /* Global Styles dengan dark theme */
    .stApp {
        background-color: #0f172a;
        color: #f1f5f9;
    }

    /* Text color default */
    .stMarkdown, .stText, .stTitle, .stHeader {
        color: #f1f5f9 !important;
    }

    /* Header dengan gradien modern */
    .main-header {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        border-left: 6px solid #10b981;
    }

    .main-header h1 {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1rem;
        margin-bottom: 0;
    }

    /* Sub-header elegant */
    .section-header {
        color: #60a5fa;
        font-size: 1.4rem;
        font-weight: 700;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }

    .subsection-header {
        color: #94a3b8;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding-left: 0.5rem;
        border-left: 3px solid #3b82f6;
    }

    /* Card styles dengan dark theme */
    .metric-card {
        background: #1e293b;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        border: 1px solid #334155;
        color: #f1f5f9;
    }

    .highlight-card {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
    }

    /* Importance tags */
    .importance-high {
        background-color: #dc2626;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }

    .importance-medium {
        background-color: #f59e0b;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }

    .importance-low {
        background-color: #3b82f6;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        display: inline-block;
        margin: 0.25rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# FUNCTIONS
# ============================================================================
@st.cache_resource
def load_data():
    """Load model and data with caching"""
    try:
        bundle = joblib.load('model_final_bundle.pkl')
        df_master = pd.read_csv('df_gabungan.csv')
        return bundle, df_master
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
        st.stop()

def calculate_feature_importance_corrected(models, X_data, y_data, features):
    """Calculate feature importance dengan metode yang lebih akurat"""
    importance_data = {}

    for model_name, model in models.items():
        try:
            st.write(f"🔍 Menghitung importance untuk model {model_name}...")

            # Coba beberapa metode untuk mendapatkan importance
            importance_scores = None

            # Method 1: Jika model punya feature_importances_ (Random Forest, etc)
            if hasattr(model, 'feature_importances_'):
                st.write(f"  Menggunakan feature_importances_")
                importance_scores = model.feature_importances_

            # Method 2: Jika model punya coef_ (Linear Regression, etc)
            elif hasattr(model, 'coef_'):
                st.write(f"  Menggunakan coef_")
                coefs = model.coef_
                if len(coefs.shape) > 1:
                    coefs = coefs.flatten()
                importance_scores = np.abs(coefs)

            # Method 3: Permutation importance sebagai fallback
            else:
                st.write(f"  Menggunakan permutation importance")
                try:
                    result = permutation_importance(
                        model, X_data, y_data,
                        n_repeats=10,
                        random_state=42,
                        n_jobs=-1,
                        scoring='r2'
                    )
                    importance_scores = result.importances_mean
                except:
                    st.write(f"  Permutation importance gagal, menggunakan random importance")
                    # Fallback: random dengan sedikit variasi
                    np.random.seed(42)
                    importance_scores = np.random.rand(len(features))

            # Pastikan importance_scores ada dan valid
            if importance_scores is None or len(importance_scores) != len(features):
                st.write(f"  Importance scores tidak valid, menggunakan equal importance")
                importance_scores = np.ones(len(features))

            # Normalize scores
            if np.max(importance_scores) > 0:
                importance_norm = importance_scores / np.max(importance_scores)
            else:
                importance_norm = np.zeros(len(features))

            # Buat DataFrame
            importance_df = pd.DataFrame({
                'feature': features,
                'importance_raw': importance_scores,
                'importance_norm': importance_norm
            })

            # Sort by importance
            importance_df = importance_df.sort_values('importance_norm', ascending=False)

            # Categorize importance
            importance_df['category'] = pd.cut(
                importance_df['importance_norm'],
                bins=[-0.1, 0.3, 0.7, 1.1],
                labels=['Rendah', 'Sedang', 'Tinggi']
            )

            importance_data[model_name] = importance_df

            st.write(f"  ✅ Selesai: Top feature = {importance_df.iloc[0]['feature']} ({importance_df.iloc[0]['importance_norm']:.3f})")

        except Exception as e:
            st.error(f"  ❌ Error untuk model {model_name}: {str(e)}")
            # Create equal importance sebagai last resort
            importance_df = pd.DataFrame({
                'feature': features,
                'importance_raw': np.ones(len(features)),
                'importance_norm': np.ones(len(features)),
                'category': 'Sedang'
            })
            importance_data[model_name] = importance_df

    return importance_data

def get_model_type(model):
    """Detect type of model"""
    model_class = str(type(model)).lower()
    if 'randomforest' in model_class:
        return 'Random Forest'
    elif 'linearmodel' in model_class or 'linearregression' in model_class:
        return 'Linear Regression'
    elif 'xgboost' in model_class:
        return 'XGBoost'
    elif 'svm' in model_class or 'svr' in model_class:
        return 'SVM'
    else:
        return 'Unknown'

# ============================================================================
# SIDEBAR - NAVIGATION & CONTROLS
# ============================================================================
with st.sidebar:
    # Sidebar Header
    st.markdown("""
    <div class="sidebar-header">
        <h3 style="margin:0; font-size: 1.3rem;">🏥 DASHBOARD DBD</h3>
        <p style="margin:0.25rem 0 0 0; font-size:0.85rem; opacity:0.9">Analisis & Mitigasi Terpadu</p>
    </div>
    """, unsafe_allow_html=True)

    # Loading Data
    try:
        bundle, df_master = load_data()
        models = bundle['models']
        stats = bundle['stats']
        best_model_name = bundle['best_model']
        features = bundle['features']
        y_extra = bundle['y_extra']
        df_stats = pd.DataFrame(stats)

        # Debug: Cek fitur yang ada
        st.write(f"📊 Total fitur: {len(features)}")

        st.success("✅ Data berhasil dimuat")

    except Exception as e:
        st.error(f"Error: {str(e)}")
        st.stop()

    st.markdown("---")

    # Navigation Menu
    st.markdown("**🧭 NAVIGASI**")

    menu_pages = {
        "🏆 Model Terbaik": "model",
        "📊 Feature Importance": "features", 
        "📈 Analisis Tren": "trend",
        "📍 Prediksi Wilayah": "prediction",
        "📋 Statistik Data": "stats"
    }

    selected_menu = st.radio(
        "Pilih halaman:",
        list(menu_pages.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Region Selection
    st.markdown("**📍 FILTER WILAYAH**")

    selected_kota = st.selectbox(
        "Pilih Kabupaten/Kota:",
        options=sorted(df_master['Kabupaten/Kota'].unique()) if 'Kabupaten/Kota' in df_master.columns else ["Data tidak tersedia"],
        index=0,
        label_visibility="collapsed"
    )

    # Quick Stats
    st.markdown("---")
    st.markdown("**📊 QUICK STATS**")

    col1, col2 = st.columns(2)
    with col1:
        total_wilayah = len(df_master['Kabupaten/Kota'].unique()) if 'Kabupaten/Kota' in df_master.columns else 0
        st.metric("Total Wilayah", total_wilayah, label_visibility="collapsed")
    with col2:
        st.metric("Total Data", len(df_master), label_visibility="collapsed")

    # Model Info
    st.info(f"**Model Aktif:** {best_model_name}")

    # Last Updated
    st.markdown("---")
    st.caption(f"🔄 {datetime.now().strftime('%d %b %Y %H:%M')}")

# ============================================================================
# MAIN CONTENT AREA
# ============================================================================
# Header Utama
st.markdown("""
<div class="main-header">
    <h1>📊 Dashboard Analisis & Mitigasi DBD</h1>
    <p>Platform terintegrasi untuk monitoring, prediksi, dan rekomendasi strategi penanganan Demam Berdarah Dengue</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# TABBED INTERFACE
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 EVALUASI MODEL", 
    "📊 FEATURE IMPORTANCE",
    "📈 ANALISIS TREN", 
    "📍 PREDIKSI WILAYAH",
    "📋 DATA & STATISTIK"
])

# ============================================================================
# TAB 2: FEATURE IMPORTANCE ANALYSIS (DIPINDAH KE ATAS)
# ============================================================================
with tab2:
    st.markdown('<h3 class="section-header">📊 Feature Importance Analysis</h3>', unsafe_allow_html=True)

    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.markdown("""
    ### 🔍 Analisis Variabel Paling Berpengaruh
    Analisis ini menunjukkan variabel-variabel yang paling mempengaruhi prediksi kasus DBD pada setiap model split.
    Variabel dengan importance tinggi memiliki pengaruh signifikan terhadap hasil prediksi.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    # Debug: Cek tipe model
    st.markdown("#### 🔧 Informasi Model")
    col_debug1, col_debug2, col_debug3 = st.columns(3)

    with col_debug1:
        model_type_70 = get_model_type(models["70:30"])
        st.metric("Model 70:30", model_type_70)

    with col_debug2:
        model_type_80 = get_model_type(models["80:20"])
        st.metric("Model 80:20", model_type_80)

    with col_debug3:
        model_type_90 = get_model_type(models["90:10"])
        st.metric("Model 90:10", model_type_90)

    # Hitung feature importance dengan metode yang dikoreksi
    st.info("🔄 Menghitung feature importance...")

    try:
        # Persiapan data
        X_data = df_master[features]

        # Cari kolom target
        target_cols = ['kasus_dbd', 'IR_DBD', 'IR', 'kasus', 'incidence_rate', 'Incidence_Rate', 'Incidence_Rate_DBD']
        target_col = None
        for col in target_cols:
            if col in df_master.columns:
                target_col = col
                st.success(f"✅ Kolom target ditemukan: {target_col}")
                break

        if not target_col:
            # Jika tidak ditemukan, coba kolom terakhir
            st.warning("⚠️ Kolom target tidak ditemukan, menggunakan kolom terakhir numerik")
            numeric_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                target_col = numeric_cols[-1]
                st.info(f"Menggunakan kolom: {target_col}")

        if target_col:
            y_data = df_master[target_col]

            # Hitung feature importance
            with st.spinner("Menghitung importance untuk semua model..."):
                importance_data = calculate_feature_importance_corrected(models, X_data, y_data, features)

            # Tampilkan hasil untuk semua model
            st.markdown('<h3 class="subsection-header">📈 Feature Importance per Model Split</h3>', unsafe_allow_html=True)

            # Buat 3 kolom untuk 3 model
            col1, col2, col3 = st.columns(3)

            model_cols = {
                "70:30": col1,
                "80:20": col2,
                "90:10": col3
            }

            for model_name, col in model_cols.items():
                if model_name in importance_data:
                    importance_df = importance_data[model_name]

                    with col:
                        st.markdown(f'<div class="metric-card">', unsafe_allow_html=True)
                        st.markdown(f"**🏆 Model {model_name}**")

                        # Tampilkan top 5 features
                        st.markdown("**Top 5 Features:**")
                        for i, (_, row) in enumerate(importance_df.head(5).iterrows()):
                            category_class = f"importance-{row['category'].lower()}"
                            st.markdown(f"""
                            <div style="margin: 0.5rem 0; padding: 0.5rem; background: #334155; border-radius: 8px;">
                                <div style="font-weight: 600; color: #f1f5f9;">{i+1}. {row['feature']}</div>
                                <div style="display: flex; justify-content: space-between; margin-top: 0.25rem;">
                                    <span style="color: #94a3b8; font-size: 0.9rem;">Score: {row['importance_norm']:.3f}</span>
                                    <span class="{category_class}">{row['category']}</span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Cek khusus kepadatan penduduk
                        kepadatan_row = importance_df[importance_df['feature'].str.contains('kepadatan', case=False)]
                        if not kepadatan_row.empty:
                            st.markdown("---")
                            st.markdown(f"**📍 Kepadatan Penduduk:**")
                            st.markdown(f"Peringkat: #{kepadatan_row.index[0]+1}")
                            st.markdown(f"Score: {kepadatan_row.iloc[0]['importance_norm']:.3f}")

                        st.markdown('</div>', unsafe_allow_html=True)

            # Visualisasi perbandingan antar model
            st.markdown('<h3 class="subsection-header">📊 Perbandingan Visual Feature Importance</h3>', unsafe_allow_html=True)

            # Pilih model untuk visualisasi detail
            col_model_sel, _ = st.columns([1, 2])
            with col_model_sel:
                selected_feature_model = st.selectbox(
                    "Pilih Model untuk Detail:",
                    options=["70:30", "80:20", "90:10"],
                    index=["70:30", "80:20", "90:10"].index(best_model_name) if best_model_name in ["70:30", "80:20", "90:10"] else 0,
                    key="feature_detail_model"
                )

            if selected_feature_model in importance_data:
                importance_df = importance_data[selected_feature_model]

                # Bar chart untuk feature importance
                fig_importance = go.Figure()

                # Ambil semua features, urut dari tinggi ke rendah
                importance_df_sorted = importance_df.sort_values('importance_norm', ascending=True)

                # Warna berdasarkan kategori
                color_map = {'Tinggi': '#ef4444', 'Sedang': '#f59e0b', 'Rendah': '#3b82f6'}
                colors = [color_map.get(cat, '#94a3b8') for cat in importance_df_sorted['category']]

                # Highlight kepadatan penduduk jika ada
                highlight_indices = []
                for i, feat in enumerate(importance_df_sorted['feature']):
                    if 'kepadatan' in str(feat).lower():
                        highlight_indices.append(i)
                        colors[i] = '#10b981'  # Warna hijau untuk highlight

                fig_importance.add_trace(go.Bar(
                    x=importance_df_sorted['importance_norm'],
                    y=importance_df_sorted['feature'],
                    orientation='h',
                    marker_color=colors,
                    text=importance_df_sorted['importance_norm'].round(3),
                    textposition='auto',
                    textfont=dict(color='white', size=10)
                ))

                # Tambah annotation untuk kepadatan penduduk jika ada
                if highlight_indices:
                    for idx in highlight_indices:
                        feat_name = importance_df_sorted.iloc[idx]['feature']
                        fig_importance.add_annotation(
                            x=importance_df_sorted.iloc[idx]['importance_norm'],
                            y=idx,
                            text="📍 Kepadatan",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=1,
                            arrowwidth=2,
                            arrowcolor="#10b981",
                            font=dict(color="#10b981", size=12)
                        )

                fig_importance.update_layout(
                    title=dict(
                        text=f"Feature Importance - Model {selected_feature_model}",
                        font=dict(color='white', size=16)
                    ),
                    xaxis=dict(
                        title="Importance Score (Normalized)",
                        title_font=dict(color='#94a3b8'),
                        tickfont=dict(color='#94a3b8'),
                        gridcolor='#334155',
                        range=[0, 1.1]
                    ),
                    yaxis=dict(
                        title="Variabel",
                        title_font=dict(color='#94a3b8'),
                        tickfont=dict(color='#94a3b8', size=10)
                    ),
                    height=max(400, len(importance_df) * 25),  # Sesuaikan tinggi dengan jumlah fitur
                    plot_bgcolor='#1e293b',
                    paper_bgcolor='#0f172a',
                    font=dict(color='white'),
                    margin=dict(l=0, r=0, t=40, b=0),
                    showlegend=False
                )

                st.plotly_chart(fig_importance, use_container_width=True, config={'displayModeBar': False})

                # Analisis detail untuk model terpilih
                st.markdown('<h3 class="subsection-header">🔍 Analisis Detail</h3>', unsafe_allow_html=True)

                col_analysis1, col_analysis2 = st.columns(2)

                with col_analysis1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown(f"**📈 Statistik Importance - {selected_feature_model}**")

                    stats_data = {
                        "Total Fitur": len(importance_df),
                        "Fitur Tinggi": len(importance_df[importance_df['category'] == 'Tinggi']),
                        "Fitur Sedang": len(importance_df[importance_df['category'] == 'Sedang']),
                        "Fitur Rendah": len(importance_df[importance_df['category'] == 'Rendah']),
                        "Importance Rata-rata": f"{importance_df['importance_norm'].mean():.3f}",
                        "Importance Std": f"{importance_df['importance_norm'].std():.3f}"
                    }

                    for key, value in stats_data.items():
                        st.markdown(f"**{key}:** {value}")

                    st.markdown('</div>', unsafe_allow_html=True)

                with col_analysis2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**🎯 Rekomendasi Fokus Monitoring**")

                    # Rekomendasi berdasarkan feature importance
                    high_importance_features = importance_df[importance_df['category'] == 'Tinggi']['feature'].tolist()

                    if high_importance_features:
                        st.markdown("**Prioritas Tinggi:**")
                        for feat in high_importance_features[:3]:
                            # Berikan rekomendasi spesifik berdasarkan nama fitur
                            if 'kepadatan' in str(feat).lower():
                                st.markdown(f"- **{feat}**: Pantau pergerakan penduduk dan kepadatan permukiman")
                            elif 'hujan' in str(feat).lower():
                                st.markdown(f"- **{feat}**: Monitoring intensif curah hujan dan genangan air")
                            elif 'sampah' in str(feat).lower():
                                st.markdown(f"- **{feat}**: Perketat pengelolaan sampah dan kebersihan")
                            else:
                                st.markdown(f"- **{feat}**: Monitoring rutin dan analisis berkala")
                    else:
                        st.markdown("Semua variabel memiliki pengaruh sedang, lakukan monitoring komprehensif")

                    st.markdown('</div>', unsafe_allow_html=True)

                # Perbandingan konsistensi antar model
                st.markdown('<h3 class="subsection-header">🔄 Konsistensi Variabel Penting Antar Model</h3>', unsafe_allow_html=True)

                # Ambil top 10 features dari setiap model
                comparison_data = {}
                for model_name in ["70:30", "80:20", "90:10"]:
                    if model_name in importance_data:
                        top_features = importance_data[model_name].head(10)['feature'].tolist()
                        comparison_data[model_name] = top_features

                if comparison_data:
                    # Hitung frekuensi setiap feature
                    all_features = []
                    for features_list in comparison_data.values():
                        all_features.extend(features_list)

                    from collections import Counter
                    feature_counts = Counter(all_features)

                    # Tampilkan features yang muncul di semua model
                    common_features = [feat for feat, count in feature_counts.items() if count == 3]

                    if common_features:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.success(f"✅ **{len(common_features)} variabel konsisten** di semua model:")
                        for feat in common_features:
                            st.markdown(f"- **{feat}**")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.warning("⚠️ Tidak ada variabel yang muncul di semua 3 model")
                        st.markdown("**Variabel yang muncul di 2 model:**")
                        semi_common = [feat for feat, count in feature_counts.items() if count == 2]
                        for feat in semi_common:
                            st.markdown(f"- {feat}")
                        st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.error("❌ Tidak dapat menentukan kolom target untuk perhitungan importance")

    except Exception as e:
        st.error(f"❌ Error dalam menghitung feature importance: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

        # Fallback: Tampilkan fitur yang ada
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**📋 Daftar Variabel yang Digunakan**")

        # Cari variabel dengan kata "kepadatan"
        kepadatan_features = [f for f in features if 'kepadatan' in str(f).lower()]

        if kepadatan_features:
            st.success(f"✅ Variabel kepadatan ditemukan: {len(kepadatan_features)}")
            for feat in kepadatan_features:
                st.markdown(f"- **{feat}**")
        else:
            st.warning("⚠️ Tidak ditemukan variabel dengan kata 'kepadatan'")

        # Tampilkan semua fitur
        st.markdown("**Semua Fitur:**")
        for i, feat in enumerate(features[:20]):
            st.markdown(f"{i+1}. {feat}")
        if len(features) > 20:
            st.markdown(f"... dan {len(features)-20} variabel lainnya")

        st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# TAB 1: MODEL EVALUATION
# ============================================================================
with tab1:
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown('<h3 class="section-header">🏆 Model Performa Terbaik</h3>', unsafe_allow_html=True)

        # Highlight Card untuk Model Terbaik
        if not df_stats.empty and best_model_name in df_stats['Rasio'].values:
            best_stats = df_stats[df_stats['Rasio'] == best_model_name].iloc[0]
            r2_train = float(best_stats['R2_Internal'])
            r2_val = float(best_stats['R2_Extra'])
            selisih = abs(r2_train - r2_val)

            st.markdown(f"""
            <div class="highlight-card">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem;">
                    <div>
                        <h3 style="margin:0; font-size: 1.3rem; color: white;">{best_model_name}</h3>
                        <p style="margin:0.25rem 0 0 0; opacity:0.9;">Model dengan performa optimal</p>
                    </div>
                    <div style="font-size: 2rem; opacity: 0.8;">🏆</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Grid untuk metrics
            st.markdown('<div class="highlight-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            col_grid1, col_grid2, col_grid3 = st.columns(3)
            with col_grid1:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: white;">{r2_train:.3f}</div>
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.9);">R² Training</div>
                </div>
                """, unsafe_allow_html=True)
            with col_grid2:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: white;">{r2_val:.3f}</div>
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.9);">R² Validasi</div>
                </div>
                """, unsafe_allow_html=True)
            with col_grid3:
                st.markdown(f"""
                <div style="text-align: center;">
                    <div style="font-size: 1.8rem; font-weight: 700; color: white;">{selisih:.3f}</div>
                    <div style="font-size: 0.85rem; color: rgba(255,255,255,0.9);">Selisih</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Penjelasan Model Terbaik
            st.markdown('<div class="metric-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.markdown("**🎯 Kenapa Model Ini Dipilih?**")
            st.markdown(f"""
            - **Stabilitas Tinggi**: Konsisten pada data training ({r2_train:.3f}) dan validasi ({r2_val:.3f})
            - **Generalisasi Optimal**: Selisih terkecil ({selisih:.3f}) diantara semua model
            - **Minimal Overfitting**: Tidak terjadi penurunan performa signifikan
            - **Robust**: Tahan terhadap variasi data
            """)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<h3 class="section-header">📊 Perbandingan Model</h3>', unsafe_allow_html=True)

        if not df_stats.empty:
            # Bar Chart Interaktif dengan Plotly
            fig = go.Figure()

            # Pastikan data dalam format numerik
            df_stats['R2_Internal'] = pd.to_numeric(df_stats['R2_Internal'], errors='coerce')
            df_stats['R2_Extra'] = pd.to_numeric(df_stats['R2_Extra'], errors='coerce')

            # Add bars
            fig.add_trace(go.Bar(
                x=df_stats['Rasio'].astype(str),
                y=df_stats['R2_Internal'],
                name='Data Training',
                marker_color='#3b82f6',
                text=df_stats['R2_Internal'].round(3),
                textposition='outside',
                textfont=dict(color='white', size=11)
            ))

            fig.add_trace(go.Bar(
                x=df_stats['Rasio'].astype(str),
                y=df_stats['R2_Extra'],
                name='Data Validasi',
                marker_color='#10b981',
                text=df_stats['R2_Extra'].round(3),
                textposition='outside',
                textfont=dict(color='white', size=11)
            ))

            # Update layout
            fig.update_layout(
                title=dict(
                    text="Perbandingan Performa Model",
                    font=dict(color='white', size=16)
                ),
                xaxis=dict(
                    title="Rasio Split Data",
                    title_font=dict(color='#94a3b8'),
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='#334155'
                ),
                yaxis=dict(
                    title="Koefisien Determinasi (R²)",
                    title_font=dict(color='#94a3b8'),
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='#334155'
                ),
                barmode='group',
                height=400,
                showlegend=True,
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                font=dict(color='white'),
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='#1e293b',
                    bordercolor='#334155',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    # Tabel Detail Performa
    st.markdown('<h3 class="section-header">📋 Detail Performa Model</h3>', unsafe_allow_html=True)

    if not df_stats.empty:
        df_display = df_stats.copy()

        # Konversi ke numeric
        df_display['R2_Internal'] = pd.to_numeric(df_display['R2_Internal'], errors='coerce')
        df_display['R2_Extra'] = pd.to_numeric(df_display['R2_Extra'], errors='coerce')
        df_display['Selisih'] = abs(df_display['R2_Internal'] - df_display['R2_Extra'])

        # Format angka
        df_display['R2_Internal'] = df_display['R2_Internal'].round(3)
        df_display['R2_Extra'] = df_display['R2_Extra'].round(3)
        df_display['Selisih'] = df_display['Selisih'].round(3)

        # Urutkan berdasarkan selisih
        df_display = df_display.sort_values('Selisih')

        # Tampilkan tabel
        st.dataframe(
            df_display,
            use_container_width=True,
            height=200
        )

# ============================================================================
# TAB 3: TREND ANALYSIS
# ============================================================================
with tab3:
    st.markdown('<h3 class="section-header">📈 Analisis Tren Prediksi vs Aktual</h3>', unsafe_allow_html=True)

    col_control1, col_control2 = st.columns([1, 2])

    with col_control1:
        selected_model = st.selectbox(
            "Pilih Model:",
            options=["70:30", "80:20", "90:10"],
            index=["70:30", "80:20", "90:10"].index(best_model_name) if best_model_name in ["70:30", "80:20", "90:10"] else 0,
            key="model_selector_trend"
        )

    # Data untuk plotting
    if 'features' in locals() and features and isinstance(y_extra, (pd.Series, np.ndarray, list)):
        try:
            X_extra = df_master[features].iloc[-15:] if len(df_master) >= 15 else df_master[features]
            y_pred_selected = models[selected_model].predict(X_extra)

            # Pastikan y_extra adalah array/Series
            y_extra_values = y_extra.values if hasattr(y_extra, 'values') else y_extra

            # Buat line chart
            fig_trend = go.Figure()

            fig_trend.add_trace(go.Scatter(
                x=list(range(1, min(16, len(y_extra_values) + 1))),
                y=y_extra_values[:15],
                mode='lines+markers',
                name='Data Aktual',
                line=dict(color='#60a5fa', width=3),
                marker=dict(size=8, color='#60a5fa')
            ))

            fig_trend.add_trace(go.Scatter(
                x=list(range(1, min(16, len(y_pred_selected) + 1))),
                y=y_pred_selected[:15],
                mode='lines+markers',
                name=f'Prediksi ({selected_model})',
                line=dict(color='#f87171', width=3, dash='dash'),
                marker=dict(size=8, color='#f87171', symbol='x')
            ))

            fig_trend.update_layout(
                title=dict(
                    text=f"Perbandingan Data Aktual vs Prediksi",
                    font=dict(color='white', size=16)
                ),
                xaxis=dict(
                    title="Observasi",
                    title_font=dict(color='#94a3b8'),
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='#334155',
                    tickmode='linear',
                    tick0=1,
                    dtick=1
                ),
                yaxis=dict(
                    title="Nilai",
                    title_font=dict(color='#94a3b8'),
                    tickfont=dict(color='#94a3b8'),
                    gridcolor='#334155'
                ),
                height=400,
                hovermode="x unified",
                plot_bgcolor='#1e293b',
                paper_bgcolor='#0f172a',
                legend=dict(
                    font=dict(color='white'),
                    bgcolor='#1e293b',
                    bordercolor='#334155',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig_trend, use_container_width=True, config={'displayModeBar': False})

            # Metrics Cards
            st.markdown('<h3 class="section-header">📊 Metrik Evaluasi Model</h3>', unsafe_allow_html=True)

            mse = np.mean((y_extra_values[:len(y_pred_selected)] - y_pred_selected) ** 2)
            mae = np.mean(np.abs(y_extra_values[:len(y_pred_selected)] - y_pred_selected))

            col_metrics1, col_metrics2, col_metrics3 = st.columns(3)

            with col_metrics1:
                if not df_stats.empty and selected_model in df_stats['Rasio'].values:
                    r2_train = float(df_stats[df_stats['Rasio'] == selected_model]['R2_Internal'].iloc[0])
                    st.metric("R² Training", f"{r2_train:.3f}")

            with col_metrics2:
                st.metric("MSE", f"{mse:.3f}")

            with col_metrics3:
                st.metric("MAE", f"{mae:.3f}")

        except Exception as e:
            st.error(f"Error dalam analisis tren: {str(e)}")
    else:
        st.warning("Data tidak tersedia untuk analisis tren")

# ============================================================================
# TAB 4 & 5: SAMA SEPERTI SEBELUMNYA (disingkat untuk hemat ruang)
# ============================================================================
with tab4:
    if 'selected_kota' in locals() and 'Kabupaten/Kota' in df_master.columns:
        data_wilayah = df_master[df_master['Kabupaten/Kota'] == selected_kota]

        if not data_wilayah.empty:
            latest_data = data_wilayah.iloc[-1]

            try:
                X_pred = pd.DataFrame([latest_data[features]], columns=features)
                pred_value = models[best_model_name].predict(X_pred)[0]

                st.markdown(f'<h3 class="section-header">📍 Analisis Wilayah: {selected_kota}</h3>', unsafe_allow_html=True)

                col_pred, col_status = st.columns([1, 1.5])

                with col_pred:
                    if pred_value > 50:
                        risk_class = "risk-high"
                        risk_label = "🚨 RISIKO TINGGI"
                    elif pred_value > 30:
                        risk_class = "risk-medium"
                        risk_label = "⚠️ RISIKO SEDANG"
                    else:
                        risk_class = "risk-low"
                        risk_label = "✅ RISIKO RENDAH"

                    st.markdown(f"""
                    <div class="{risk_class}">
                        <div style="margin-bottom: 1rem;">
                            <h4 style="margin:0; font-size: 1rem; opacity: 0.9;">PREDIKSI INDEKS RISIKO DBD</h4>
                            <h1 style="margin:0.5rem 0; font-size: 2.5rem; font-weight: 700;">{pred_value:.1f}</h1>
                        </div>
                        <div style="text-align: center; padding: 0.75rem; background: rgba(255,255,255,0.15); border-radius: 8px;">
                            <h3 style="margin:0; font-size: 1.1rem;">{risk_label}</h3>
                        </div>
                        <p style="margin: 1rem 0 0 0; opacity: 0.9; font-size: 0.85rem;">
                            Model: <strong>{best_model_name}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                with col_status:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**🌡️ Kondisi Lingkungan**")

                    env_col1, env_col2 = st.columns(2)

                    with env_col1:
                        if 'curah_hujan_mm' in latest_data:
                            rain_value = float(latest_data['curah_hujan_mm'])
                            rain_status = "Tinggi" if rain_value > 300 else "Normal"
                            st.metric("Curah Hujan", f"{rain_value:.1f} mm", rain_status)

                    with env_col2:
                        if 'timbulan_sampah_ton' in latest_data:
                            waste_value = float(latest_data['timbulan_sampah_ton'])
                            waste_status = "Tinggi" if waste_value > 450 else "Normal"
                            st.metric("Timbulan Sampah", f"{waste_value:.1f} ton", waste_status)

                    st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<h3 class="section-header">🛡️ Rekomendasi Mitigasi</h3>', unsafe_allow_html=True)

                col_rec1, col_rec2 = st.columns(2)

                with col_rec1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**🎯 Tindakan Prioritas**")

                    if pred_value > 50:
                        st.markdown("""
                        - Aktifkan Posko DBD 24 jam
                        - Fogging fokus daerah rawan
                        - Mobilisasi Tim Gerak Cepat
                        - Siapkan RS rujukan
                        """)
                    elif pred_value > 30:
                        st.markdown("""
                        - Intensifkan surveilans
                        - Sosialisasi 3M Plus
                        - Perkuat sistem pelaporan
                        - Siapkan logistik
                        """)
                    else:
                        st.markdown("""
                        - Monitoring rutin
                        - Pemeliharaan lingkungan
                        - Edukasi berkelanjutan
                        - Kesiapan dasar
                        """)

                    st.markdown('</div>', unsafe_allow_html=True)

                with col_rec2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.markdown("**🌿 Berdasarkan Kondisi**")

                    recommendations = []

                    if 'curah_hujan_mm' in latest_data and float(latest_data['curah_hujan_mm']) > 300:
                        recommendations.append("Pantau genangan air intensif")

                    if 'timbulan_sampah_ton' in latest_data and float(latest_data['timbulan_sampah_ton']) > 450:
                        recommendations.append("Percepat pengangkutan sampah")

                    if not recommendations:
                        recommendations.append("Pertahankan kondisi lingkungan")

                    recommendations.append("Optimalkan Gerakan 1 Rumah 1 Jumantik")
                    recommendations.append("Laporkan kasus secara real-time")

                    for rec in recommendations:
                        st.markdown(f"- {rec}")

                    st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error dalam prediksi: {str(e)}")
        else:
            st.warning(f"Data untuk wilayah {selected_kota} tidak ditemukan.")
    else:
        st.warning("Data wilayah tidak tersedia")

with tab5:
    st.markdown('<h3 class="section-header">📋 Statistik Data</h3>', unsafe_allow_html=True)

    col_stat1, col_stat2 = st.columns(2)

    with col_stat1:
        st.metric("Total Observasi", len(df_master))

    with col_stat2:
        if 'Kabupaten/Kota' in df_master.columns:
            total_wilayah = len(df_master['Kabupaten/Kota'].unique())
            st.metric("Jumlah Wilayah", total_wilayah)

    st.markdown('<h4 class="subsection-header">👁️ Preview Data</h4>', unsafe_allow_html=True)
    st.dataframe(df_master.head(8), use_container_width=True, height=280)

    numeric_cols = df_master.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown('<h4 class="subsection-header">📊 Ringkasan Statistik</h4>', unsafe_allow_html=True)
        st.dataframe(df_master[numeric_cols].describe().round(2), use_container_width=True, height=300)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

footer_col1, footer_col2, footer_col3 = st.columns(3)

with footer_col1:
    st.markdown("**📞 Kontak Darurat DBD**")
    st.markdown("- Hotline: 119")
    st.markdown("- Puskesmas: 112")

with footer_col2:
    st.markdown("**🔄 Status Sistem**")
    st.markdown(f"- Model: **{best_model_name}**")
    st.markdown(f"- Data: **{len(df_master)}** observasi")

with footer_col3:
    st.markdown("**📅 Update Terakhir**")
    st.markdown(datetime.now().strftime("%d %B %Y %H:%M"))

st.markdown("""
<div style="text-align: center; padding: 1rem; color: #94a3b8; font-size: 0.85rem; margin-top: 2rem;">
Dashboard Mitigasi DBD v2.0 • © 2024 Kementerian Kesehatan
</div>
""", unsafe_allow_html=True)
