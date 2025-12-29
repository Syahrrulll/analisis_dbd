# **Analisis Variabel Ekologis Morbiditas DBD di Jawa Tengah**

**Strategi Mitigasi Menggunakan Random Forest Regressor**

---

## **📌 Ringkasan Proyek**
Sistem analisis data kesehatan dan lingkungan yang dirancang untuk mengidentifikasi faktor ekologis paling signifikan terhadap penyebaran **Demam Berdarah Dengue (DBD)** di Jawa Tengah. Dengan menggunakan algoritma **Random Forest Regressor**, sistem ini mampu memberikan rekomendasi berbasis data untuk pengambilan keputusan mitigasi DBD yang lebih efektif dan terarah.

### **👥 Tim Pengembang:**
- **Darmayanti** (F1G123004)
- **Muhammad Syahrul Mubarak** (F1G123030)

---

## **🎯 Tujuan Utama**
1. **Mengidentifikasi** variabel ekologis dominan yang memengaruhi morbiditas DBD.
2. **Membangun model prediktif** untuk memperkirakan tingkat penyebaran DBD berdasarkan kondisi lingkungan.
3. **Menyediakan dashboard interaktif** yang memudahkan analisis dan visualisasi hasil prediksi.

---

## **🔧 Teknologi yang Digunakan**
| Kategori              | Teknologi                          |
|------------------------|------------------------------------|
| Bahasa Pemrograman     | Python                             |
| Analisis & Manipulasi Data | Pandas, NumPy                   |
| Visualisasi            | Matplotlib, Seaborn               |
| Machine Learning       | Scikit-Learn (Random Forest)       |
| Deployment & Dashboard | Streamlit, Joblib                  |
| Format Penyimpanan Model | `.pkl` (Pickle)                  |

---

## **📂 Struktur Direktori Proyek**
```
.
│── data_dbd.csv                # Data kasus DBD
├── curah_hujan_fix.csv         # Data iklim
├── pengelolaan_sampah_fix.csv  # Data lingkungan
├── persentase_penduduk.csv     # Data demografi
├── sanitasi.csv                # Data akses sanitasi
├── DBD.ipynb                       # Notebook 

eksperimen & pelatihan model
├── app.py                          # Aplikasi dashboard Streamlit
├── requirements.txt                # Daftar library python
└── model_final_bundle.pkl          # Model & metadata hasil export
```

---

## **🚀 Cara Menjalankan Proyek**

### **1. Prasyarat**
- **Python 3.8+** (disarankan versi terbaru)
- **Pip** (Python Package Installer)
- **Git** (opsional, untuk cloning repository)

### **2. Instalasi Dependensi**
```bash
pip install -r requirements.txt
```

### **3. Menjalankan Analisis & Pelatihan Model**
1. Buka file `DBD.ipynb` di **Jupyter Notebook** atau **Google Colab**.
2. Jalankan seluruh sel untuk:
   - **Menggabungkan dataset**
   - **Membersihkan dan memproses data**
   - **Melatih model Random Forest**
   - **Mengekspor model** ke `model_final_bundle.pkl`

### **4. Menjalankan Dashboard Interaktif**
```bash
streamlit run app.py
```
Dashboard akan terbuka di browser default Anda (biasanya di `http://localhost:8501`).

---

## **📊 Dataset yang Digunakan**
| Nama File                     | Deskripsi                                     |
|-------------------------------|-----------------------------------------------|
| `data_dbd.csv`                | Data kasus DBD per kabupaten/kota (2019-2022) |
| `curah_hujan_fix.csv`         | Data curah hujan tahunan (mm)                 |
| `pengelolaan_sampah_fix.csv`  | Data timbulan sampah (ton)                    |
| `persentase_penduduk.csv`     | Data jumlah dan kepadatan penduduk            |
| `sanitasi.csv`                | Data persentase akses sanitasi layak          |

---

## **🔍 Metode Analisis**

### **A. Feature Engineering**
- **Incidence Rate (IR)** dihitung sebagai:
  \[
  IR = \frac{\text{Kasus DBD}}{\text{Penduduk}} \times 100.000
  \]
  *(Standar morbiditas per 100.000 penduduk)*

### **B. Pemodelan Machine Learning**
- **Algoritma:** `RandomForestRegressor` (Scikit-Learn)
- **Rasio Data:**
  - **70:30** (Training:Testing) → **Paling stabil** berdasarkan hasil evaluasi
  - 80:20
  - 90:10
- **Metrik Evaluasi:**
  - **R² Score** (Koefisien Determinasi)
  - **MAE** (Mean Absolute Error)
  - **RMSE** (Root Mean Squared Error)

### **C. Feature Importance**
Model mengurutkan variabel berdasarkan pengaruhnya terhadap prediksi IR DBD:
1. **Curah Hujan**
2. **Kepadatan Penduduk**
3. **Akses Sanitasi**
4. **Timbulan Sampah**

---

## **🎨 Fitur Dashboard (Streamlit)**
Dashboard interaktif ini memungkinkan:
- **📈 Visualisasi Tren DBD** per tahun dan wilayah.
- **🔮 Prediksi Real-Time** dengan input parameter lingkungan.
- **📊 Analisis Feature Importance** untuk identifikasi faktor dominan.
- **🗺️ Pemetaan Wilayah Risiko** berdasarkan prediksi model.

---

## **📈 Hasil & Insight Utama**
- **Curah hujan** merupakan faktor paling signifikan dalam penyebaran DBD.
- Model dengan rasio **70:30** menunjukkan performa paling stabil dengan **R² = 0,14** pada data eksternal.
- **Kepadatan penduduk** dan **akses sanitasi** juga berkontribusi penting, meskipun dengan pengaruh lebih rendah.

---

## **📝 Rekomendasi Mitigasi**
Berdasarkan analisis, strategi mitigasi prioritas meliputi:
1. **Pengelolaan drainase** di wilayah curah hujan tinggi.
2. **Program sanitasi terpadu** di daerah padat penduduk.
3. **Pengawasan pengelolaan sampah** untuk mengurangi potensi perkembangbiakan nyamuk.

---

## **🔮 Pengembangan Selanjutnya**
- Integrasi data cuaca **real-time** untuk prediksi lebih akurat.
- Penambahan variabel **sosio-ekonomi** dan **perilaku masyarakat**.
- Deployment model sebagai **API** untuk integrasi dengan sistem Dinas Kesehatan.

---



> **⚠️ Catatan Penting:** Dashboard ini dirancang sebagai alat bantu pengambilan keputusan dan tidak menggantikan konsultasi dengan ahli epidemiologi atau kebijakan resmi pemerintah.



---
---
---

# **Dashboard Mitigasi DBD - Streamlit Web Application**

## **🎯 Cara Kerja Aplikasi Web**

Aplikasi web Streamlit ini menyediakan **dashboard interaktif** untuk analisis, prediksi, dan monitoring Demam Berdarah Dengue (DBD) di Jawa Tengah.

---

## **🚀 Cara Menjalankan Dashboard**

### **Prasyarat:**
1. Pastikan semua library sudah terinstall (`requirements.txt`)
2. File `model_final_bundle.pkl` dan `df_gabungan.csv` tersedia
3. Python 3.8+ terinstall

### **Langkah Menjalankan:**
```bash
streamlit run app.py
```
Dashboard akan terbuka di browser default di alamat `http://localhost:8501`

---

## **🔧 Arsitektur Dashboard**

### **A. Struktur Halaman (Tab-based Interface)**
Dashboard menggunakan sistem tab untuk organisasi konten:

| Tab | Deskripsi | Fitur Utama |
|-----|-----------|-------------|
| **🏆 EVALUASI MODEL** | Analisis performa model ML | Perbandingan split ratio, visualisasi R² score |
| **📊 FEATURE IMPORTANCE** | Analisis variabel berpengaruh | Feature importance, kategori kepentingan |
| **📈 ANALISIS TREN** | Visualisasi prediksi vs aktual | Line chart, metrik evaluasi |
| **📍 PREDIKSI WILAYAH** | Prediksi risiko per wilayah | Estimasi risiko, rekomendasi mitigasi |
| **📋 DATA & STATISTIK** | Eksplorasi dataset | Preview data, statistik deskriptif |

---

## **🖥️ Komponen Utama Dashboard**

### **1. SIDEBAR (Navigasi & Kontrol)**
- **Menu Navigasi**: 5 tab utama untuk berpindah halaman
- **Filter Wilayah**: Dropdown untuk memilih kabupaten/kota
- **Quick Stats**: Statistik cepat (total wilayah, total data)
- **Info Model**: Menampilkan model aktif dan waktu update

### **2. MAIN CONTENT (Konten Dinamis)**
Konten berubah berdasarkan tab yang dipilih:

#### **📊 Tab Feature Importance**
**Fungsi:** Menganalisis variabel paling berpengaruh terhadap prediksi DBD

**Cara Kerja:**
1. **Loading Data**: Membaca model dan dataset
2. **Feature Importance Calculation**:
   - Menggunakan `feature_importances_` (Random Forest)
   - Atau `coef_` (Linear Regression)
   - Atau `permutation_importance` (fallback method)
3. **Kategorisasi**:
   - **Tinggi** (>0.7): Pengaruh sangat signifikan
   - **Sedang** (0.3-0.7): Pengaruh moderat
   - **Rendah** (<0.3): Pengaruh minimal
4. **Visualisasi**:
   - Bar chart horizontal berwarna
   - Highlight variabel kunci (kepadatan penduduk)
   - Perbandingan antar model

**Output:**
- Ranking variabel berdasarkan pengaruh
- Rekomendasi monitoring prioritas
- Analisis konsistensi antar model

#### **🏆 Tab Evaluasi Model**
**Fungsi:** Membandingkan performa model dengan rasio split berbeda

**Cara Kerja:**
1. **Model Loading**: Memuat 3 model (70:30, 80:20, 90:10)
2. **Metrik Evaluasi**:
   - R² Training (data internal)
   - R² Validasi (data eksternal)
   - Selisih (indikator overfitting)
3. **Penentuan Model Terbaik**: Berdasarkan selisih terkecil
4. **Visualisasi**: Bar chart perbandingan antar model

#### **📈 Tab Analisis Tren**
**Fungsi:** Visualisasi prediksi vs data aktual

**Cara Kerja:**
1. **Data Preparation**: Mengambil data validasi (15 observasi terakhir)
2. **Prediksi**: Menggunakan model yang dipilih
3. **Visualisasi**: Line chart dengan dua garis (aktual vs prediksi)
4. **Metrik**: MSE dan MAE untuk evaluasi akurasi

#### **📍 Tab Prediksi Wilayah**
**Fungsi:** Prediksi risiko DBD per kabupaten/kota

**Cara Kerja:**
1. **Input**: Wilayah dipilih dari sidebar
2. **Preprocessing**: Mengambil data terbaru wilayah tersebut
3. **Prediksi**: Menggunakan model terbaik
4. **Klasifikasi Risiko**:
   - **Rendah** (<30): ✅ Warna hijau
   - **Sedang** (30-50): ⚠️ Warna kuning
   - **Tinggi** (>50): 🚨 Warna merah
5. **Rekomendasi**: Tindakan mitigasi spesifik

#### **📋 Tab Data & Statistik**
**Fungsi:** Eksplorasi dataset

**Cara Kerja:**
1. **Data Preview**: Tampilkan 8 baris pertama
2. **Statistik Deskriptif**: Mean, std, min, max
3. **Quick Metrics**: Total observasi, jumlah wilayah

---

## **🎨 Desain & UX**

### **Dark Theme Premium**
- **Background**: `#0f172a` (dark navy blue)
- **Cards**: `#1e293b` dengan border `#334155`
- **Text**: `#f1f5f9` (light gray)
- **Accents**: Gradient biru-hijau untuk header

### **Komponen Visual**
1. **Metric Cards**: Kotak informasi dengan shadow effect
2. **Highlight Cards**: Untuk informasi penting dengan gradient
3. **Importance Tags**: Badge warna untuk kategori kepentingan
4. **Interactive Charts**: Plotly dengan tema custom

### **Responsive Design**
- Layout wide mode untuk layar besar
- Columns system untuk penataan konten
- Container width adjustment

---

## **🔗 Integrasi Data & Model**

### **Data Flow:**
```
Data Sources (CSV) → Data Wrangling → Feature Engineering → Model Training → Model Export → Streamlit Dashboard
```

### **Model Bundle:**
File `model_final_bundle.pkl` berisi:
- 3 model Random Forest (70:30, 80:20, 90:10)
- Statistik performa
- Metadata (features, best model info)
- Data validasi

---

## **⚙️ Teknik Pemrograman Khusus**

### **Caching System**
```python
@st.cache_resource
def load_data():
    """Cache model dan data untuk performance"""
```

### **Error Handling**
- Try-except untuk semua operasi kritis
- Fallback mechanism untuk feature importance
- User-friendly error messages

### **Dynamic Styling**
- CSS custom untuk tema gelap
- Conditional formatting berdasarkan nilai
- Animasi dan transisi halus

---

## **📱 Fitur Interaktif**

### **1. Real-time Prediction**
- Input: Pilih wilayah dari dropdown
- Output: Prediksi risiko + rekomendasi
- Visual: Warna berubah berdasarkan tingkat risiko

### **2. Model Comparison**
- Switch antara 3 model
- Update chart secara real-time
- Detail metrik untuk setiap model

### **3. Feature Analysis**
- Filter by model split
- Highlight variabel penting
- Exportable insights

### **4. Data Exploration**
- Interactive data table
- Statistical summaries
- Quick filters

---

## **🔧 Troubleshooting**

### **Common Issues:**

| Issue | Solution |
|-------|----------|
| **Model tidak loading** | Pastikan `model_final_bundle.pkl` ada di direktori yang sama |
| **Data tidak muncul** | Cek file `df_gabungan.csv` dan format kolom |
| **Chart error** | Update Plotly ke versi terbaru |
| **Slow performance** | Gunakan `@st.cache_resource` untuk data besar |

### **Debug Mode:**
```python
# Tambahkan di app.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## **🔮 Fitur Future Development**

### **Rencana Pengembangan:**
1. **Real-time Data Integration** dengan API Dinas Kesehatan
2. **Geospatial Mapping** dengan peta interaktif
3. **Alert System** untuk threshold tertentu
4. **Multi-user Authentication** untuk akses terbatas
5. **Export Functionality** untuk laporan PDF/Excel

---

## **📞 Support & Documentation**

### **Untuk Pengembang:**
- **Code Structure**: Modular dengan fungsi terpisah
- **Documentation**: Docstring untuk semua fungsi utama
- **Version Control**: Git dengan commit messages yang jelas

### **Untuk Pengguna:**
- **User Guide**: Tooltips dan petunjuk di UI
- **Error Messages**: Bahasa Indonesia yang jelas
- **Contact Info**: Footer dengan informasi kontak

---

## **🎯 Kesimpulan**

Dashboard ini menyediakan **solusi end-to-end** untuk:
1. **Monitoring** tren DBD secara real-time
2. **Prediksi** risiko berdasarkan kondisi lingkungan
3. **Analisis** faktor dominan penyebaran DBD
4. **Rekomendasi** tindakan mitigasi berbasis data

Dengan antarmuka yang intuitif dan analisis yang mendalam, dashboard ini menjadi alat bantu keputusan yang efektif untuk petugas kesehatan dan pemerintah daerah dalam penanganan DBD.