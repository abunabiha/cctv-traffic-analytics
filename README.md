# Smart CCTV Traffic Analytics & Anomaly Detection - Tri-Lokasi Bandar Lampung

Sistem analitik video cerdas (*Intelligent Video Analytics*) berbasis Deep Learning (YOLOv8 + ByteTrack), Transformasi Perspektif Homografi, dan Mesin Aturan Anomali Temporal untuk pemantauan arus lalu lintas, klasifikasi kendaraan Standar Indonesia (Golongan I s.d. VI), penegakan hukum helm ETLE (Golongan VI-A vs VI-B), dan deteksi kecelakaan secara otomatis pada kamera pengawas Kota Bandar Lampung dari portal [Seribu Wajah Kota Bandar Lampung](https://seribuwajah.bandarlampungkota.go.id/list).

### 📍 Lokasi Pengamatan (Tri-Lokasi):
1. **CCTV Perempatan Jendral Sudirman (ID: 282) [LOKASI 1 / DEFAULT]**  
   Simpang empat bersinyal di pusat kota Jl. Jendral Sudirman dengan manuver 4 lajur, antrean *traffic light*, dan kepadatan tinggi sepeda motor dan kendaraan pribadi (*interrupted flow*).
2. **CCTV Flyover Mall Boemi Kedaton (ID: 190) [LOKASI 2]**  
   Jembatan layang (flyover) 2 lajur sepanjang 262 meter di atas persimpangan KA dan Jl. Teuku Umar - Jl. Z.A. Pagar Alam dengan arus menerus bebas hambatan (*elevated continuous flow*).
3. **CCTV Unila Arah Rajabasa (ID: 312) [LOKASI 3]**  
   Jalan arteri dua arah terowongan bawah tanah bebas hambatan dengan pemisah fisik median barrier (*depressed continuous flow*).

---

## 🌟 Fitur Unggulan Sistem

1. **Pengambilan Aliran Live Stream Multi-Kamera (HLS & Snapshot)**:
   - Handshake otomatis sesi token HTTP Cookie (`cookieCheck=1` $\to$ `hlsSession`).
   - Mendukung streaming dinamis HLS m3u8 untuk CCTV 282 (Sudirman), CCTV 190 (Flyover MBK), dan CCTV 312 (Unila) tanpa buffering di Google Drive (menggunakan SSD lokal `/tmp`).
   - Mekanisme *instant fallback* ke snapshot kalibrasi berlatensi rendah jika koneksi CCTV kota mengalami *jitter*.

2. **Deteksi & Klasifikasi Kendaraan Standar Indonesia (Golongan I s.d. VI)**:
   - **Golongan I**: Mobil pribadi, sedan, jip, pick-up, bus, dan truk kecil ($L < 5.0\text{ m}$).
   - **Golongan II**: Truk besar 2 gandar ($5.0\text{ m} \le L < 8.0\text{ m}$).
   - **Golongan III**: Truk besar 3 gandar / Tronton ($8.0\text{ m} \le L < 11.5\text{ m}$).
   - **Golongan IV**: Truk besar 4 gandar ($11.5\text{ m} \le L < 14.0\text{ m}$).
   - **Golongan V**: Truk trailer 5+ gandar ($L \ge 14.0\text{ m}$).
   - **Golongan VI-A (Taat Helm)**: Pengendara sepeda motor menggunakan helm SNI.
   - **Golongan VI-B (Melanggar ETLE)**: Pengendara sepeda motor tanpa pelindung kepala helm.
   - **Road Region of Interest (ROI) Adaptif**: Poligon penyaring area jalan (`SUDIRMAN_ROAD_ROI`, `MBK_ROAD_ROI`, dan `DEFAULT_ROAD_ROI`) untuk mengabaikan trotoar ruko, pejalan kaki, dan atap gedung sekitar.

3. **Estimasi Kecepatan Nyata Kendaraan (km/jam)**:
   - Menggunakan **Transformasi Homografi Planar ($\mathbf{H}$)** yang memetakan piksel kamera $(u, v)$ ke bidang fisik meter $(X, Y)$:
     * **Simpang Sudirman**: Bidang jalan $12.0 \times 28.0\text{ meter}$.
     * **Flyover Mall Boemi Kedaton**: Bentang layang $10.0 \times 30.0\text{ meter}$.
     * **Underpass Unila**: Bidang jalan $7.5 \times 45.0\text{ meter}$.
   - Dihaluskan dengan filter eksponensial *Exponential Moving Average (EMA)* ($\alpha = 0.35$).

4. **Mesin Deteksi Anomali & Keselamatan Jalan Tri-Lokasi**:
   - 🚨 **Tabrakan Kendaraan (*Collision / Crash*)**: Bounding box overlap $\text{IoU} \ge 0.45$ disertai deselerasi drastis menuju diam.
   - 🏍️ **Sepeda Motor Terjatuh (*Fallen Motorcycle / Spill*)**: Pembalikan rasio aspek geometris motor $\text{Aspect Ratio} = W/H \ge 1.30$.
   - 🛑 **Kemacetan Simpang & Kendaraan Mogok (*Gridlock / Obstruction*)**: Kendaraan terhenti $\ge 3.0\text{ detik}$ di lajur aktif atau bentang layang.
   - ⚠️ **Anomali Khusus Flyover MBK**: Deteksi kendaraan mogok di jembatan layang (risiko tabrakan beruntun tinggi), motor tergelincir di sambungan siar muai (*expansion joint*), dan *overspeeding* turunan.
   - ⛔ **Pelanggaran Lawan Arah (*Wrong-Way Driving*)**: 
     * Pada Simpang Sudirman: Memantau kendaraan melaju melawan konfigurasi arus belokan resmi.
     * Pada Underpass Unila: Aturan *barrier-aware* linier $x_{\text{barrier}}(y)$ membedakan hak lajur naik vs lajur turun tanpa *false positive*.
   - 📸 **Perekaman Bukti Visual Otomatis**: Setiap insiden menghasilkan cuplikan gambar resolusi penuh (*snapshot proof*) dan tercatat ke CSV.

5. **Dashboard Command Center Terpadu (Streamlit)**:
   - **Tab 1: 📺 Monitor Video & Log Insiden**: Penampil video terpadu (*Unified Camera Area*) dengan tombol "🚀 Jalankan Inferensi AI" dan "🛑 Hentikan Inferensi AI (Kembali ke Siaran CCTV)", 9 kartu KPI metrik real-time, dan log insiden.
   - **Tab 2: 📊 Data Telemetri**: Tabel dataset interaktif dengan filter rentang tanggal (*Date Range Picker*), tombol unduh CSV terfilter, dan fitur pencadangan/reset data aman (*safe-guard reset*).
   - **Tab 3: 📈 Grafik**: 6 grafik visualisasi publikasi ilmiah standar IEEE/Elsevier.
   - **Tab 4: ⚖️ Analisis Komparasi**: Matriks perbandingan ilmiah 3 kolom karakteristik Simpang Sudirman (Arus Terputus) vs Flyover MBK (Arus Menerus Layang) vs Underpass Unila (Arus Menerus Bawah Tanah).
   - **Tab 5: 🛠️ Pengujian**: Suite pengujian otomatis berbasis 4 pilar (Fungsional, Non-Fungsional, Integrasi, dan AI Model Testing) langsung dari web.

---

## 📁 Struktur Berkas

```
CCTV/
├── app.py                      # Dashboard Utama Streamlit (5 Tab Komprehensif)
├── cctv_stream.py              # HLS Stream Reader & Sesi Token (Default CCTV 282)
├── detector.py                 # YOLOv8 Vehicle Detector & Heuristik Helm ETLE
├── speed_estimator.py          # Homografi Planar (Kalibrasi Sudirman & Unila)
├── anomaly_detector.py         # Mesin Deteksi Anomali (Simpang & Barrier-Aware)
├── pipeline.py                 # Pipeline Orkestrasi Analitik Real-Time
├── data_logger.py              # Perekam Telemetri & Insiden CSV (Thread-Safe)
├── charts_journal.py           # Modul 6 Grafik Publikasi Ilmiah Interaktif
├── utility_tester.py           # Runner Pengujian Fungsional & Non-Fungsional
├── run_comprehensive_tests.py  # Suite Uji Otomatis 4 Pilar (FT, NFT, IT, AIM)
├── run_analytics.py            # Runner CLI Headless Processing & Laporan JSON
├── SDLC_DOCS/                  # 5 Dokumen Formal Siklus Hidup Rekayasa (SDLC 01-05)
│   ├── 01_SDLC_METHODOLOGY_AND_FRAMEWORK.md
│   ├── 02_REQUIREMENTS_SPECIFICATION_SRS.md
│   ├── 03_SYSTEM_ARCHITECTURE_AND_DESIGN_SDD.md
│   ├── 04_ALGORITHMS_AND_MATHEMATICAL_FORMULATIONS.md
│   └── 05_VERIFICATION_VALIDATION_AND_TESTING_STD.md
├── DOKUMEN_METODOLOGI_PENELITIAN_DAN_SDLC.md # Naskah Lengkap Metodologi & Riset
├── logs/
│   ├── traffic_telemetry.csv   # Dataset telemetri kendaraan per frame
│   ├── incident_records.csv    # Catatan insiden, tabrakan, & pelanggaran ETLE
│   └── archive/                # Direktori arsip cadangan saat data di-reset
├── requirements.txt            # Dependensi pustaka Python
├── yolov8n.pt                  # Bobot model YOLOv8 Nano
└── frame_sudirman_sample.jpg   # Snapshot acuan kalibrasi geometri Simpang Sudirman
```

---

## 🚀 Panduan Menjalankan Sistem

### 1. Menjalankan Dashboard Web Interaktif
```bash
./.venv/bin/streamlit run app.py
```
Buka browser pada `http://localhost:8501`. Secara otomatis kamera primer **LIVE - PEREMPATAN JENDRAL SUDIRMAN (CCTV ID 282)** akan dimuat. Anda dapat beralih ke Underpass Unila kapan saja melalui menu dropdown lokasi di bilah samping (*sidebar*).

### 2. Menjalankan Suite Pengujian Komprehensif (4 Pilar Testing)
```bash
./.venv/bin/python run_comprehensive_tests.py
```
Memverifikasi:
- **Pilar 1**: Functional Testing (FT-01 s.d. FT-18).
- **Pilar 2**: Non-Functional Testing (latensi homografi, throughput filter tanggal, performa biner CSV).
- **Pilar 3**: Integration Testing (aliran pipeline, deteksi helm, reset & arsip).
- **Pilar 4**: AI Model & Computer Vision Testing (mAP YOLOv8, kestabilan ByteTrack, invers homografi, ROI containment).

### 3. Menjalankan Pemrosesan Headless CLI
```bash
# Memproses live stream CCTV 282 Sudirman selama 20 detik
./.venv/bin/python run_analytics.py --source live --duration 20 --output hasil_sudirman.mp4 --save-json laporan_insiden.json
```
