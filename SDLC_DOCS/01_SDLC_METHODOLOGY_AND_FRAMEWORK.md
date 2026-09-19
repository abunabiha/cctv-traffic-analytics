# DOKUMEN TAHAPAN SDLC 01: METODOLOGI PENGEMBANGAN SISTEM & KERANGKA KERJA (SDLC METHODOLOGY & FRAMEWORK)

**Nama Proyek:** Rancang Bangun Sistem Analitik Cerdas CCTV Berbasis YOLOv8, Transformasi Homografi Perspektif, dan Deteksi Anomali Temporal untuk Pemantauan Arus Lalu Lintas dan Keselamatan Jalan Koridor Perkotaan Bandar Lampung (Tri-Lokasi: Simpang Sudirman, Flyover Mall Boemi Kedaton, dan Underpass Unila)  
**Lokasi Studi Kasus (Tri-Lokasi):**  
1. **CCTV CCTV Perempatan Jendral Sudirman (CCTV ID: 282) [LOKASI 1 / DEFAULT]** - Simpang Bersinyal Sebidang Pusat Kota (*Interrupted Flow*)  
2. **CCTV CCTV Flyover Mall Boemi Kedaton (CCTV ID: 190) [LOKASI 2 / BARU]** - Jembatan Layang Menerus Bebas Hambatan (*Elevated Continuous Flow*)  
3. **CCTV CCTV Unila Arah Rajabasa (CCTV ID: 312) [LOKASI 3]** - Jalan Terowongan Bawah Tanah Terpisah Barrier Median (*Depressed Continuous Flow*)  
**Bidang Keilmuan:** *Computer Vision*, *Intelligent Transportation Systems (ITS)*, *Deep Learning*, *Traffic Anomaly Detection*, *Software Engineering*  
**Versi Dokumen:** 3.0 (Tri-Site Urban Corridor Edition)  

---

## 1. Ringkasan Eksekutif & Identifikasi Proyek

Pengembangan sistem pemantauan lalu lintas cerdas (*Smart City Traffic Video Analytics*) berbasis *Intelligent Video Analytics (IVA)* memerlukan pendekatan rekayasa perangkat lunak yang sistematis, terukur, dan mampu mengakomodasi sifat dinamis dari model *Deep Learning* serta variasi lingkungan fisik jalan raya perkotaan. 

Sistem ini dirancang untuk mengubah kamera CCTV pemantau perkotaan pasif milik Pemerintah Kota dan Dinas Perhubungan Kota Bandar Lampung menjadi sistem pemantauan proaktif dan real-time. Titik pengamatan mencakup arsitektur komparasi **Tri-Lokasi Perkotaan**:
1. **CCTV CCTV Perempatan Jendral Sudirman (CCTV ID: 282)** [Default/Primer] yang memiliki karakteristik simpang 4 bersinyal dengan volume kendaraan padat dan dinamika antrean *traffic light*.
2. **CCTV CCTV Flyover Mall Boemi Kedaton (CCTV ID: 190)** [Baru] yang memiliki karakteristik jembatan layang bebas hambatan (*elevated corridor*) 2 lajur sepanjang 262 meter dengan kecepatan menengah-tinggi dan resiko bahaya turunan serta mogok di atas bentang jembatan.
3. **CCTV CCTV Unila Arah Rajabasa (CCTV ID: 312)** untuk evaluasi karakteristik terowongan bawah tanah dua arah dengan pemisah barrier fisik median (*depressed corridor*).

Modul utama sistem mencakup pengenalan penggolongan kendaraan Standar Indonesia (Golongan I s.d. VI), pemantauan kepatuhan penggunaan helm SNI (Golongan VI-A vs VI-B), estimasi kecepatan kendaraan nyata (km/jam) menggunakan matriks homografi perspektif tri-lokasi, deteksi dini anomali dan kecelakaan lalu lintas dengan aturan pembatas median fisik (*barrier-aware*) dan bahaya layang, serta logging telemetri otomatis ke format CSV.

---

## 2. Metodologi Pengembangan: Integrasi Agile-Scrum dan CRISP-DM

Untuk menggabungkan keunggulan siklus rekayasa perangkat lunak modern dengan karakteristik eksperimental *data science* dan *computer vision*, sistem ini dikembangkan menggunakan kerangka kerja hibrida **Agile-Scrum terintegrasi CRISP-DM (Cross-Industry Standard Process for Data Mining)**.

```
+-----------------------------------------------------------------------------+
|                 KERANGKA KERJA HIBRIDA: CRISP-DM + AGILE SCRUM              |
+-----------------------------------------------------------------------------+
|  CRISP-DM Dimension (Data & AI)     |  Agile SDLC Dimension (Software)     |
|-------------------------------------+---------------------------------------|
|  1. Business & Problem Understanding |  Sprint 0: Requirement & Backlog Prep |
|  2. Data Ingestion & Exploration    |  Sprint 1: Stream Engine & Protocol   |
|  3. Data Preparation & Calibration   |  Sprint 2: Spatial ROI & Homography   |
|  4. AI Modeling & Tracking          |  Sprint 3: YOLOv8 & ByteTrack Tuning  |
|  5. Rule Engine & Anomaly Logic     |  Sprint 4: Barrier-Aware Anomaly Dev  |
|  6. Telemetry & Data Logging        |  Sprint 5: Real-time CSV Data Logger  |
|  7. Dashboard & UI Implementation   |  Sprint 6: Streamlit Command Center   |
|  8. Evaluation & Load Testing       |  Sprint 7: Stress Test & Hardening    |
+-----------------------------------------------------------------------------+
```

### Alasan Pemilihan Model Hibrida:
1. **Iteratif & Adaptif:** Mengakomodasi ketidakpastian kondisi visual di lapangan (perubahan cuaca, pencahayaan malam, sudut kamera miring/oblique).
2. **Umpan Balik Cepat (*Rapid Feedback Loop*):** Setiap sprint menghasilkan *working increment* yang dapat langsung diuji pada aliran HLS langsung dari server Dishub.
3. **Kepatuhan Regulasi Nasional:** Mengintegrasikan standar klasifikasi tarif tol/jalan Kementerian PUPR / Kepmenhub dan aturan ETLE (Electronic Traffic Law Enforcement) Korlantas Polri.

---

## 3. Rincian Fase Siklus Hidup Pengembangan Sistem (SDLC Phases)

### Fase 1: Rekayasa Kebutuhan & Karakterisasi Wilayah Studi (*Requirements Engineering*)
- **Karakterisasi Simpang Sudirman (CCTV ID 282 - Default/Primer):** Menganalisis topologi simpang 4 bersinyal di pusat kota Jl. Jendral Sudirman, area manuver 4 lajur, zona henti *traffic light*, titik silang belok, dan densitas tinggi sepeda motor dan kendaraan pribadi (*interrupted flow*).
- **Karakterisasi Flyover Mall Boemi Kedaton (CCTV ID 190 - Baru):** Menganalisis struktur jembatan layang (flyover) 2 lajur sepanjang 262 meter di atas persimpangan rel KA dan Jl. Teuku Umar - Jl. Z.A. Pagar Alam, dengan arus menerus (*continuous elevated flow*), kecepatan laju 35–65 km/jam, risiko *overspeeding* turunan, dan bahaya kemacetan/kendaraan mogok di atas bentang jembatan.
- **Karakterisasi Underpass Unila (CCTV ID 312):** Menganalisis turunan curam menuju terowongan bawah tanah, dinding penahan berornamen mural, dan keberadaan **Barrier Fisik Median** di tengah jalan yang memisahkan dua arah pergerakan (*depressed continuous flow*).
- Menetapkan kebutuhan fungsional: klasifikasi multi-golongan (I, II, III, IV, V, VI-A, VI-B), pengukuran kecepatan via homografi tri-lokasi, deteksi insiden (tabrakan simpang, motor jatuh, mogok di bentang jembatan/terowongan, lawan arah barrier-aware), serta ekspor dataset telemetri.

### Fase 2: Reverse Engineering API & Ingestion Aliran Video (*Video Ingestion & Protocol Analysis*)
- Melakukan deobfuskasi berkas JavaScript klien pada portal `seribuwajah.bandarlampungkota.go.id`.
- Merumuskan mekanisme *session handshake* HTTP Cookie (`cookieCheck=1` dan token `hlsSession`) untuk mengunduh playlist m3u8 (`/cctv_282/index.m3u8`, `/cctv_190/index.m3u8`, dan `/cctv_312/index.m3u8`) dan berkas *transport stream* (`.ts`).
- Membangun kelas `LiveHLSStreamReader` dengan mekanisme antrean multi-threading dan buffer pada penyimpanan lokal `/tmp` guna memitigasi latensi Google Drive CloudStorage.

### Fase 3: Pemfilteran Spasial & Kalibrasi Bidang Jalan (*Spatial ROI & Calibration*)
- Menentukan poligon *Road Region of Interest (Road ROI)* adaptif untuk mengeliminasi area non-jalan:
  * `SUDIRMAN_ROAD_ROI`: Meliputi badan jalan simpang 4 Sudirman dan mengabaikan trotoar ruko serta pepohonan.
  * `MBK_ROAD_ROI`: Meliputi bentang aspal jembatan layang Flyover MBK dan mengeliminasi atap gedung ruko serta pembatas flyover.
  * `DEFAULT_ROAD_ROI`: Meliputi aspal Underpass Unila dan mengeliminasi trotoar atas serta mural dinding.
- Mengonfigurasi matriks transformasi homografi planar $\mathbf{H}_{3 \times 3}$ tri-lokasi:
  * Simpang Sudirman ($12.0 \times 28.0$ meter area lajur simpang).
  * Flyover Mall Boemi Kedaton ($10.0 \times 30.0$ meter bentang lajur jembatan layang).
  * Underpass Unila ($7.5 \times 45.0$ meter area lajur terowongan).

### Fase 4: Pemodelan AI Deteksi, Klasifikasi, & Tracking Kendaraan
- Mengintegrasikan arsitektur YOLOv8 Nano (`yolov8n.pt`) untuk deteksi objek real-time dengan efisiensi CPU yang optimal.
- Menggabungkan model deteksi dengan pelacak berbasis asosiasi Kalman Filter *ByteTrack* untuk konsistensi ID unik kendaraan antar-frame.
- Mengembangkan subsistem Computer Vision untuk deteksi helm pengendara motor berbasis segmentasi warna kulit (ruang warna HSV dan YCbCr), rasio aspek bounding box, dan analisis kurvatur elipsoid.

### Fase 5: Perancangan Mesin Aturan Anomali & Keselamatan Jalan (*Rule Engine*)
- Mengimplementasikan aturan deteksi tabrakan (*collision*) berbasis rasio tumpang-tindih bounding box (*Intersection over Union* / IoU > 0.45) disertai deselerasi mendadak.
- Mengimplementasikan aturan deteksi sepeda motor jatuh (*fallen motorcycle/spill*) berdasarkan inversi rasio aspek ($\text{AR} > 1.30$).
- Mengimplementasikan aturan deteksi kemacetan / kendaraan mogok terhenti ($\text{speed} < 3.0\text{ km/jam}$ selama durasi $> \text{threshold}$).
- **Logika Anomali Spesifik Tri-Lokasi**:
  * Simpang Sudirman: Deteksi kemacetan antrean simpang panjang, kendaraan menerobos lampu/arah berlawanan arus, dan tabrakan di persimpangan.
  * Flyover Mall Boemi Kedaton: Deteksi kendaraan mogok/berhenti di atas jembatan layang (membahayakan keselamatan arus bebas), kendaraan roda dua tergelincir pada siar muai (*expansion joint*), dan pelanggaran kecepatan tinggi di turunan.
  * Underpass Unila: Deteksi *barrier-aware* menggunakan fungsi pemisah linier $x_{\text{barrier}}(y)$ sehingga kendaraan di lajur kiri (naik arah Rajabasa) dan lajur kanan (turun underpass) tidak memicu *false positive* lawan arah.

### Fase 6: Arsitektur Perekaman Data Telemetri & Insiden (*Data Logging Layer*)
- Merancang kelas `TrafficDataLogger` yang mencatat data kendaraan per frame ke dalam berkas `traffic_telemetry.csv` dengan penandaan `location_id` (282, 190, 312).
- Merancang pencatatan log insiden kecelakaan dan bukti snapshot terkompresi ke dalam `incident_records.csv`.
- Menerapkan thread-safe buffer flush dan pembacaan berkas cepat berbasis biner (`get_raw_csv_bytes`) untuk unduhan instan (< 0.1 ms).

### Fase 7: Implementasi Antarmuka Pengguna & Dasbor Analitik (*Interactive UI*)
- Mengembangkan Command Center berbasis Streamlit dengan arsitektur *Unified Camera Area* dan selector 3 kamera CCTV (Sudirman, Flyover MBK, Underpass Unila):
  * Mode Preview: Pemutar HTML5 terintegrasi HLS.js untuk streaming instan tanpa beban komputasi.
  * Mode Inferensi AI: Menampilkan video frame-by-frame beranotasi lengkap (bounding box, ID, golongan, kepatuhan helm, kecepatan, dan garis batas barrier) di posisi layar yang sama.
- Membangun 9 kartu metrik KPI dinamis, tabel dataset CSV terfilter tanggal, dan tab Analisis Komparasi Tri-Lokasi (Simpang Sebidang vs Jembatan Layang vs Terowongan Bawah Tanah).

### Fase 8: Pengujian Beban, Verifikasi, & Deployment (*Testing & Hardening*)
- Melakukan uji unit sintaksis AST dan fungsional pada seluruh modul backend.
- Melakukan *Performance Load Testing* multi-threading (1 s.d. 20 concurrent users) untuk menguji responsivitas HTTP server.
- Membersihkan seluruh blok kode yang tidak terpakai (*clean code*) guna menjamin pemuatan antarmuka yang cepat dan bebas *blocking*.

---

## 4. Jadwal Sprint, Rencana Kerja (WBS), & RACI Matrix

### 4.1 Work Breakdown Structure (WBS)
```
1.0 Inisiasi & Analisis Kebutuhan
    1.1 Studi Karakteristik Tri-Lokasi (Simpang Sudirman, Flyover MBK, Underpass Unila)
    1.2 Analisis Standar Penggolongan Kendaraan Indonesia (Gol I - VI)
    1.3 Spesifikasi Kebutuhan Sistem (SRS)
2.0 Rekayasa Aliran Video CCTV
    2.1 Analisis Protokol HLS & Sesi Token Dishub (3 Endpoint CCTV)
    2.2 Pembangunan LiveHLSStreamReader & Worker Buffer
3.0 Inti Computer Vision & Deep Learning
    3.1 Integrasi YOLOv8 & ByteTrack Tracker
    3.2 Kalibrasi Matriks Homografi Perspektif Tri-Lokasi (Speed Estimator)
    3.3 Algoritma Deteksi Kepatuhan Helm (HSV/YCbCr & Convexity)
4.0 Mesin Aturan Anomali (Anomaly Engine)
    4.1 Logika Deteksi Tabrakan & Deselerasi
    4.2 Logika Motor Terjatuh (Aspect Ratio Flip)
    4.3 Logika Barrier-Aware Underpass & Anomali Bentang Jembatan Layang MBK
5.0 Penyimpanan Data & Logging
    5.1 Skema Tabel traffic_telemetry.csv & incident_records.csv
    5.2 Thread-Safe File I/O & Biner Cache
6.0 Antarmuka Dasbor Web (Streamlit UI)
    6.1 Layout Command Center, KPI Cards, & Unified Video Viewer (3 CCTV Selector)
    6.2 Visual Komparasi Rekayasa Transportasi Tri-Lokasi
7.0 Pengujian & Verifikasi
    7.1 Unit & Integration Testing (Suite 16 Skenario Terpadu)
    7.2 Load Testing (Cold, Warm, Concurrency)
    7.3 Pembersihan Dead Code & Optimasi Google Drive I/O
```

### 4.2 RACI Matrix
| Aktivitas / Deliverable | System Architect | AI/CV Engineer | Software Developer | QA/Test Engineer |
| :--- | :---: | :---: | :---: | :---: |
| Spesifikasi Kebutuhan Sistem (SRS) | **Accountable** | Consulted | Informed | Consulted |
| Ingestion HLS & Sesi Stream | Consulted | Consulted | **Responsible** | Informed |
| Pipeline YOLOv8 & Tracking | Informed | **Responsible** | Consulted | Consulted |
| Kalibrasi Homografi & Kecepatan | Consulted | **Responsible** | Informed | Consulted |
| Aturan Anomali Barrier-Aware | **Accountable** | **Responsible** | Consulted | Consulted |
| UI Command Center Streamlit | Consulted | Informed | **Responsible** | Consulted |
| Load Testing & Verifikasi | Consulted | Informed | Consulted | **Responsible** |
| Final Clean Code & Dokumentasi | **Accountable** | Consulted | **Responsible** | Informed |

---

## 5. Manajemen Risiko & Strategi Mitigasi

| Risiko Potensial | Probabilitas | Dampak | Strategi Mitigasi yang Diterapkan |
| :--- | :---: | :---: | :--- |
| **Koneksi Server CCTV Terputus / Jitter HLS** | Sedang | Tinggi | Sistem dilengkapi antrean buffer mandiri dan *instant fallback* ke snapshot/frame referensi terkalibrasi sehingga visualisasi tidak pernah *hang*. |
| **Latensi I/O Google Drive CloudStorage** | Tinggi | Tinggi | Seluruh impor pustaka berat (OpenCV/PyTorch) dipindahkan ke *lazy loading*; buffer penulisan video `.ts` diarahkan ke SSD lokal `/tmp`; pemindaian berkas Streamlit dimatikan (`fileWatcherType = "none"`). |
| **False Positive Lawan Arah di Underpass** | Tinggi | Tinggi | Mengganti ambang batas vektor linier sederhana dengan fungsi pemisah median barrier dinamis $x_{\text{barrier}}(y)$ yang membedakan hak lajur naik vs lajur turun. |
| **Beban Komputasi CPU Berlebih pada Real-Time AI** | Sedang | Sedang | Memilih arsitektur YOLOv8 Nano (`yolov8n.pt`) dengan *frame sampling rate* 12.5 FPS yang sinkron dengan kapasitas pemrosesan CPU *edge*. |
| **Tagging Bounding Box Tidak Terlihat di Layar** | Sedang | Tinggi | Mengganti komponen iframe terpisah dengan *Unified Camera Area* (`camera_area = st.empty()`) yang secara instan mengganti pemutar raw dengan stream beranotasi bounding box saat tombol ditekan. |

---

## 6. Kesimpulan Dokumen

Kerangka kerja SDLC terintegrasi ini memastikan bahwa seluruh tahapan perancangan, pengembangan model AI, pengkodean backend, dan pembuatan antarmuka berjalan selaras dengan standar rekayasa perangkat lunak internasional, menghasilkan sistem analitik CCTV yang tangguh, akurat, cepat, dan siap digunakan untuk pemantauan lalu lintas jalan raya kota.
