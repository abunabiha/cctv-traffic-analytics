# DOKUMEN TAHAPAN SDLC 02: SPESIFIKASI KEBUTUHAN PERANGKAT LUNAK (SOFTWARE REQUIREMENTS SPECIFICATION - SRS)

**Standar Dokumentasi:** Mengacu pada IEEE Std 830-1998 (Recommended Practice for Software Requirements Specifications)  
**Nama Sistem:** Smart CCTV Traffic Analytics & Anomaly Detection System (Tri-Lokasi: Simpang Sudirman, Flyover Mall Boemi Kedaton, & Underpass Unila)  
**Versi:** 3.0 (Tri-Site Urban Corridor Edition)  
**Status:** Approved & Implemented  

---

## 1. PENDAHULUAN

### 1.1 Tujuan Dokumen
Dokumen Spesifikasi Kebutuhan Perangkat Lunak (*Software Requirements Specification* - SRS) ini mendefinisikan secara menyeluruh kebutuhan fungsional dan non-fungsional dari **Sistem Analitik Cerdas CCTV Lalu Lintas**. Dokumen ini menjadi rujukan formal bagi perancang sistem, pengembang AI/perangkat lunak, penguji mutu (*QA*), serta instansi terkait (Dinas Perhubungan dan Kepolisian Lalu Lintas) dalam memverifikasi kapabilitas operasional sistem.

### 1.2 Ruang Lingkup Sistem
Sistem ini merupakan perangkat lunak *Intelligent Video Analytics (IVA)* berbasis web yang memproses aliran video kamera CCTV langsung (*real-time HLS streaming*) dari tiga lokasi fasilitas transportasi perkotaan di Kota Bandar Lampung:
1. **CCTV Perempatan Jendral Sudirman (CCTV ID 282) [LOKASI 1 / DEFAULT]:** Simpang empat bersinyal perkotaan berdensitas tinggi dengan dinamika antrean kendaraan *traffic light*, area manuver 4 lajur, dan titik konflik pergerakan silang (*interrupted flow*).
2. **CCTV Flyover Mall Boemi Kedaton (CCTV ID 190) [LOKASI 2 / BARU]:** Jembatan layang (flyover) arteri 2 lajur sepanjang 262 meter di atas persimpangan rel KA dan Mall Boemi Kedaton dengan kecepatan menerus 35–65 km/jam, risiko kecelakaan turunan dan bahaya mogok di atas bentang (*elevated continuous flow*).
3. **CCTV Unila Arah Rajabasa (CCTV ID 312) [LOKASI 3]:** Ruas jalan arteri berkarakteristik turunan terowongan bebas hambatan dengan dua lajur yang dipisahkan secara fisik oleh pembatas median (*depressed continuous flow*).

Sistem bertugas mendeteksi, melacak, mengelompokkan kendaraan menurut Standar Golongan I s.d. VI Indonesia, mendeteksi kepatuhan helm pengendara roda dua (Golongan VI-A vs VI-B), mengukur kecepatan pergerakan nyata (km/jam) dengan kalibrasi homografi tri-lokasi, mendeteksi anomali/kecelakaan secara otomatis, serta mencatat data telemetri ke dalam format tabular CSV.

### 1.3 Definisi, Akronim, dan Singkatan
- **IVA:** *Intelligent Video Analytics*
- **ITS:** *Intelligent Transportation Systems*
- **HLS:** *HTTP Live Streaming* (protokol streaming video adaptif berbasis segmen `.ts` dan playlist `.m3u8`)
- **YOLOv8:** *You Only Look Once* Versi 8 (arsitektur *deep learning* deteksi objek satu tahap)
- **ByteTrack:** Algoritma pelacakan multi-objek berbasis asosiasi deteksi kemiripan spasial dan Kalman Filter
- **Homografi Planar:** Transformasi proyektif antara dua bidang datar dalam geometri proyektif
- **ETLE:** *Electronic Traffic Law Enforcement* (Penegakan Hukum Lalu Lintas Elektronik)
- **Golongan I - VI:** Sistem klasifikasi jenis kendaraan dan tarif jalan standar nasional Republik Indonesia
- **Barrier Median:** Struktur fisik pembatas di tengah jalan yang memisahkan dua arah arus lalu lintas

---

## 2. DESKRIPSI KESELURUHAN SISTEM

### 2.1 Perspektif Produk
Sistem beroperasi sebagai aplikasi web analitik mandiri (*standalone web analytics application*) yang bertindak sebagai lapisan kecerdasan (*intelligence layer*) di atas infrastruktur CCTV publik Dinas Perhubungan Kota Bandar Lampung (`seribuwajah.bandarlampungkota.go.id`). Sistem tidak membutuhkan instalasi perangkat keras sensor fisik tambahan di permukaan aspal (*non-intrusive vision-based approach*).

```
[Server CCTV Dishub] ---> (HLS Stream m3u8) ---> [LiveHLSStreamReader]
                                                        │
                                                        ▼
                                          [Pipeline Analitik Cerdas]
                                          ├─ YOLOv8 Vehicle Detector
                                          ├─ ByteTrack Multi-Object Tracker
                                          ├─ Homography Speed Estimator
                                          ├─ Barrier-Aware Anomaly Engine
                                          └─ Real-Time CSV Data Logger
                                                        │
                                                        ▼
                                          [Streamlit Command Center UI]
                                          ├─ Unified Camera Stream + Tagging
                                          ├─ 9 KPI Metrics Cards
                                          ├─ Telemetry & Incident Tables
                                          └─ Comparative Analytics Tab
```

### 2.2 Karakteristik Pengguna
1. **Operator Command Center / Dishub:** Mengawasi visual jalan raya, memantau laju kecepatan rerata, dan menerima notifikasi peringatan insiden darurat.
2. **Petugas Penegak Hukum (ETLE / Satlantas):** Memverifikasi pelanggaran tidak menggunakan helm (Golongan VI-B) dan pelanggaran melawan arah melintasi median barrier.
3. **Peneliti / Analis Transportasi:** Mengunduh berkas telemetri `traffic_telemetry.csv` dan rekaman insiden `incident_records.csv` untuk analisis statistik makro maupun mikro lalu lintas.

### 2.3 Batasan Sistem (*Constraints*)
1. Aliran video bergantung pada ketersediaan koneksi internet dan status operasional server CCTV Dishub Bandar Lampung.
2. Resolusi input kamera standar adalah 1280x720 piksel pada rasio aspek 16:9.
3. Pemrosesan harus mampu berjalan optimal pada arsitektur CPU tanpa kewajiban keberadaan GPU diskrit server.

---

## 3. KEBUTUHAN FUNGSIONAL (*FUNCTIONAL REQUIREMENTS*)

### FR-01: Ingestion Aliran Video HLS Real-Time Multi-Kamera
- **Deskripsi:** Sistem harus mampu melakukan *handshake* sesi HTTP Cookie dengan server CCTV Dishub, mengunduh playlist `index.m3u8` dan `main_stream.m3u8` secara dinamis, serta membaca segmen `.ts` secara kontinu tanpa interupsi untuk tiga kamera pengamatan: **CCTV ID 282 (Simpang Sudirman)**, **CCTV ID 190 (Flyover Mall Boemi Kedaton)**, dan **CCTV ID 312 (Underpass Unila)**.
- **Kriteria Penerimaan:** Sistem mempertahankan *buffer* video di direktori SSD lokal `/tmp` dan mampu menyajikan frame video ke antrean antarmuka dengan latensi rendah.

### FR-02: Pemfilteran Spasial Area Badan Jalan (*Road ROI*)
- **Deskripsi:** Sistem harus membatasi area analisis deteksi kendaraan hanya pada badan jalan raya yang relevan menggunakan poligon *Road Region of Interest (ROI)* adaptif per lokasi.
- **Kriteria Penerimaan:** 
  * Pada Simpang Sudirman (`SUDIRMAN_ROAD_ROI`): Mengisolasi area persimpangan 4 lajur dan mengabaikan trotoar ruko serta area parkir samping.
  * Pada Flyover Mall Boemi Kedaton (`MBK_ROAD_ROI`): Mengisolasi bentang aspal 2 lajur jembatan layang dan mengabaikan atap bangunan komersial samping.
  * Pada Underpass Unila (`DEFAULT_ROAD_ROI`): Mengisolasi badan aspal terowongan dan mengabaikan trotoar atas serta mural dinding dengan toleransi margin batas -45 piksel.

### FR-03: Deteksi & Pelacakan Objek Multi-Kendaraan (*Tracking*)
- **Deskripsi:** Sistem harus mendeteksi keberadaan kendaraan pada setiap frame dan mempertahankan ID pelacakan unik (*track_id*) antar-frame secara konsisten.
- **Kriteria Penerimaan:** Menggunakan model YOLOv8 Nano terintegrasi pelacak ByteTrack dengan mekanisme *centroid fallback tracker* jika ID tracker sempat terputus.

### FR-04: Klasifikasi Kendaraan Standar Indonesia (Golongan I s.d. VI)
- **Deskripsi:** Sistem harus memetakan deteksi dasar COCO ke dalam kategori resmi Golongan I s.d. VI:
  * **Golongan I:** Sedan, Jip, Pick-up, Bus, dan Truk Kecil.
  * **Golongan II:** Truk dengan 2 (dua) gandar (sumbu roda).
  * **Golongan III:** Truk dengan 3 (tiga) gandar (Tronton).
  * **Golongan IV:** Truk dengan 4 (empat) gandar.
  * **Golongan V:** Truk dengan 5 (lima) gandar atau lebih (Trailer).
  * **Golongan VI-A:** Sepeda Motor Taat Hukum (Pengendara Memakai Helm SNI).
  * **Golongan VI-B:** Sepeda Motor Melanggar Aturan (Pengendara Tanpa Helm).
- **Kriteria Penerimaan:** Penggolongan didasarkan pada kombinasi kelas dasar, estimasi dimensi fisik meter (panjang & lebar), rasio aspek bounding box, dan deteksi atribut helm.

### FR-05: Penegakan Hukum Kepatuhan Helm (ETLE Motor)
- **Deskripsi:** Sistem harus mengevaluasi area atas pengendara sepeda motor untuk membedakan antara pengendara yang mengenakan helm (Golongan VI-A) dan tanpa helm (Golongan VI-B).
- **Kriteria Penerimaan:** Evaluasi menggunakan segmentasi warna kulit pada ruang warna HSV dan YCbCr, rasio kebulatan batok kepala (*convexity*), reflektansi specular, serta konsistensi temporal voting (minimal 15 frame terakhir).

### FR-06: Estimasi Kecepatan Nyata Kendaraan (km/jam)
- **Deskripsi:** Sistem harus mengukur laju kecepatan translasi setiap kendaraan dalam satuan km/jam.
- **Kriteria Penerimaan:** Menggunakan matriks homografi perspektif $\mathbf{H}$ spesifik lokasi:
  * Simpang Sudirman: Dikalibrasi pada bidang jalan $12.0 \times 28.0\text{ meter}$.
  * Flyover Mall Boemi Kedaton: Dikalibrasi pada bidang layang $10.0 \times 30.0\text{ meter}$.
  * Underpass Unila: Dikalibrasi pada bidang jalan $7.5 \times 45.0\text{ meter}$.
  * Koordinat fisik $(X, Y)$ dihaluskan dengan *Exponential Moving Average (EMA)* ($\alpha = 0.35$).

### FR-07: Mesin Deteksi Anomali & Insiden Lalu Lintas
Sistem harus mendeteksi secara otomatis kondisi bahaya spesifik:
1. **FR-07.1 (Tabrakan / Collision):** Dua kendaraan memiliki tumpang tindih bounding box $\text{IoU} > 0.45$ disertai perlambatan drastis menjadi $< 5\text{ km/jam}$.
2. **FR-07.2 (Sepeda Motor Jatuh / Spill):** Kendaraan roda dua mengalami pembalikan rasio aspek geometris di mana lebar bounding box menjadi jauh lebih besar daripada tinggi ($\text{AR} > 1.30$).
3. **FR-07.3 (Kemacetan Simpang & Kendaraan Berhenti / Obstruction):** Kendaraan terhenti pada badan jalan aktif melebihi batas waktu toleransi ($\ge 3.0\text{ detik}$).
4. **FR-07.4 (Lawan Arah & Pelanggaran Lajur Khusus / Wrong-Way & Facility Hazard):** 
   - Pada Simpang Sudirman: Mendeteksi kendaraan melaju berlawanan arah dengan konfigurasi arus belokan resmi.
   - Pada Flyover Mall Boemi Kedaton: Mendeteksi kendaraan mogok/berhenti mendadak di atas bentang jembatan layang (resiko tabrakan beruntun), overspeeding turunan, dan manuver berbahaya.
   - Pada Underpass Unila (dua arah dipisah barrier fisik median):
     * Lajur Kanan ($x > x_{\text{barrier}}(y)$): Arah legal adalah turun ($\Delta y > 0$). Pelanggaran terpicu jika melaju ke atas melawan arus ($\Delta y < -25\text{ px}$).
     * Lajur Kiri ($x \le x_{\text{barrier}}(y)$): Arah legal adalah naik ke Rajabasa ($\Delta y < 0$). Pelanggaran terpicu jika melaju ke bawah memasuki lajur berlawanan ($\Delta y > 25\text{ px}$).
     * Pelanggaran juga terpicu apabila kendaraan memotong/melintasi garis koordinat median barrier.

### FR-08: Perekaman Data Telemetri & Insiden ke CSV
- **Deskripsi:** Sistem harus mencatat setiap frame pengamatan ke dalam berkas `logs/traffic_telemetry.csv` dan rekaman insiden ke `logs/incident_records.csv` dengan menyertakan atribut `location_id` (282, 190, 312) dan `location_name`.
- **Kriteria Penerimaan:** Berkas CSV tersimpan secara thread-safe dan dapat diunduh langsung dari dasbor web melalui pembacaan biner instan.

### FR-09: Antarmuka Unified Camera Display dengan Tombol Kontrol Inferensi
- **Deskripsi:** Antarmuka harus menyajikan area kamera terpadu (*Unified Camera Area*) dengan kontrol penuh atas siklus inferensi untuk ketiga kamera.
- **Kriteria Penerimaan:** Saat mode preview aktif, pemutar video menampilkan siaran langsung HLS. Saat tombol "🚀 Jalankan Inferensi AI" ditekan, pemutar secara mulus beralih menampilkan frame AI beranotasi lengkap. Disediakan tombol eksplisit "🛑 Hentikan Inferensi AI (Kembali ke Siaran CCTV)" untuk kembali ke mode live player kapan saja.

### FR-10: Analisis Komparasi Tri-Lokasi Fasilitas Transportasi
- **Deskripsi:** Menyediakan visualisasi perbandingan karakteristik lalu lintas antara tiga tipologi fasilitas: Simpang Sebidang (Sudirman), Jembatan Layang (Flyover MBK), dan Terowongan Bawah Tanah (Underpass Unila).
- **Kriteria Penerimaan:** Menyajikan metrik KPI side-by-side 3 kolom, grafik perbandingan distribusi Golongan I s.d. VI, tingkat kepatuhan helm, dan matriks ilmiah rekayasa transportasi.

### FR-11: Fitur Pemfilteran Data Telemetri Berdasarkan Rentang Tanggal
- **Deskripsi:** Pengguna dapat memfilter dataset telemetri dan insiden berdasarkan tanggal awal (*start date*) dan tanggal akhir (*end date*).
- **Kriteria Penerimaan:** Dasbor menyediakan date range picker interaktif; visualisasi tabel dan tombol unduh CSV terfilter beradaptasi secara dinamis.

### FR-12: Fitur Pencadangan dan Pengosongan Data (Reset Data Safe-Guard)
- **Deskripsi:** Sistem menyediakan mekanisme reset log aktif dengan pengamanan konfirmasi dua langkah (*checkbox confirmation*).
- **Kriteria Penerimaan:** Saat reset dieksekusi, berkas aktif disalin otomatis ke direktori `logs/archive/` dengan penamaan cap waktu ISO, kemudian berkas log aktif dikosongkan (hanya menyisakan baris header).

---

## 4. KEBUTUHAN NON-FUNGSIONAL (*NON-FUNCTIONAL REQUIREMENTS*)

| ID | Parameter Kebutuhan | Spesifikasi Target |
| :--- | :--- | :--- |
| **NFR-01** | **Kecepatan Inferensi (Throughput)** | Minimal 12.0 – 18.0 Frame Per Detik (FPS) pada CPU standar x86_64 / ARM64. |
| **NFR-02** | **Waktu Pemuatan Awal (Cold Load)** | Antarmuka web harus muncul di peramban dalam waktu $< 1.0\text{ detik}$ (*Fast Initial Paint*). |
| **NFR-03** | **Responsivitas Server (Latency)** | Median latensi HTTP respons server $< 10.0\text{ ms}$ pada beban normal. |
| **NFR-04** | **Stabilitas Koneksi Streaming** | Dilengkapi mekanisme pemulihan otomatis (*auto-recovery*) jika terjadi *drop-packet* pada server CCTV Dishub. |
| **NFR-05** | **Beban I/O Penyimpanan** | Tidak melakukan penulisan berkas berulang pada Google Drive CloudStorage; operasi penulisan berkas sementara dialokasikan di SSD lokal `/tmp`. |
| **NFR-06** | **Akurasi Deteksi Helm** | Precision $\ge 85\%$ pada jarak visual efektif kamera ($10 - 45\text{ meter}$). |
| **NFR-07** | **Integritas Data CSV** | Skema kolom standar IEEE/ISO dengan format tanggal ISO 8601 dan angka floating-point presisi 2 desimal. |
| **NFR-08** | **Desain Antarmuka (Usability)** | Mengadopsi tema gelap modern (*Command Center Dark Theme*) dengan kontras tinggi untuk kenyamanan visual operator 24/7. |

---

## 5. MATRIKS KETERLACAKAN KEBUTUHAN (*TRACEABILITY MATRIX*)

| Kebutuhan Fungsional (FR) | Modul Pelaksana | Berkas Sumber Kode | Status Uji |
| :--- | :--- | :--- | :---: |
| **FR-01: HLS Ingestion** | `LiveHLSStreamReader`, `CCTVStreamManager` | `cctv_stream.py` | Passed |
| **FR-02: Road ROI** | `VehicleDetector.is_in_roi` | `detector.py` | Passed |
| **FR-03: Tracking** | `VehicleDetector.detect_and_track` | `detector.py` | Passed |
| **FR-04: Golongan I-VI** | `classify_indonesian_golongan` | `detector.py` | Passed |
| **FR-05: Deteksi Helm** | `detect_helmet` | `detector.py` | Passed |
| **FR-06: Speed Homografi** | `SpeedEstimator` | `speed_estimator.py` | Passed |
| **FR-07: Anomaly Engine** | `SmartAnomalyDetector` (Barrier-Aware) | `anomaly_detector.py` | Passed |
| **FR-08: Data Logging** | `TrafficDataLogger` | `data_logger.py` | Passed |
| **FR-09: Unified Camera UI**| `run_live_stream_hls`, `camera_area` | `app.py` | Passed |
| **FR-10: Komparasi Lokasi** | `render_comparison_tab` | `app.py` | Passed |
| **FR-11: Filter Rentang Tanggal** | `filter_df_by_date`, date picker UI | `app.py` | Passed |
| **FR-12: Reset & Archive Data** | `reset_logs`, `reset_system_data` | `data_logger.py`, `app.py` | Passed |
