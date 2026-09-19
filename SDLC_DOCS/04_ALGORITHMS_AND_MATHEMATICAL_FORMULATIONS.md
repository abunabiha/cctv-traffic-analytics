# DOKUMEN TAHAPAN SDLC 04: FORMULASI MATEMATIS DAN ALGORITMA (ALGORITHMS AND MATHEMATICAL FORMULATIONS)

**Topik:** Pemodelan Matematis, Algoritma Computer Vision, dan Persamaan Inti Sistem Analitik Cerdas CCTV (Tri-Lokasi)  
**Versi:** 3.0 (Tri-Site Urban Corridor Edition)  
**Standar Keilmuan:** IEEE Standards / Peer-Reviewed Intelligent Transportation Systems (ITS) Formulations  

---

## 1. PENDAHULUAN & NOTASI MATEMATIS

Sistem analitik cerdas CCTV ini mengombinasikan model pembelajaran mendalam (*Deep Learning*), aljabar linier geometri proyektif, teori estimasi stokastik, dan logika heuristik temporal. Dokumen ini merumuskan seluruh persamaan matematis dan pseudocode algoritma yang menjadi landasan operasional sistem.

### Tabel Notasi Simbolik:
| Simbol | Makna Matematis |
| :--- | :--- |
| $\mathbf{I} \in \mathbb{R}^{H \times W \times 3}$ | Matriks citra frame video dengan tinggi $H=720$ dan lebar $W=1280$ piksel |
| $\mathcal{B} = (x_1, y_1, x_2, y_2)$ | Koordinat sudut kotak pembatas (*bounding box*) pada bidang citra |
| $(u, v)$ | Koordinat piksel 2D pada bidang proyeksi kamera |
| $(X, Y)$ | Koordinat metrik dunia nyata 2D pada bidang jalan (*ground plane*) dalam satuan meter |
| $\mathbf{H} \in \mathbb{R}^{3 \times 3}$ | Matriks Transformasi Homografi Planar |
| $\text{IoU}$ | *Intersection over Union* (derajat tumpang-tindih dua kotak pembatas) |
| $v(t)$ | Kecepatan kendaraan terestimasi pada saat $t$ dalam satuan km/jam |
| $x_{\text{barrier}}(y)$ | Koordinat absis batas pemisah median barrier fisik sebagai fungsi ordinat $y$ |

---

## 2. ARSITEKTUR DETEKSI DEEP LEARNING (YOLOv8 NANO)

### 2.1 Mekanisme Anchor-Free & TaskAlignedAssigner
Berbeda dengan arsitektur deteksi berjangkar (*anchor-based*), YOLOv8 memprediksi langsung jarak relatif dari titik pusat sel fitur ke 4 sisi batas kotak pembatas:
$$\mathcal{B} = \left( x_c - l \cdot s, \; y_c - t \cdot s, \; x_c + r \cdot s, \; y_c + b \cdot s \right)$$
di mana $(x_c, y_c)$ adalah titik pusat sel kisi (*grid cell*), $s$ adalah faktor langkah *stride* piramida fitur ($s \in \{8, 16, 32\}$), dan $(l, t, r, b)$ adalah offset jarak ke sisi kiri, atas, kanan, dan bawah.

### 2.2 Fungsi Kerugian Gabungan (Multi-Task Loss Function)
Pelatihan model mengoptimalkan tiga komponen kerugian secara simultan:
$$\mathcal{L}_{\text{total}} = \lambda_{\text{box}} \mathcal{L}_{\text{CIoU}} + \lambda_{\text{cls}} \mathcal{L}_{\text{BCE}} + \lambda_{\text{dfl}} \mathcal{L}_{\text{DFL}}$$

1. **Complete IoU Loss ($\mathcal{L}_{\text{CIoU}}$):**
   $$\mathcal{L}_{\text{CIoU}} = 1 - \text{IoU} + \frac{\rho^2(\mathbf{b}, \mathbf{b}^{gt})}{c^2} + \alpha v$$
   di mana $\rho(\mathbf{b}, \mathbf{b}^{gt})$ adalah jarak Euclidean antara titik pusat kotak prediksi dan kotak acuan (*ground truth*), $c$ adalah panjang diagonal kotak penutup terkecil, dan $v$ mengukur konsistensi rasio aspek:
   $$v = \frac{4}{\pi^2} \left( \arctan\frac{w^{gt}}{h^{gt}} - \arctan\frac{w}{h} \right)^2, \quad \alpha = \frac{v}{(1 - \text{IoU}) + v}$$

2. **Distribution Focal Loss ($\mathcal{L}_{\text{DFL}}$):**
   $$\mathcal{L}_{\text{DFL}}(S_i, S_{i+1}) = - \left( (y_{i+1} - y)\log(S_i) + (y - y_i)\log(S_{i+1}) \right)$$
   Memungkinkan model mempelajari representasi distribusi probabilitas regresi batas kotak di sekitar nilai kontinu target $y$.

3. **Binary Cross-Entropy Loss ($\mathcal{L}_{\text{BCE}}$):**
   $$\mathcal{L}_{\text{BCE}} = - \sum_{k=1}^{C} \left[ y_k \log(\hat{p}_k) + (1 - y_k)\log(1 - \hat{p}_k) \right]$$

---

## 3. PELACAKAN MULTI-OBJEK (*BYTETRACKING MATHEMATICAL MODEL*)

### 3.1 Model Keadaan Kalman Filter (State Space Representation)
Vektor keadaan gerak kendaraan dimodelkan dalam ruang 8-dimensi:
$$\mathbf{x} = \left[ x_c, \; y_c, \; s, \; r, \; \dot{x}_c, \; \dot{y}_c, \; \dot{s}, \; \dot{r} \right]^T$$
di mana $(x_c, y_c)$ adalah titik tengah kotak pembatas, $s = w \times h$ adalah skala area kotak, $r = \frac{w}{h}$ adalah rasio aspek, dan komponen bertitik merepresentasikan turunan pertama terhadap waktu (kecepatan gerak piksel).

**Persamaan Prediksi Keadaan:**
$$\mathbf{x}_{k|k-1} = \mathbf{F} \mathbf{x}_{k-1|k-1} + \mathbf{w}_k, \quad \mathbf{P}_{k|k-1} = \mathbf{F} \mathbf{P}_{k-1|k-1} \mathbf{F}^T + \mathbf{Q}$$

**Persamaan Pembaruan Pengukuran:**
$$\mathbf{K}_k = \mathbf{P}_{k|k-1} \mathbf{H}^T \left( \mathbf{H} \mathbf{P}_{k|k-1} \mathbf{H}^T + \mathbf{R} \right)^{-1}$$
$$\mathbf{x}_{k|k} = \mathbf{x}_{k|k-1} + \mathbf{K}_k \left( \mathbf{z}_k - \mathbf{H} \mathbf{x}_{k|k-1} \right)$$
$$\mathbf{P}_{k|k} = (\mathbf{I} - \mathbf{K}_k \mathbf{H}) \mathbf{P}_{k|k-1}$$

### 3.2 Asosiasi Dua Tahap (Two-Stage Hungarian Matching)
- **Tahap 1:** Mencocokkan deteksi berkeyakinan tinggi ($conf \ge 0.50$) dengan lintasan aktif menggunakan metrik jarak biaya $1 - \text{IoU}$.
- **Tahap 2:** Mencocokkan deteksi berkeyakinan rendah ($0.15 \le conf < 0.50$) dengan sisa lintasan yang belum terhubung, mencegah hilangnya objek saat tertutup sebagian (*occlusion*).

---

## 4. TRANSFORMASI PERSPEKTIF HOMOGRAFI (BIRD'S-EYE VIEW)

### 4.1 Persamaan Pemetaan Koordinat Planar
Transformasi proyektif dari bidang citra kamera $(u, v)$ ke bidang jalan riil $(X, Y)$ dalam meter diformulasikan melalui koordinat homogen:
$$\begin{bmatrix} X' \\ Y' \\ W \end{bmatrix} = \mathbf{H} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Koordinat Kartesius fisik diperoleh melalui de-homogenisasi:
$$X = \frac{X'}{W} = \frac{h_{11}u + h_{12}v + h_{13}}{h_{31}u + h_{32}v + h_{33}}, \quad Y = \frac{Y'}{W} = \frac{h_{21}u + h_{22}v + h_{23}}{h_{31}u + h_{32}v + h_{33}}$$

### 4.2 Kalibrasi Titik Kontrol Tanah Tri-Lokasi (Ground Control Points - GCP)
Matriks $\mathbf{H}$ diselesaikan menggunakan metode *Direct Linear Transformation (DLT)* berbasis 4 titik acuan kalibrasi lapangan:

#### A. Konfigurasi 1: CCTV Perempatan Jendral Sudirman (CCTV ID 282) [Default / Simpang Sebidang]
Dikalibrasi pada area persimpangan 4 lajur selebar $12.0\text{ meter}$ dan panjang bentang $28.0\text{ meter}$:
- Titik 1 (Kiri Atas): $(u_1, v_1) = (380, 190) \longrightarrow (X_1, Y_1) = (0.0, 28.0)\text{ meter}$
- Titik 2 (Kanan Atas): $(u_2, v_2) = (760, 190) \longrightarrow (X_2, Y_2) = (12.0, 28.0)\text{ meter}$
- Titik 3 (Kanan Bawah): $(u_3, v_3) = (800, 710) \longrightarrow (X_3, Y_3) = (12.0, 0.0)\text{ meter}$
- Titik 4 (Kiri Bawah): $(u_4, v_4) = (120, 710) \longrightarrow (X_4, Y_4) = (0.0, 0.0)\text{ meter}$

#### B. Konfigurasi 2 (Baru): CCTV Flyover Mall Boemi Kedaton (CCTV ID 190) [Jembatan Layang Menerus]
Dikalibrasi pada bentang jembatan layang selebar $10.0\text{ meter}$ (2 lajur @ 5.0m) dan panjang bentang $30.0\text{ meter}$:
- Titik 1 (Kiri Atas): $(u_1, v_1) = (410, 160) \longrightarrow (X_1, Y_1) = (0.0, 30.0)\text{ meter}$
- Titik 2 (Kanan Atas): $(u_2, v_2) = (860, 160) \longrightarrow (X_2, Y_2) = (10.0, 30.0)\text{ meter}$
- Titik 3 (Kanan Bawah): $(u_3, v_3) = (930, 710) \longrightarrow (X_3, Y_3) = (10.0, 0.0)\text{ meter}$
- Titik 4 (Kiri Bawah): $(u_4, v_4) = (180, 710) \longrightarrow (X_4, Y_4) = (0.0, 0.0)\text{ meter}$

#### C. Konfigurasi 3: UNDERPASS UNILA ARAH RAJABASA (CCTV ID 312) [Terowongan Bawah Tanah]
Dikalibrasi pada area terowongan dua lajur selebar $7.5\text{ meter}$ dan panjang bentang $45.0\text{ meter}$:
- Titik 1 (Kiri Atas): $(u_1, v_1) = (240, 150) \longrightarrow (X_1, Y_1) = (0.0, 45.0)\text{ meter}$
- Titik 2 (Kanan Atas): $(u_2, v_2) = (410, 150) \longrightarrow (X_2, Y_2) = (7.5, 45.0)\text{ meter}$
- Titik 3 (Kanan Bawah): $(u_3, v_3) = (650, 680) \longrightarrow (X_3, Y_3) = (7.5, 0.0)\text{ meter}$
- Titik 4 (Kiri Bawah): $(u_4, v_4) = (180, 680) \longrightarrow (X_4, Y_4) = (0.0, 0.0)\text{ meter}$

### 4.3 Estimasi Dimensi Fisik Kendaraan
Panjang ($L$) dan lebar ($W$) kendaraan dalam satuan meter dihitung dari 4 koordinat sudut bounding box yang ditransformasikan via $\mathbf{H}$:
$$L = \left\| \mathbf{H}(u_{\text{center}}, v_2) - \mathbf{H}(u_{\text{center}}, v_1) \right\|_2$$
$$W = \left\| \mathbf{H}(u_2, v_2) - \mathbf{H}(u_1, v_2) \right\|_2$$

---

## 5. ESTIMASI KECEPATAN & PENGHALUS FILTER EMA

### 5.1 Kecepatan Sesaat (Instantaneous Velocity)
Perpindahan posisi fisik kendaraan antara frame $t-1$ dan frame $t$ dihitung menggunakan jarak Euclidean:
$$\Delta d_t = \sqrt{\left( X_t - X_{t-1} \right)^2 + \left( Y_t - Y_{t-1} \right)^2} \quad (\text{meter})$$
Kecepatan translasi sesaat dalam km/jam dirumuskan sebagai:
$$v_{\text{inst}}(t) = \frac{\Delta d_t}{\Delta t} \times 3.6 \quad \text{di mana } \Delta t = t_{\text{curr}} - t_{\text{prev}}$$

### 5.2 Exponential Moving Average (EMA) Smoothing
Untuk mengeliminasi getaran mikro (*jitter*) pada deteksi kotak pembatas, digunakan filter eksponensial:
$$v_{\text{smooth}}(t) = \alpha \cdot v_{\text{inst}}(t) + (1 - \alpha) \cdot v_{\text{smooth}}(t-1)$$
dengan faktor kehalusan $\alpha = 0.35$.

---

## 6. LOGIKA DETEKSI ANOMALI TRI-LOKASI FASILITAS

### 6.1 Deteksi Anomali pada Simpang Sudirman (CCTV ID 282 - Default/Simpang Sebidang)
Pada simpang bersinyal pusat kota, mesin anomali mengawasi dinamika konflik manuver dan antrean:
1. **Deteksi Kemacetan Persimpangan (*Intersection Gridlock*):** Jumlah kendaraan berhenti ($v < 2.0\text{ km/jam}$) di dalam kotak tengah simpang melebihi densitas ambang $\rho_{\text{gridlock}} \ge 6\text{ kendaraan}$ selama durasi $> 5.0\text{ detik}$.
2. **Tabrakan Sudut Silang (*Crossing Angle Collision*):** Vektor gerak dua kendaraan yang saling berpotongan $\vec{v}_1 \cdot \vec{v}_2 < 0$ disertai tumpang-tindih $\text{IoU} \ge 0.40$ dan perlambatan seketika.

### 6.2 Deteksi Anomali pada Flyover Mall Boemi Kedaton (CCTV ID 190 - Baru / Jembatan Layang Menerus)
Pada fasilitas jalan layang bebas hambatan, anomali diprioritaskan untuk bahaya kecepatan tinggi dan hambatan mendadak:
1. **Kendaraan Mogok di Bentang Layang (*Elevated Free-Flow Obstruction*):** Kendaraan terhenti ($v < 3.0\text{ km/jam}$) di tengah badan jembatan layang selama $\Delta t \ge 3.0\text{ detik}$. Pada kecepatan operasional 50 km/jam, rintangan mendadak di atas flyover berpotensi fatal menimbulkan tabrakan beruntun (*pile-up crash*).
2. **Sepeda Motor Tergelincir pada Siar Muai (*Expansion Joint Spill*):** Deteksi inversi rasio aspek ($\text{AR} > 1.30$) sepeda motor yang melintasi sambungan siar muai baja (*steel expansion joint*) jembatan saat kondisi basah atau licin.
3. **Pelanggaran Kecepatan Turunan (*Downhill Overspeeding*):** Kecepatan melebihi ambang batas aman flyover perkotaan ($v > 65.0\text{ km/jam}$) pada zona turunan bentang menuju Jl. Z.A. Pagar Alam.

### 6.3 Formulasi Garis Pemisah Median Barrier Underpass Unila (CCTV ID 312 - Terowongan Bawah Tanah)
Underpass Unila dipisahkan oleh struktur median fisik permanen. Pada bidang citra kamera beresolusi $1280 \times 720$, batas pemisah median dimodelkan sebagai persamaan linier:
$$x_{\text{barrier}}(y) = 110.0 + 65.0 \times \left(\frac{y}{720.0}\right)$$

```
     y=0 (Puncak Atas) ───>  x_barrier(0) = 110.0 px
     y=360 (Tengah)    ───>  x_barrier(360) = 142.5 px
     y=720 (Dasar Bawah) ──>  x_barrier(720) = 175.0 px
```

### 6.4 Aturan Verifikasi Lawan Arah Sesuai Lajur Underpass
Diberikan titik kontak roda kendaraan pada saat ini $(x_t, y_t)$ dan titik kontak $k$-frame sebelumnya $(x_{t-k}, y_{t-k})$ dengan $\Delta y = y_t - y_{t-k}$:

$$\text{Anomali Lawan Arah} = 
\begin{cases} 
\text{TRUE}, & \text{jika } x_t > x_{\text{barrier}}(y_t) \text{ dan } \Delta y < -25.0\text{ px} \quad (\text{Lajur Kanan Melaju Naik}) \\
\text{TRUE}, & \text{jika } x_t \le x_{\text{barrier}}(y_t) \text{ dan } \Delta y > +25.0\text{ px} \quad (\text{Lajur Kiri Melaju Turun}) \\
\text{TRUE}, & \text{jika } \text{sgn}\left(x_t - x_{\text{barrier}}(y_t)\right) \neq \text{sgn}\left(x_{t-k} - x_{\text{barrier}}(y_{t-k})\right) \quad (\text{Menyeberang Barrier}) \\
\text{FALSE}, & \text{lainnya (Kendaraan di Jalur yang Sah)}
\end{cases}$$

**Implikasi:** Kendaraan di lajur kiri yang melaju naik ke arah Rajabasa ($\Delta y < 0$) dan kendaraan di lajur kanan yang melaju turun ke terowongan ($\Delta y > 0$) **keduanya berstatus NORMAL / SAH**.

### 6.5 Aturan Tabrakan & Deselerasi Ekstrem (Collision)
$$\text{Collision}(A, B) = \left( \text{IoU}(\mathcal{B}_A, \mathcal{B}_B) \ge 0.45 \right) \;\wedge\; \left( v_A(t) < 5.0\text{ km/h} \;\vee\; v_B(t) < 5.0\text{ km/h} \right) \;\wedge\; \left( |\Delta v| \ge 20.0\text{ km/h} \right)$$

### 6.6 Aturan Sepeda Motor Terjatuh (Spill Anomaly)
Berdasarkan sifat biomekanika kendaraan roda dua, sepeda motor yang tegak memiliki tinggi bounding box yang lebih besar daripada lebarnya ($h > w \implies \text{AR} < 0.85$). Ketika terjatuh, terjadi pembalikan sumbu geometris (*spatial aspect ratio flip*):
$$\text{Motorcycle Fallen} = (\text{class} = \text{motorcycle}) \;\wedge\; \left( \text{AR} = \frac{w}{h} > 1.30 \right) \;\wedge\; (v < 8.0\text{ km/h})$$

---

## 7. ALGORITMA COMPUTER VISION DETEKSI HELM (ETLE GOLONGAN VI)

Untuk mengklasifikasikan sepeda motor ke dalam **Golongan VI-A (Taat Helm)** atau **Golongan VI-B (Melanggar / Tanpa Helm)**, dievaluasi algoritma segmentasi *multi-feature*:

```
Algoritma: Deteksi Kepatuhan Helm Pengendara
Input    : Citra Bounding Box Motor B = (x1, y1, x2, y2), Frame Asli I
Output   : (has_helmet: Boolean, confidence: Float)

1. Ekstraksi ROI Kepala:
   y_head_start = max(0, y1)
   y_head_end   = min(H, y1 + 0.32 * (y2 - y1))
   x_head_start = max(0, x1 + 0.15 * (x2 - x1))
   x_head_end   = min(W, x2 - 0.15 * (x2 - x1))
   ROI_head     = I[y_head_start : y_head_end, x_head_start : x_head_end]

2. Masking Warna Kulit Ganda (Dual Color-Space Skin Segmentation):
   Mask_HSV   = inRange(HSV(ROI_head), [0, 25, 40], [25, 180, 255])
   Mask_YCbCr = inRange(YCrCb(ROI_head), [0, 133, 77], [255, 173, 127])
   Mask_Skin  = bitwise_and(Mask_HSV, Mask_YCbCr)
   Skin_Ratio = countNonZero(Mask_Skin) / Area(ROI_head)

3. Analisis Bentuk Geometri (Convexity & Circularity):
   Contours   = findContours(Canny(ROI_head, 50, 150))
   Max_Contour= argmax(Area(Contours))
   Circularity= 4 * pi * Area(Max_Contour) / (Perimeter(Max_Contour)^2)

4. Evaluasi Keputusan:
   IF Skin_Ratio > 0.32 THEN
       has_helmet = FALSE
       confidence = min(0.95, Skin_Ratio * 2.2)
   ELSE IF Circularity > 0.65 THEN
       has_helmet = TRUE
       confidence = 0.88
   ELSE
       has_helmet = TRUE  (Prior Taat Konservatif)
       confidence = 0.70

5. Temporal Majority Voting Buffer (Ukuran Jendela N = 15 frame):
   Helmet_Decision = (Sum(History[track_id]) / len(History[track_id])) >= 0.50
   RETURN (Helmet_Decision, confidence)
```
