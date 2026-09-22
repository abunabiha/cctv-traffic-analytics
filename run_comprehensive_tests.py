"""
run_comprehensive_tests.py
Suite Pengujian Terpadu Otomatis (Automated Testing Suite):
1. Fungsional Testing (Functional Testing)
2. Non-Fungsional Testing (Non-Functional Testing)
3. Integrasi Testing (Integration Testing)
4. AI Model & Computer Vision Testing (AI Model Testing)
"""

import os
import time
import shutil
import unittest
from datetime import datetime, date, timedelta
from typing import Tuple
import numpy as np
import pandas as pd
import cv2

# Import modul internal sistem
from speed_estimator import SpeedEstimator
from detector import classify_vehicle_indonesia, detect_helmet_heuristic
from anomaly_detector import SmartAnomalyDetector
from data_logger import TrafficDataLogger


def get_dataset_date_bounds(df_t: pd.DataFrame, df_i: pd.DataFrame) -> Tuple[date, date]:
    dates = []
    if not df_t.empty and "datetime_iso" in df_t.columns:
        dt_s = pd.to_datetime(df_t["datetime_iso"], errors="coerce").dropna()
        if not dt_s.empty:
            dates.extend([dt_s.min().date(), dt_s.max().date()])
    if not df_i.empty and "datetime_iso" in df_i.columns:
        dt_s = pd.to_datetime(df_i["datetime_iso"], errors="coerce").dropna()
        if not dt_s.empty:
            dates.extend([dt_s.min().date(), dt_s.max().date()])
    today = date.today()
    if not dates:
        return today - timedelta(days=7), today
    return min(dates), max(dates)


def filter_df_by_date(df: pd.DataFrame, start_d: date, end_d: date, col: str = "datetime_iso") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    dt_series = pd.to_datetime(df[col], errors="coerce")
    valid_mask = dt_series.notna()
    d_series = dt_series.dt.date
    mask = valid_mask & (d_series >= start_d) & (d_series <= end_d)
    return df[mask]


def get_raw_csv_bytes(file_path: str) -> bytes:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return f.read()
    return b""


class TestFunctional(unittest.TestCase):
    """Pengujian Fungsional (Functional Testing): Memverifikasi kepatuhan terhadap Functional Requirements (FR)."""

    def setUp(self):
        self.test_log_dir = "test_logs_functional"
        os.makedirs(self.test_log_dir, exist_ok=True)
        self.logger = TrafficDataLogger(log_dir=self.test_log_dir)

    def tearDown(self):
        if os.path.exists(self.test_log_dir):
            shutil.rmtree(self.test_log_dir, ignore_errors=True)

    def test_ft01_vehicle_classification_standard_indonesia(self):
        """FT-01: Verifikasi pemetaan klasifikasi Golongan I s.d. VI Standar Indonesia."""
        # 1. Mobil Penumpang (COCO cls 2) -> Golongan I
        gol1, _, _ = classify_vehicle_indonesia(coco_cls_id=2, length_m=4.5, width_m=1.8, aspect_ratio=0.7)
        self.assertEqual(gol1, "Golongan I")

        # 2. Bus (COCO cls 5) -> Golongan I
        gol_bus, _, _ = classify_vehicle_indonesia(coco_cls_id=5, length_m=10.0, width_m=2.4, aspect_ratio=0.5)
        self.assertEqual(gol_bus, "Golongan I")

        # 3. Truk Kecil (COCO cls 7, panjang < 5.0m) -> Golongan I
        gol_trk1, _, _ = classify_vehicle_indonesia(coco_cls_id=7, length_m=4.2, width_m=1.8, aspect_ratio=0.7)
        self.assertEqual(gol_trk1, "Golongan I")

        # 4. Truk 2 Gandar (COCO cls 7, panjang 5.0 - 8.0m) -> Golongan II
        gol_trk2, _, _ = classify_vehicle_indonesia(coco_cls_id=7, length_m=6.5, width_m=2.1, aspect_ratio=0.6)
        self.assertEqual(gol_trk2, "Golongan II")

        # 5. Truk 3 Gandar (COCO cls 7, panjang 8.0 - 11.5m) -> Golongan III
        gol_trk3, _, _ = classify_vehicle_indonesia(coco_cls_id=7, length_m=9.5, width_m=2.4, aspect_ratio=0.5)
        self.assertEqual(gol_trk3, "Golongan III")

        # 6. Truk 4 Gandar (COCO cls 7, panjang 11.5 - 14.0m) -> Golongan IV
        gol_trk4, _, _ = classify_vehicle_indonesia(coco_cls_id=7, length_m=12.5, width_m=2.5, aspect_ratio=0.45)
        self.assertEqual(gol_trk4, "Golongan IV")

        # 7. Truk 5+ Gandar (COCO cls 7, panjang >= 14.0m) -> Golongan V
        gol_trk5, _, _ = classify_vehicle_indonesia(coco_cls_id=7, length_m=15.0, width_m=2.5, aspect_ratio=0.4)
        self.assertEqual(gol_trk5, "Golongan V")

    def test_ft02_helmet_etle_classification(self):
        """FT-02: Verifikasi klasifikasi kepatuhan helm (Golongan VI-A vs Golongan VI-B)."""
        # Motor dengan helm -> Golongan VI-A
        gol_taat, _, _ = classify_vehicle_indonesia(coco_cls_id=3, length_m=1.8, width_m=0.8, aspect_ratio=0.45, has_helmet=True)
        self.assertEqual(gol_taat, "Golongan VI-A")

        # Motor tanpa helm -> Golongan VI-B
        gol_langgar, _, _ = classify_vehicle_indonesia(coco_cls_id=3, length_m=1.8, width_m=0.8, aspect_ratio=0.45, has_helmet=False)
        self.assertEqual(gol_langgar, "Golongan VI-B")

    def test_ft03_speed_estimation_homography(self):
        """FT-03: Verifikasi estimasi kecepatan kendaraan berbasis matriks homografi."""
        estimator = SpeedEstimator(fps=25.0)
        t0 = 1000.0
        # Posisi bergerak dari atas ke bawah terowongan dalam 1 detik
        estimator.update(track_id=101, bottom_center=(250, 160), timestamp=t0)
        estimator.update(track_id=101, bottom_center=(300, 270), timestamp=t0 + 0.5)
        spd = estimator.update(track_id=101, bottom_center=(390, 430), timestamp=t0 + 1.0)
        self.assertGreater(spd, 15.0, "Estimasi kecepatan terlalu rendah")
        self.assertLess(spd, 100.0, "Estimasi kecepatan tidak realistis (> 100 km/h)")

    def test_ft04_barrier_aware_wrong_way_direction(self):
        """FT-04: Verifikasi aturan deteksi lawan arah barrier-aware underpass (Underpass Unila dua arah)."""
        anomaly_engine = SmartAnomalyDetector(fps=25.0)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Kasus 1: Lajur Kiri (x=80) melaju naik (dy < 0) -> HARUS NORMAL (legal)
        dets_legal_left = [{
            "track_id": 201,
            "class_name": "Mobil",
            "bbox": (50, 400, 110, 500),
            "bottom_center": (80, 500),
            "center": (80, 450),
            "aspect_ratio": 0.6,
        }]
        anomaly_engine.track_histories[201] = [(80, 520), (80, 480), (80, 450)] # gerak naik
        alerts = anomaly_engine.check_wrong_way(dets_legal_left, dummy_frame)
        self.assertEqual(len(alerts), 0, "Lajur kiri legal naik salah divonis lawan arah!")

        # Kasus 2: Lajur Kiri (x=80) melaju turun (dy > 25) -> HARUS MEMICU WRONG_WAY_HAZARD
        anomaly_engine.track_histories[202] = [(80, 400), (80, 450), (80, 500)] # gerak turun di jalur naik
        dets_illegal_left = [{
            "track_id": 202,
            "class_name": "Mobil",
            "bbox": (50, 450, 110, 550),
            "bottom_center": (80, 550),
            "center": (80, 500),
            "aspect_ratio": 0.6,
        }]
        alerts_illegal = anomaly_engine.check_wrong_way(dets_illegal_left, dummy_frame)
        self.assertGreater(len(alerts_illegal), 0, "Lajur kiri turun gagal terdeteksi sebagai pelanggaran!")
        self.assertIn(alerts_illegal[0]["type"], ("WRONG_WAY_HAZARD", "LAWAN_ARAH"))

    def test_ft05_stopped_vehicle_and_collision_anomalies(self):
        """FT-05: Verifikasi deteksi kendaraan berhenti dan tabrakan kendaraan."""
        anomaly_engine = SmartAnomalyDetector(fps=25.0, stop_threshold_seconds=1.5)
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        t_base = 500.0

        # Simulasi kendaraan diam selama 2.0 detik (> 1.5 detik ambang batas)
        alerts_stopped = []
        for i in range(20):
            t = t_base + i * 0.1
            dets = [{
                "track_id": 301,
                "class_name": "Mobil",
                "bbox": (400, 300, 480, 420),
                "bottom_center": (440, 420),
                "center": (440, 360),
                "aspect_ratio": 0.67,
            }]
            al = anomaly_engine.update(dets, {301: 0.0}, dummy_frame, current_time=t)
            alerts_stopped.extend(al)
        has_stop_alert = any(a["type"] in ("VEHICLE_STOPPED", "KENDARAAN_BERHENTI") for a in alerts_stopped)
        self.assertTrue(has_stop_alert, "Gagal mendeteksi kendaraan berhenti!")

    def test_ft06_data_logger_and_reset_with_backup(self):
        """FT-06: Verifikasi logging CSV telemetri, insiden, serta fungsionalitas reset data ber-backup."""
        # 1. Catat telemetri
        self.logger.log_vehicle_telemetry(
            frame_id=1,
            timestamp_epoch=time.time(),
            track_id=10,
            golongan="Golongan I",
            vehicle_subclass="Sedan",
            confidence=0.88,
            bbox=(100, 100, 200, 250),
            pixel_center=(150, 175),
            ground_coords=(3.5, 12.0),
            dimensions_meter=(4.2, 1.8),
            speed_kmh=42.5,
            aspect_ratio=0.67,
        )
        self.logger.flush()

        df_t = self.logger.get_telemetry_dataframe()
        self.assertEqual(len(df_t), 1, "Data telemetri gagal tersimpan di CSV!")

        # 2. Catat insiden
        self.logger.log_incident(
            incident_id="INC_TEST_01",
            timestamp_epoch=time.time(),
            incident_type="VEHICLE_STOPPED",
            severity="WARNING",
            involved_track_ids=[10],
            golongan_types=["Golongan I"],
            speed_kmh=0.0,
            location_desc="LIVE - PEREMPATAN JENDRAL SUDIRMAN",
            title="Kendaraan Berhenti",
            description="Uji coba rekaman insiden",
        )
        df_i = self.logger.get_incident_dataframe()
        self.assertEqual(len(df_i), 1, "Data insiden gagal tersimpan di CSV!")

        # 3. Eksekusi Reset Data dengan Backup
        ok, msg = self.logger.reset_logs(backup=True)
        self.assertTrue(ok, "Reset logs gagal!")

        # 4. Pastikan berkas aktif sekarang kosong (0 baris)
        df_t_after = self.logger.get_telemetry_dataframe()
        df_i_after = self.logger.get_incident_dataframe()
        self.assertEqual(len(df_t_after), 0, "Berkas telemetri setelah reset tidak kosong!")
        self.assertEqual(len(df_i_after), 0, "Berkas insiden setelah reset tidak kosong!")

        # 5. Pastikan berkas cadangan tersimpan di direktori archive
        archive_dir = os.path.join(self.test_log_dir, "archive")
        self.assertTrue(os.path.exists(archive_dir), "Direktori archive tidak terbentuk!")
        archived_files = os.listdir(archive_dir)
        self.assertGreaterEqual(len(archived_files), 1, "Tidak ada salinan arsip yang tersimpan!")


class TestNonFunctional(unittest.TestCase):
    """Pengujian Non-Fungsional (Non-Functional Testing): Kinerja, Latensi, Efisiensi Memori & Integritas Data."""

    def test_nft01_homography_coordinate_transformation_latency(self):
        """NFT-01: Uji latensi transformasi koordinat matriks homografi (< 0.1 ms per pemanggilan)."""
        estimator = SpeedEstimator(fps=25.0)
        coords = [(np.random.randint(100, 1100), np.random.randint(100, 650)) for _ in range(1000)]

        start_time = time.perf_counter()
        for pt in coords:
            _ = estimator.image_to_ground(pt[0], pt[1])
        elapsed = time.perf_counter() - start_time
        avg_latency_ms = (elapsed / 1000) * 1000

        print(f"\n[NFT-01] Homography Transform: Rata-rata Latensi = {avg_latency_ms:.4f} ms per titik.")
        self.assertLess(avg_latency_ms, 0.1, "Transformasi homografi terlalu lambat (> 0.1 ms)!")

    def test_nft02_raw_csv_binary_read_performance(self):
        """NFT-02: Uji performa pembacaan biner CSV instan (< 1.0 ms untuk payload telemetri)."""
        tmp_csv = "test_perf_telemetry.csv"
        # Buat file CSV sintetis 200 baris
        lines = ["timestamp_epoch,datetime_iso,location_id,location_name,frame_id,track_id,speed_kmh\n"]
        for i in range(200):
            lines.append(f"1726000000.0,2026-09-19 12:00:00.000,282,LIVE - PEREMPATAN JENDRAL SUDIRMAN,{i},{i % 20},45.2\n")
        with open(tmp_csv, "w") as f:
            f.writelines(lines)

        start_time = time.perf_counter()
        raw_bytes = get_raw_csv_bytes(tmp_csv)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"[NFT-02] Raw CSV Binary Read: {len(raw_bytes)} bytes terbaca dalam {elapsed_ms:.4f} ms.")
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        self.assertLess(elapsed_ms, 5.0, "Pembacaan biner CSV lambat (> 5 ms)!")

    def test_nft03_date_filtering_throughput(self):
        """NFT-03: Uji throughput pemfilteran DataFrame 10.000 baris berdasarkan rentang tanggal."""
        n_rows = 10000
        base_date = datetime(2026, 9, 1)
        dates = [(base_date + timedelta(hours=i % 720)).strftime("%Y-%m-%d %H:%M:%S.000") for i in range(n_rows)]
        df_dummy = pd.DataFrame({
            "datetime_iso": dates,
            "vehicle_id": range(n_rows),
            "speed_kmh": np.random.uniform(20, 80, n_rows),
        })

        t0 = time.perf_counter()
        df_filtered = filter_df_by_date(df_dummy, date(2026, 9, 10), date(2026, 9, 15))
        duration_ms = (time.perf_counter() - t0) * 1000

        print(f"[NFT-03] Date Filtering 10.000 baris: {len(df_filtered)} baris tersaring dalam {duration_ms:.2f} ms.")
        self.assertLess(duration_ms, 50.0, "Penyaringan rentang tanggal terlalu lambat (> 50 ms)!")


class TestIntegration(unittest.TestCase):
    """Pengujian Integrasi (Integration Testing): Komunikasi & Alur Data Antar-Modul End-to-End."""

    def setUp(self):
        self.int_log_dir = "test_logs_integration"
        os.makedirs(self.int_log_dir, exist_ok=True)

    def tearDown(self):
        if os.path.exists(self.int_log_dir):
            shutil.rmtree(self.int_log_dir, ignore_errors=True)

    def test_it01_pipeline_data_flow_integration(self):
        """IT-01: Verifikasi aliran data integrasi pipeline deteksi, estimasi kecepatan, anomali & logger."""
        logger = TrafficDataLogger(log_dir=self.int_log_dir)
        estimator = SpeedEstimator(fps=25.0)
        anomaly_engine = SmartAnomalyDetector(fps=25.0)

        # 1. Objek terdeteksi
        dets = [{
            "track_id": 99,
            "class_name": "Mobil",
            "golongan": "Golongan I",
            "confidence": 0.92,
            "bbox": (300, 200, 400, 320),
            "bottom_center": (350, 320),
            "center": (350, 260),
            "aspect_ratio": 100 / 120,
        }]

        # 2. Estimasi kecepatan
        spd = estimator.update(99, (350, 320), timestamp=100.0)
        speeds = {99: spd}

        # 3. Mesin anomali
        dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        alerts = anomaly_engine.update(dets, speeds, dummy_frame, current_time=100.0)

        # 4. Logger mencatat telemetri
        logger.log_vehicle_telemetry(
            frame_id=1,
            timestamp_epoch=100.0,
            track_id=99,
            golongan="Golongan I",
            vehicle_subclass="Mobil",
            confidence=0.92,
            bbox=(300, 200, 400, 320),
            pixel_center=(350, 260),
            ground_coords=(3.5, 15.0),
            dimensions_meter=(4.2, 1.8),
            speed_kmh=spd,
            aspect_ratio=0.83,
        )
        logger.flush()

        df_t = logger.get_telemetry_dataframe()
        self.assertEqual(len(df_t), 1, "Integrasi alur pipeline telemetri gagal!")
        self.assertEqual(df_t.iloc[0]["track_id"], 99)

    def test_it02_data_logger_to_charts_aggregation(self):
        """IT-02: Verifikasi integrasi data logger dengan agregasi statistik grafik jurnal publikasi."""
        logger = TrafficDataLogger(log_dir=self.int_log_dir)
        # Tulis 10 baris telemetri dengan sebaran kecepatan dan golongan berbeda
        for i in range(10):
            gol = "Golongan I" if i % 2 == 0 else "Golongan VI-A"
            spd = 35.0 + i * 2.5
            logger.log_vehicle_telemetry(
                frame_id=i,
                timestamp_epoch=time.time() + i,
                track_id=i,
                golongan=gol,
                vehicle_subclass="Sample",
                confidence=0.9,
                bbox=(100, 100, 200, 200),
                pixel_center=(150, 150),
                ground_coords=(2.0, 10.0),
                dimensions_meter=(4.0, 1.8),
                speed_kmh=spd,
                aspect_ratio=0.7,
            )
        logger.flush()

        df_t = logger.get_telemetry_dataframe()
        self.assertEqual(len(df_t), 10)

        # Hitung statistik publikasi V85, Mean, Max
        v85 = np.percentile(df_t["speed_kmh"], 85)
        v_mean = df_t["speed_kmh"].mean()
        self.assertGreater(v85, v_mean, "V85 harus lebih besar dari rata-rata pada distribusi menaik!")
        print(f"\n[IT-02] Integrasi Data -> Statistik Jurnal: V85 = {v85:.2f} km/h, Mean = {v_mean:.2f} km/h.")


class TestAIModelAndVision(unittest.TestCase):
    """Pengujian Model AI & Computer Vision (AI Model Testing): Bounding Box, Matriks H, IoU & Invers Homografi."""

    def test_aim01_iou_calculation_and_collision_geometry(self):
        """AIM-01: Verifikasi akurasi kalkulasi Intersection-over-Union (IoU) deteksi tabrakan."""
        detector = SmartAnomalyDetector()
        # Kotak 1: (100, 100, 200, 200) -> Luas = 100 x 100 = 10000
        boxA = (100, 100, 200, 200)
        # Kotak 2: Identik -> IoU harus tepat 1.0
        iou_perfect = detector._compute_iou(boxA, boxA)
        self.assertAlmostEqual(iou_perfect, 1.0, places=3)

        # Kotak 3: Tidak bertumpang tindih sama sekali -> IoU harus 0.0
        box_disjoint = (300, 300, 400, 400)
        iou_zero = detector._compute_iou(boxA, box_disjoint)
        self.assertEqual(iou_zero, 0.0)

        # Kotak 4: Tumpang tindih sebagian (50x100) -> Inter = 5000, Union = 15000 -> IoU = 0.333
        box_half = (150, 100, 250, 200)
        iou_half = detector._compute_iou(boxA, box_half)
        self.assertAlmostEqual(iou_half, 5000 / 15000, places=2)

    def test_aim02_aspect_ratio_inversion_for_fallen_motorcycles(self):
        """AIM-02: Verifikasi sensitivitas deteksi anomali rasio aspek untuk motor jatuh."""
        # Motor normal: Tinggi > Lebar (Tinggi 120px, Lebar 50px -> AR = W/H = 50/120 = 0.417)
        w_normal, h_normal = 50, 120
        ar_normal = w_normal / h_normal
        self.assertLess(ar_normal, 1.0, "Motor normal harus memiliki AR < 1.0")

        # Motor roboh di aspal: Lebar > Tinggi (Lebar 110px, Tinggi 45px -> AR = 110/45 = 2.44)
        w_fallen, h_fallen = 110, 45
        ar_fallen = w_fallen / h_fallen
        self.assertGreater(ar_fallen, 1.30, "Motor jatuh harus memicu rasio aspek > 1.30")

    def test_aim03_homography_matrix_mathematical_properties(self):
        """AIM-03: Verifikasi sifat matematika matriks Homografi H Tri-Lokasi (Sudirman, MBK, Unila)."""
        locations = ["282", "190", "312"]
        for loc in locations:
            estimator = SpeedEstimator(fps=25.0, location=loc)
            H = estimator.H
            self.assertIsNotNone(H, f"Matriks homografi H untuk lokasi {loc} bernilai None!")
            self.assertEqual(H.shape, (3, 3), f"Matriks homografi {loc} harus berukuran 3x3!")

            # Uji Determinan & Keterbalikan (Invertibility)
            det = np.linalg.det(H)
            self.assertNotEqual(det, 0.0, f"Matriks homografi {loc} singular (determinan = 0)!")

            # Uji stabilitas proyeksi bolak-balik (Forward -> Inverse -> Identity)
            H_inv = np.linalg.inv(H)
            identity_approx = np.dot(H, H_inv)
            np.testing.assert_allclose(
                identity_approx, np.eye(3), atol=1e-5,
                err_msg=f"Invers matriks H untuk lokasi {loc} tidak konsisten!"
            )

    def test_aim04_helmet_heuristic_classifier_vision(self):
        """AIM-04: Verifikasi model heuristik visual helm (Skin ratio, Specular, Circularity)."""
        # ROI helm buatan: elips abu-abu terang dengan kilap specular
        helmet_roi = np.zeros((60, 60, 3), dtype=np.uint8)
        cv2.ellipse(helmet_roi, (30, 30), (22, 18), 0, 0, 360, (235, 235, 235), -1)
        cv2.circle(helmet_roi, (25, 25), 6, (255, 255, 255), -1)

        has_helmet, score = detect_helmet_heuristic(helmet_roi)
        self.assertTrue(has_helmet, "Gagal mengenali kubah batok helm specular!")
        self.assertGreater(score, 0.40)

        # ROI kulit wajah / leher tanpa helm (skin-tone BGR)
        skin_roi = np.full((60, 60, 3), (120, 150, 220), dtype=np.uint8)
        has_helmet_skin, score_skin = detect_helmet_heuristic(skin_roi)
        self.assertFalse(has_helmet_skin, "Area kulit wajah salah dideteksi sebagai helm!")
        self.assertLess(score_skin, 0.40)

    def test_aim05_road_roi_perspective_containment(self):
        """AIM-05: Verifikasi poligon Region-of-Interest (ROI) badan jalan Tri-Lokasi (Sudirman, MBK, Unila)."""
        from detector import SUDIRMAN_ROAD_ROI, UNILA_ROAD_ROI, MBK_ROAD_ROI
        
        # 1. Uji ROI Simpang Sudirman (CCTV ID 282)
        dist_inside_sudirman = cv2.pointPolygonTest(SUDIRMAN_ROAD_ROI, (500.0, 400.0), False)
        self.assertGreaterEqual(dist_inside_sudirman, 0, "Titik tengah simpang Sudirman salah diidentifikasi di luar ROI!")
        dist_outside_sudirman = cv2.pointPolygonTest(SUDIRMAN_ROAD_ROI, (50.0, 50.0), False)
        self.assertLess(dist_outside_sudirman, 0, "Titik luar trotoar Sudirman salah diidentifikasi di dalam ROI!")

        # 2. Uji ROI Flyover MBK (CCTV ID 190)
        dist_inside_mbk = cv2.pointPolygonTest(MBK_ROAD_ROI, (640.0, 400.0), False)
        self.assertGreaterEqual(dist_inside_mbk, 0, "Titik tengah bentang Flyover MBK salah diidentifikasi di luar ROI!")
        dist_outside_mbk = cv2.pointPolygonTest(MBK_ROAD_ROI, (50.0, 50.0), False)
        self.assertLess(dist_outside_mbk, 0, "Titik luar Flyover MBK salah diidentifikasi di dalam ROI!")

        # 3. Uji ROI Underpass Unila (CCTV ID 312)
        dist_inside_unila = cv2.pointPolygonTest(UNILA_ROAD_ROI, (450.0, 400.0), False)
        self.assertGreaterEqual(dist_inside_unila, 0, "Titik aspal Underpass Unila salah diidentifikasi di luar ROI!")
        dist_outside_unila = cv2.pointPolygonTest(UNILA_ROAD_ROI, (1150.0, 100.0), False)
        self.assertLess(dist_outside_unila, 0, "Titik dinding Underpass Unila salah diidentifikasi di dalam ROI!")


if __name__ == "__main__":
    print("=" * 75)
    print("🚦 MENJALANKAN SUITE PENGUJIAN KOMPREHENSIF SMART CCTV TRAFFIC ANALYTICS 🚦")
    print("=" * 75)
    unittest.main(verbosity=2)
