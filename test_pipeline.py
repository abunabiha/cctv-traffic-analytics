"""
test_pipeline.py
Skrip pengujian unit dan verifikasi menyeluruh untuk seluruh modul:
1. SpeedEstimator (Homography & Speed calculation)
2. VehicleDetector (YOLO & ROI filtering)
3. SmartAnomalyDetector (Kecelakaan, Motor Jatuh, Kendaraan Berhenti, Lawan Arah)
4. UnderpassAnalyticsPipeline (Integrasi end-to-end)
"""

import time
import numpy as np
import cv2
from speed_estimator import SpeedEstimator
from detector import VehicleDetector, DEFAULT_ROAD_ROI
from anomaly_detector import SmartAnomalyDetector
from pipeline import UnderpassAnalyticsPipeline


def test_speed_estimator():
    print("\n=== [1] Menguji Modul SpeedEstimator ===")
    estimator = SpeedEstimator(fps=25.0)

    # Titik awal mobil di atas underpass (px: 240, 150)
    # Setelah 1 detik (25 frame), mobil melaju ke bawah (px: 380, 420)
    t0 = 100.0
    s1 = estimator.update(track_id=1, bottom_center=(240, 150), timestamp=t0)
    s2 = estimator.update(track_id=1, bottom_center=(290, 260), timestamp=t0 + 0.4)
    s3 = estimator.update(track_id=1, bottom_center=(380, 420), timestamp=t0 + 1.0)

    print(f"  Estimasi kecepatan awal: {s1} km/h")
    print(f"  Estimasi kecepatan pertengahan: {s2} km/h")
    print(f"  Estimasi kecepatan akhir: {s3} km/h")
    assert s3 > 15.0 and s3 < 100.0, f"Kecepatan tidak wajar: {s3} km/h"
    print("  -> SpeedEstimator LOLOS verifikasi! Nilai berada di kisaran wajar.")


def test_anomaly_detection_rules():
    print("\n=== [2] Menguji Mesin Deteksi Anomali & Kecelakaan ===")
    detector = SmartAnomalyDetector(fps=25.0, stop_threshold_seconds=2.0)
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    # --- Skenario A: Kendaraan Berhenti di Underpass ---
    print("  -> Skenario A: Uji Kendaraan Berhenti...")
    t_start = 100.0
    alerts_stopped = []
    for step in range(25):  # 25 iterasi x 0.1 detik = 2.5 detik
        t_now = t_start + step * 0.1
        dets = [{
            "track_id": 10,
            "class_name": "Mobil",
            "bbox": (400, 400, 500, 520),
            "bottom_center": (450, 520),
            "center": (450, 460),
            "aspect_ratio": 100 / 120,
        }]
        speeds = {10: 0.0}  # Mobil terhenti
        new_al = detector.update(dets, speeds, dummy_frame, current_time=t_now)
        alerts_stopped.extend(new_al)

    assert len(alerts_stopped) > 0, "Gagal mendeteksi kendaraan berhenti!"
    print(f"  [BERHASIL] Alert Terdeteksi: {alerts_stopped[0]['title']}")

    # --- Skenario B: Kendaraan Lawan Arah ---
    print("  -> Skenario B: Uji Kendaraan Lawan Arah (Mundur / Berlawanan)...")
    detector_ww = SmartAnomalyDetector(fps=25.0)
    alerts_ww = []
    t_start = 200.0
    for step in range(10):
        t_now = t_start + step * 0.1
        # Objek bergerak ke atas (y mengecil: 500 -> 350)
        y_pos = 500 - step * 15
        dets = [{
            "track_id": 20,
            "class_name": "Mobil",
            "bbox": (350, y_pos, 450, y_pos + 80),
            "bottom_center": (400, y_pos + 80),
            "center": (400, y_pos + 40),
            "aspect_ratio": 100 / 80,
        }]
        speeds = {20: 30.0}
        new_al = detector_ww.update(dets, speeds, dummy_frame, current_time=t_now)
        alerts_ww.extend(new_al)

    assert len(alerts_ww) > 0, "Gagal mendeteksi kendaraan lawan arah!"
    print(f"  [BERHASIL] Alert Terdeteksi: {alerts_ww[0]['title']}")

    # --- Skenario C: Sepeda Motor Terjatuh ---
    print("  -> Skenario C: Uji Sepeda Motor Terjatuh...")
    detector_fall = SmartAnomalyDetector(fps=25.0)
    alerts_fall = []
    t_start = 300.0
    # Tahap 1: Motor melaju normal (w < h, ar = 0.5, speed = 40 km/h)
    for step in range(5):
        t_now = t_start + step * 0.1
        dets = [{
            "track_id": 30,
            "class_name": "Sepeda Motor",
            "bbox": (400, 300 + step * 10, 440, 380 + step * 10),
            "bottom_center": (420, 380 + step * 10),
            "center": (420, 340 + step * 10),
            "aspect_ratio": 40 / 80,  # 0.5 (tegak)
        }]
        speeds = {30: 40.0}
        detector_fall.update(dets, speeds, dummy_frame, current_time=t_now)

    # Tahap 2: Motor terjatuh di jalan (w > h, ar = 1.6, speed = 0 km/h)
    for step in range(5):
        t_now = t_start + 0.5 + step * 0.1
        dets = [{
            "track_id": 30,
            "class_name": "Sepeda Motor",
            "bbox": (400, 350, 490, 400),
            "bottom_center": (445, 400),
            "center": (445, 375),
            "aspect_ratio": 90 / 50,  # 1.8 (rebah)
        }]
        speeds = {30: 0.0}
        new_al = detector_fall.update(dets, speeds, dummy_frame, current_time=t_now)
        alerts_fall.extend(new_al)

    assert len(alerts_fall) > 0, "Gagal mendeteksi sepeda motor terjatuh!"
    print(f"  [BERHASIL] Alert Terdeteksi: {alerts_fall[0]['title']}")

    # --- Skenario D: Tabrakan / Kecelakaan Dua Kendaraan ---
    print("  -> Skenario D: Uji Tabrakan / Crash...")
    detector_crash = SmartAnomalyDetector(fps=25.0)
    alerts_crash = []
    t_start = 400.0

    # Tahap 1: Kedua mobil melaju kencang berdekatan
    for step in range(5):
        t_now = t_start + step * 0.1
        dets = [
            {
                "track_id": 41,
                "class_name": "Mobil",
                "bbox": (380, 300 + step * 10, 460, 380 + step * 10),
                "bottom_center": (420, 380 + step * 10),
                "center": (420, 340 + step * 10),
                "aspect_ratio": 80 / 80,
            },
            {
                "track_id": 42,
                "class_name": "Mobil",
                "bbox": (400, 330 + step * 10, 480, 410 + step * 10),
                "bottom_center": (440, 410 + step * 10),
                "center": (440, 370 + step * 10),
                "aspect_ratio": 80 / 80,
            }
        ]
        speeds = {41: 50.0, 42: 45.0}
        detector_crash.update(dets, speeds, dummy_frame, current_time=t_now)

    # Tahap 2: Benturan & berhenti bersamaan (IoU tinggi, speed nol)
    for step in range(5):
        t_now = t_start + 0.5 + step * 0.1
        dets = [
            {
                "track_id": 41,
                "class_name": "Mobil",
                "bbox": (390, 360, 460, 430),
                "bottom_center": (425, 430),
                "center": (425, 395),
                "aspect_ratio": 70 / 70,
            },
            {
                "track_id": 42,
                "class_name": "Mobil",
                "bbox": (400, 370, 470, 440),
                "bottom_center": (435, 440),
                "center": (435, 405),
                "aspect_ratio": 70 / 70,
            }
        ]
        speeds = {41: 0.0, 42: 0.0}
        new_al = detector_crash.update(dets, speeds, dummy_frame, current_time=t_now)
        alerts_crash.extend(new_al)

    assert len(alerts_crash) > 0, "Gagal mendeteksi tabrakan kendaraan!"
    print(f"  [BERHASIL] Alert Terdeteksi: {alerts_crash[0]['title']}")

    # --- Skenario E: Pelanggaran Pengendara Sepeda Motor Tanpa Helm (ETLE Golongan VI-B) ---
    print("  -> Skenario E: Uji Pelanggaran Helm ETLE (Golongan VI-B)...")
    detector_helm = SmartAnomalyDetector(fps=25.0)
    alerts_helm = []
    t_start = 500.0
    for step in range(5):
        t_now = t_start + step * 0.1
        dets = [{
            "track_id": 55,
            "golongan": "Golongan VI-B",
            "class_name": "Sepeda Motor (Tanpa Helm)",
            "bbox": (350, 200 + step * 10, 390, 270 + step * 10),
            "bottom_center": (370, 270 + step * 10),
            "center": (370, 235 + step * 10),
            "aspect_ratio": 40 / 70,
            "has_helmet": False,
        }]
        speeds = {55: 45.0}
        new_al = detector_helm.update(dets, speeds, dummy_frame, current_time=t_now)
        alerts_helm.extend(new_al)

    assert len(alerts_helm) > 0, "Gagal mendeteksi pelanggaran helm!"
    print(f"  [BERHASIL] Alert Terdeteksi: {alerts_helm[0]['title']}")


def test_pipeline_on_sample_video():
    print("\n=== [3] Menguji Pipeline pada Sampel Video Live CCTV 312 ===")
    pipeline = UnderpassAnalyticsPipeline(model_path="yolov8n.pt", conf_threshold=0.3)
    cap = cv2.VideoCapture("underpass_live_sample.mp4")

    processed = 0
    while processed < 30:  # Uji 30 frame
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, detections, alerts, stats = pipeline.process_frame(frame)
        processed += 1

    cap.release()
    print(f"  Frame terproses: {processed}")
    print(f"  Status pipeline: OK, dimensi output: {annotated_frame.shape}")
    print(f"  Ringkasan statistik: {stats}")
    assert annotated_frame.shape == (720, 1280, 3)
    print("  -> UnderpassAnalyticsPipeline LOLOS pengujian integrasi!")


def test_pipeline_on_sudirman_sample():
    print("\n=== [4] Menguji Pipeline pada Sampel Video CCTV Sudirman 282 ===")
    pipeline = UnderpassAnalyticsPipeline(
        model_path="yolov8n.pt",
        conf_threshold=0.25,
        location_id=282,
        location_name="Perempatan Jl. Jendral Sudirman",
    )
    cap = cv2.VideoCapture("sudirman_live_sample.mp4")
    processed = 0
    while processed < 25:
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, detections, alerts, stats = pipeline.process_frame(frame)
        processed += 1
    cap.release()
    print(f"  Frame terproses: {processed}")
    print(f"  Status pipeline Sudirman: OK, dimensi output: {annotated_frame.shape}")
    print(f"  Ringkasan statistik: {stats}")
    assert annotated_frame.shape == (720, 1280, 3)
    print("  -> CCTV Sudirman Pipeline LOLOS pengujian integrasi!")


if __name__ == "__main__":
    test_speed_estimator()
    test_anomaly_detection_rules()
    test_pipeline_on_sample_video()
    test_pipeline_on_sudirman_sample()
    print("\n🎉 SEMUA PENGUJIAN UNIT & ANOMALI SELESAI DENGAN SUKSES! 🎉\n")
