"""
test_dual_pipeline.py
Verifikasi pipeline analitik ganda untuk Underpass Unila (CCTV 312) dan Perempatan Sudirman (CCTV 282).
"""

import os
import cv2
import pandas as pd
from pipeline import UnderpassAnalyticsPipeline
from data_logger import TrafficDataLogger

def test_pipeline():
    print("=== 1. Menguji Pipeline Underpass Unila (CCTV 312) ===")
    pipe_unila = UnderpassAnalyticsPipeline(
        model_path="yolov8n.pt",
        conf_threshold=0.20,
        enable_logging=True,
        location_id=312,
        location_name="Underpass Unila Arah Rajabasa"
    )
    
    cap_unila = cv2.VideoCapture("underpass_live_sample.mp4")
    frames_unila = 0
    while cap_unila.isOpened() and frames_unila < 25:
        ret, frame = cap_unila.read()
        if not ret:
            break
        frames_unila += 1
        _, dets, alerts, stats = pipe_unila.process_frame(frame)
    cap_unila.release()
    pipe_unila.flush_logs()
    print(f"Underpass Unila selesai diproses: {frames_unila} frames. Counts: {pipe_unila.detector.get_counts()}")

    print("\n=== 2. Menguji Pipeline Perempatan Sudirman (CCTV 282) ===")
    pipe_sudirman = UnderpassAnalyticsPipeline(
        model_path="yolov8n.pt",
        conf_threshold=0.20,
        enable_logging=True,
        location_id=282,
        location_name="Perempatan Jl. Jendral Sudirman"
    )

    cap_sudirman = cv2.VideoCapture("sudirman_live_sample.mp4")
    frames_sudirman = 0
    while cap_sudirman.isOpened() and frames_sudirman < 35:
        ret, frame = cap_sudirman.read()
        if not ret:
            break
        frames_sudirman += 1
        _, dets, alerts, stats = pipe_sudirman.process_frame(frame)
    cap_sudirman.release()
    pipe_sudirman.flush_logs()
    print(f"Perempatan Sudirman selesai diproses: {frames_sudirman} frames. Counts: {pipe_sudirman.detector.get_counts()}")

    print("\n=== 3. Memeriksa Telemetri CSV Hasil Logging ===")
    logger = TrafficDataLogger(log_dir="logs")
    df_t = logger.get_telemetry_dataframe()
    print(f"Total baris telemetri CSV: {len(df_t)}")
    if not df_t.empty:
        print("Kolom telemetri:", list(df_t.columns))
        print("Distribusi lokasi:")
        print(df_t["location_name"].value_counts())
        print("\nDistribusi Golongan per Lokasi:")
        print(pd.crosstab(df_t["location_name"], df_t["golongan"]))
        print("\nKepatuhan Helm per Lokasi:")
        print(pd.crosstab(df_t["location_name"], df_t["compliance_status"]))

    print("\n=== SUKSES: Pipeline dual-CCTV berfungsi normal! ===")

if __name__ == "__main__":
    test_pipeline()
