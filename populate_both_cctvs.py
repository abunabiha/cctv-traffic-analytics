import cv2
import pandas as pd
from pipeline import UnderpassAnalyticsPipeline
from data_logger import TrafficDataLogger

print("=== Memproses Sampel Underpass Unila (Frame 30 - 110) ===")
pipe_unila = UnderpassAnalyticsPipeline(
    model_path="yolov8n.pt",
    conf_threshold=0.20,
    enable_logging=True,
    location_id=312,
    location_name="Underpass Unila Arah Rajabasa"
)
cap_u = cv2.VideoCapture("underpass_live_sample.mp4")
f_idx = 0
while cap_u.isOpened() and f_idx < 110:
    ret, frame = cap_u.read()
    if not ret:
        break
    f_idx += 1
    if f_idx >= 30:
        pipe_unila.process_frame(frame)
cap_u.release()
pipe_unila.flush_logs()
print("Underpass Unila counts:", pipe_unila.detector.get_counts())

print("\n=== Memproses Sampel Perempatan Sudirman (Frame 1 - 120) ===")
pipe_s = UnderpassAnalyticsPipeline(
    model_path="yolov8n.pt",
    conf_threshold=0.20,
    enable_logging=True,
    location_id=282,
    location_name="Perempatan Jl. Jendral Sudirman"
)
cap_s = cv2.VideoCapture("sudirman_live_sample.mp4")
f_idx = 0
while cap_s.isOpened() and f_idx < 120:
    ret, frame = cap_s.read()
    if not ret:
        break
    f_idx += 1
    pipe_s.process_frame(frame)
cap_s.release()
pipe_s.flush_logs()
print("Sudirman counts:", pipe_s.detector.get_counts())

logger = TrafficDataLogger(log_dir="logs")
df_t = logger.get_telemetry_dataframe()
print(f"\nTotal telemetri terekam: {len(df_t)}")
print(pd.crosstab(df_t["location_name"], df_t["golongan"]))
