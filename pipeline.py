"""
pipeline.py
Pipeline integrasi utama: Deteksi Kendaraan (Golongan I - VI), Estimasi Kecepatan, Deteksi Anomali,
dan Pencatatan Telemetri Real-Time ke File CSV untuk Riset Ilmiah.
"""

import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np

from detector import VehicleDetector, GOLONGAN_COLORS
from speed_estimator import SpeedEstimator
from anomaly_detector import SmartAnomalyDetector
from data_logger import TrafficDataLogger


class UnderpassAnalyticsPipeline:
    """Mengintegrasikan seluruh pipeline analitik video CCTV dan pencatatan dataset CSV."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.35,
        fps: float = 25.0,
        stop_threshold_seconds: float = 3.0,
        log_dir: str = "logs",
        enable_logging: bool = True,
        location_id: int = 282,
        location_name: str = "LIVE - PEREMPATAN JENDRAL SUDIRMAN",
    ):
        self.location_id = location_id
        self.location_name = location_name
        loc_name_lower = location_name.lower()
        if location_id in (190, 280) or "mbk" in loc_name_lower or "kedaton" in loc_name_lower or "boemi" in loc_name_lower:
            loc_key = "mbk"
        elif location_id == 282 or "sudirman" in loc_name_lower:
            loc_key = "sudirman"
        else:
            loc_key = "unila"
        self.location_key = loc_key

        self.detector = VehicleDetector(model_path=model_path, conf_threshold=conf_threshold, location=loc_key)
        self.speed_estimator = SpeedEstimator(fps=fps, location=loc_key)
        self.anomaly_detector = SmartAnomalyDetector(
            fps=fps,
            stop_threshold_seconds=stop_threshold_seconds,
            location=loc_key,
        )
        self.fps = fps
        self.frame_count = 0
        self.enable_logging = enable_logging
        self.logger = TrafficDataLogger(log_dir=log_dir) if enable_logging else None

        # Riwayat kecepatan untuk statistik rata-rata
        self.speed_samples = []

    def process_frame(
        self,
        frame: np.ndarray,
        current_time: Optional[float] = None,
        draw_overlays: bool = True,
    ) -> Tuple[np.ndarray, List[Dict], List[Dict], Dict]:
        """
        Memproses satu frame video secara komprehensif:
        1. Deteksi, pelacakan & penggolongan kendaraan (Golongan I - VI)
        2. Estimasi kecepatan per kendaraan (Homografi)
        3. Deteksi anomali & kecelakaan
        4. Pencatatan telemetri kendaraan & insiden ke format CSV
        5. Render visualisasi overlay
        """
        self.frame_count += 1
        now = current_time if current_time is not None else time.time()

        # 1. Deteksi, pelacakan, dan klasifikasi ke Golongan I - VI
        detections = self.detector.detect_and_track(frame, speed_estimator=self.speed_estimator)

        # 2. Estimasi kecepatan untuk tiap kendaraan
        speeds = {}
        active_ids = set()
        for det in detections:
            tid = det["track_id"]
            if tid > 0:
                active_ids.add(tid)
                spd = self.speed_estimator.update(tid, det["bottom_center"], now)
                speeds[tid] = spd
                det["speed_kmh"] = spd
                if spd > 5.0:
                    self.speed_samples.append(spd)
            else:
                det["speed_kmh"] = 0.0

        # Bersihkan cache track ID lama
        self.speed_estimator.cleanup(active_ids)

        # 3. Deteksi anomali dan insiden
        new_alerts = self.anomaly_detector.update(detections, speeds, frame, now)
        active_alerts = self.anomaly_detector.get_active_alerts()

        # 4. Pencatatan data telemetri ke CSV secara real-time
        if self.enable_logging and self.logger is not None:
            alert_tid_map = {}
            for alt in active_alerts:
                for tid in alt.get("track_ids", []):
                    alert_tid_map[tid] = alt["type"]

            for det in detections:
                tid = det["track_id"]
                spd = speeds.get(tid, 0.0)
                bx, by = det["bottom_center"]
                gx, gy = self.speed_estimator.pixel_to_meter(bx, by)
                status_str = alert_tid_map.get(tid, "NORMAL")

                # Kepatuhan helm
                gol = det["golongan"]
                if gol == "Golongan VI-A":
                    helmet_str = "HELM"
                    compliance_str = "TAAT_HUKUM"
                elif gol == "Golongan VI-B":
                    helmet_str = "TANPA_HELM"
                    compliance_str = "MELANGGAR"
                else:
                    helmet_str = "N/A"
                    compliance_str = "NORMAL"

                self.logger.log_vehicle_telemetry(
                    frame_id=self.frame_count,
                    timestamp_epoch=now,
                    track_id=tid,
                    golongan=gol,
                    vehicle_subclass=det["class_name"],
                    confidence=det["conf"],
                    bbox=det["bbox"],
                    pixel_center=det["center"],
                    ground_coords=(gx, gy),
                    dimensions_meter=(det["length_meter"], det["width_meter"]),
                    speed_kmh=spd,
                    aspect_ratio=det["aspect_ratio"],
                    location_id=self.location_id,
                    location_name=self.location_name,
                    helmet_status=helmet_str,
                    compliance_status=compliance_str,
                    status=status_str,
                )

            for alt in new_alerts:
                g_types = [
                    d["golongan"] for d in detections if d["track_id"] in alt["track_ids"]
                ]
                max_spd = max([speeds.get(tid, 0.0) for tid in alt["track_ids"]], default=0.0)
                self.logger.log_incident(
                    incident_id=alt["id"],
                    timestamp_epoch=alt["timestamp"],
                    incident_type=alt["type"],
                    severity=alt["severity"],
                    involved_track_ids=alt["track_ids"],
                    golongan_types=g_types,
                    speed_kmh=max_spd,
                    location_desc=self.location_name,
                    title=alt["title"],
                    description=alt["description"],
                    location_id=self.location_id,
                    location_name=self.location_name,
                )

        # 5. Gambar visualisasi overlay pada frame
        annotated_frame = frame.copy()
        if draw_overlays:
            annotated_frame = self._render_annotations(annotated_frame, detections, speeds, active_alerts)

        # 6. Hitung statistik dinamis
        avg_speed = float(np.mean(self.speed_samples[-50:])) if self.speed_samples else 0.0
        max_speed = float(np.max(self.speed_samples)) if self.speed_samples else 0.0

        stats = {
            "total_frames": self.frame_count,
            "counts_by_golongan": self.detector.get_counts(),
            "total_vehicles": len(self.detector.counted_ids),
            "current_active_vehicles": len(detections),
            "avg_speed_kmh": round(avg_speed, 1),
            "max_speed_kmh": round(max_speed, 1),
            "active_alerts_count": len(active_alerts),
            "total_incidents": len(self.anomaly_detector.get_incident_log()),
        }

        return annotated_frame, detections, new_alerts, stats

    def flush_logs(self):
        """Memaksa penulisan semua data buffer ke file CSV di disk."""
        if self.logger:
            self.logger.flush()
            self.logger.flush_incidents()

    def _render_annotations(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        speeds: Dict[int, float],
        active_alerts: List[Dict],
    ) -> np.ndarray:
        """Menggambar bounding box kendaraan dengan kode Golongan dan label kecepatan."""
        h, w = frame.shape[:2]

        # 1. Gambar batas area ROI jalan (garis tipis hijau transparan)
        overlay = frame.copy()
        cv2.polylines(overlay, [self.detector.road_roi], isClosed=True, color=(0, 220, 0), thickness=2)
        if self.location_key == "unila":
            # Garis Barrier Median Pembatas Dua Arah
            cv2.line(overlay, (110, 0), (175, h), (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(overlay, "[BARRIER PEMBATAS DUA ARAH]", (130, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        loc_banner = f"AREA PANTAUAN {self.location_name.upper()} (GOLONGAN I-VI)"
        cv2.putText(overlay, loc_banner, (180, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 2. Gambar bounding box dan label tiap kendaraan
        for det in detections:
            tid = det["track_id"]
            golongan = det["golongan"]
            subclass = det["class_name"]
            x1, y1, x2, y2 = det["bbox"]
            spd = speeds.get(tid, 0.0)

            # Warna berdasarkan Golongan
            color = GOLONGAN_COLORS.get(golongan, (0, 215, 255))

            # Merah jika sedang terlibat insiden
            is_in_alert = any(tid in a.get("track_ids", []) for a in active_alerts)
            if is_in_alert:
                color = (0, 0, 255)

            # Gambar bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label teks: "Gol I: Sedan #1 | 42 km/h" atau "Gol VI-B: Motor #1 [TANPA HELM!] | 38 km/h"
            id_str = f"#{tid}" if tid > 0 else ""
            if golongan == "Golongan VI-A":
                badge_str = "[HELM: YA]"
            elif golongan == "Golongan VI-B":
                badge_str = "[MELANGGAR: TANPA HELM]"
            else:
                badge_str = ""

            label = f"{golongan} {id_str} {badge_str} | {spd:.0f} km/h".strip()

            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), color, -1)
            text_color = (255, 255, 255) if (is_in_alert or golongan == "Golongan VI-B") else (0, 0, 0)
            cv2.putText(frame, label, (x1 + 4, max(14, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, text_color, 2)

            # Titik kontak roda jalan
            cx, cy = det["bottom_center"]
            cv2.circle(frame, (int(cx), int(cy)), 4, color, -1)

        # 3. Banner alert insiden jika ada alert aktif
        if active_alerts:
            top_alert = active_alerts[-1]
            banner_color = (0, 0, 220) if top_alert["severity"] == "CRITICAL" else (0, 140, 255)

            cv2.rectangle(frame, (0, 0), (w, 45), banner_color, -1)
            alert_text = f"PERINGATAN: [{top_alert['type']}] {top_alert['title']}"
            cv2.putText(frame, alert_text, (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        return frame
