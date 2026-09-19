"""
data_logger.py
Modul pencatatan telemetri lalu lintas dan log insiden kecelakaan secara real-time ke format CSV
dengan dukungan Penggolongan Kendaraan Indonesia (Golongan I s.d. Golongan VI).
"""

import os
import csv
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd


class TrafficDataLogger:
    """Mengelola pencatatan data deteksi kendaraan dan insiden ke file CSV secara real-time."""

    def __init__(self, log_dir: str = "logs", auto_flush_interval: int = 10):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

        self.telemetry_csv_path = os.path.join(self.log_dir, "traffic_telemetry.csv")
        self.incident_csv_path = os.path.join(self.log_dir, "incident_records.csv")

        self.auto_flush_interval = auto_flush_interval
        self._telemetry_buffer: List[Dict] = []
        self._incident_buffer: List[Dict] = []
        self._counter = 0

        self._init_csv_headers()

    def _init_csv_headers(self, force: bool = False):
        """Membuat file CSV dengan header kolom terstandarisasi jika file belum ada atau jika dipaksa (force reset)."""
        telemetry_headers = [
            "timestamp_epoch",
            "datetime_iso",
            "location_id",
            "location_name",
            "frame_id",
            "track_id",
            "golongan",
            "vehicle_subclass",
            "confidence",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "pixel_center_x",
            "pixel_center_y",
            "ground_x_meter",
            "ground_y_meter",
            "length_meter",
            "width_meter",
            "speed_kmh",
            "aspect_ratio",
            "helmet_status",
            "compliance_status",
            "status"
        ]

        incident_headers = [
            "incident_id",
            "timestamp_epoch",
            "datetime_iso",
            "location_id",
            "location_name",
            "incident_type",
            "severity",
            "involved_track_ids",
            "golongan_types",
            "speed_kmh",
            "location_description",
            "title",
            "description"
        ]

        need_reinit_telemetry = force or not os.path.exists(self.telemetry_csv_path)
        if not need_reinit_telemetry:
            with open(self.telemetry_csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "location_id" not in first_line:
                    need_reinit_telemetry = True

        if need_reinit_telemetry:
            with open(self.telemetry_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(telemetry_headers)

        need_reinit_incident = force or not os.path.exists(self.incident_csv_path)
        if not need_reinit_incident:
            with open(self.incident_csv_path, "r", encoding="utf-8") as f:
                first_line = f.readline()
                if "location_id" not in first_line:
                    need_reinit_incident = True

        if need_reinit_incident:
            with open(self.incident_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(incident_headers)

    def log_vehicle_telemetry(
        self,
        frame_id: int,
        timestamp_epoch: float,
        track_id: int,
        golongan: str,
        vehicle_subclass: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        pixel_center: Tuple[float, float],
        ground_coords: Tuple[float, float],
        dimensions_meter: Tuple[float, float],
        speed_kmh: float,
        aspect_ratio: float,
        location_id: int = 312,
        location_name: str = "Underpass Unila",
        helmet_status: str = "N/A",
        compliance_status: str = "NORMAL",
        status: str = "NORMAL"
    ):
        """Merekam baris data telemetri kendaraan per frame."""
        dt_str = datetime.fromtimestamp(timestamp_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        x1, y1, x2, y2 = bbox
        cx, cy = pixel_center
        gx, gy = ground_coords
        l_m, w_m = dimensions_meter

        row = {
            "timestamp_epoch": round(timestamp_epoch, 3),
            "datetime_iso": dt_str,
            "location_id": location_id,
            "location_name": location_name,
            "frame_id": frame_id,
            "track_id": track_id,
            "golongan": golongan,
            "vehicle_subclass": vehicle_subclass,
            "confidence": round(confidence, 3),
            "bbox_x1": int(x1),
            "bbox_y1": int(y1),
            "bbox_x2": int(x2),
            "bbox_y2": int(y2),
            "pixel_center_x": round(cx, 1),
            "pixel_center_y": round(cy, 1),
            "ground_x_meter": round(gx, 2),
            "ground_y_meter": round(gy, 2),
            "length_meter": round(l_m, 2),
            "width_meter": round(w_m, 2),
            "speed_kmh": round(speed_kmh, 1),
            "aspect_ratio": round(aspect_ratio, 2),
            "helmet_status": helmet_status,
            "compliance_status": compliance_status,
            "status": status
        }
        self._telemetry_buffer.append(row)
        self._counter += 1

        if len(self._telemetry_buffer) >= self.auto_flush_interval:
            self.flush()

    def log_incident(
        self,
        incident_id: str,
        timestamp_epoch: float,
        incident_type: str,
        severity: str,
        involved_track_ids: List[int],
        golongan_types: List[str],
        speed_kmh: float,
        location_desc: str,
        title: str,
        description: str,
        location_id: int = 312,
        location_name: str = "Underpass Unila",
    ):
        """Merekam insiden / anomali yang terdeteksi ke log insiden."""
        dt_str = datetime.fromtimestamp(timestamp_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = {
            "incident_id": incident_id,
            "timestamp_epoch": round(timestamp_epoch, 3),
            "datetime_iso": dt_str,
            "location_id": location_id,
            "location_name": location_name,
            "incident_type": incident_type,
            "severity": severity,
            "involved_track_ids": ";".join(map(str, involved_track_ids)),
            "golongan_types": ";".join(golongan_types),
            "speed_kmh": round(speed_kmh, 1),
            "location_description": location_desc,
            "title": title,
            "description": description
        }
        self._incident_buffer.append(row)
        self.flush_incidents()

    def flush(self):
        """Menulis buffer data telemetri ke file CSV."""
        if not self._telemetry_buffer:
            return

        with open(self.telemetry_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in self._telemetry_buffer:
                writer.writerow([
                    r["timestamp_epoch"],
                    r["datetime_iso"],
                    r["location_id"],
                    r["location_name"],
                    r["frame_id"],
                    r["track_id"],
                    r["golongan"],
                    r["vehicle_subclass"],
                    r["confidence"],
                    r["bbox_x1"],
                    r["bbox_y1"],
                    r["bbox_x2"],
                    r["bbox_y2"],
                    r["pixel_center_x"],
                    r["pixel_center_y"],
                    r["ground_x_meter"],
                    r["ground_y_meter"],
                    r["length_meter"],
                    r["width_meter"],
                    r["speed_kmh"],
                    r["aspect_ratio"],
                    r["helmet_status"],
                    r["compliance_status"],
                    r["status"]
                ])
        self._telemetry_buffer.clear()

    def flush_incidents(self):
        """Menulis buffer insiden ke file CSV."""
        if not self._incident_buffer:
            return

        with open(self.incident_csv_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for r in self._incident_buffer:
                writer.writerow([
                    r["incident_id"],
                    r["timestamp_epoch"],
                    r["datetime_iso"],
                    r["location_id"],
                    r["location_name"],
                    r["incident_type"],
                    r["severity"],
                    r["involved_track_ids"],
                    r["golongan_types"],
                    r["speed_kmh"],
                    r["location_description"],
                    r["title"],
                    r["description"]
                ])
        self._incident_buffer.clear()

    def get_telemetry_dataframe(self) -> pd.DataFrame:
        """Mengambil data telemetri sebagai DataFrame pandas."""
        self.flush()
        if os.path.exists(self.telemetry_csv_path):
            try:
                return pd.read_csv(self.telemetry_csv_path)
            except Exception:
                pass
        return pd.DataFrame()

    def get_incident_dataframe(self) -> pd.DataFrame:
        """Mengambil data insiden sebagai DataFrame pandas."""
        self.flush_incidents()
        if os.path.exists(self.incident_csv_path):
            try:
                return pd.read_csv(self.incident_csv_path)
            except Exception:
                pass
        return pd.DataFrame()

    def reset_logs(self, backup: bool = True) -> Tuple[bool, str]:
        """
        Mereset seluruh data telemetri dan catatan insiden.
        Jika backup=True, membuat salinan cadangan berkas CSV ke direktori logs/archive/.
        Menginisialisasi ulang berkas CSV hanya dengan baris header.
        """
        import shutil
        self.flush()
        self.flush_incidents()

        backup_info = ""
        if backup:
            archive_dir = os.path.join(self.log_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_files = []
            if os.path.exists(self.telemetry_csv_path) and os.path.getsize(self.telemetry_csv_path) > 100:
                dest_t = os.path.join(archive_dir, f"traffic_telemetry_{ts_str}.csv")
                shutil.copy2(self.telemetry_csv_path, dest_t)
                saved_files.append(f"traffic_telemetry_{ts_str}.csv")
            if os.path.exists(self.incident_csv_path) and os.path.getsize(self.incident_csv_path) > 100:
                dest_i = os.path.join(archive_dir, f"incident_records_{ts_str}.csv")
                shutil.copy2(self.incident_csv_path, dest_i)
                saved_files.append(f"incident_records_{ts_str}.csv")
            if saved_files:
                backup_info = f"Salinan arsip tersimpan di logs/archive/ ({', '.join(saved_files)})"

        self._telemetry_buffer.clear()
        self._incident_buffer.clear()
        self._init_csv_headers(force=True)
        return True, backup_info


def filter_df_by_date(df: pd.DataFrame, start_d, end_d, col: str = "datetime_iso") -> pd.DataFrame:
    """Menyaring DataFrame berdasarkan rentang tanggal inklusif [start_d, end_d]."""
    if df is None or df.empty or col not in df.columns:
        return df if df is not None else pd.DataFrame()
    dt_series = pd.to_datetime(df[col], errors="coerce")
    valid_mask = dt_series.notna()
    d_series = dt_series.dt.date
    mask = valid_mask & (d_series >= start_d) & (d_series <= end_d)
    return df[mask]


def get_dataset_date_bounds(df_t: pd.DataFrame, df_i: pd.DataFrame):
    """Menghitung batas tanggal paling awal dan paling akhir dari data CSV telemetri & insiden."""
    from datetime import date, timedelta
    dates = []
    if df_t is not None and not df_t.empty and "datetime_iso" in df_t.columns:
        dt_s = pd.to_datetime(df_t["datetime_iso"], errors="coerce").dropna()
        if not dt_s.empty:
            dates.extend([dt_s.min().date(), dt_s.max().date()])
    if df_i is not None and not df_i.empty and "datetime_iso" in df_i.columns:
        dt_s = pd.to_datetime(df_i["datetime_iso"], errors="coerce").dropna()
        if not dt_s.empty:
            dates.extend([dt_s.min().date(), dt_s.max().date()])
    today = date.today()
    if not dates:
        return today - timedelta(days=7), today
    return min(dates), max(dates)


