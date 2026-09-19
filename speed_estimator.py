"""
speed_estimator.py
Modul estimasi kecepatan dan dimensi kendaraan berdasarkan transformasi perspektif (Homography)
pada CCTV Underpass Unila Arah Rajabasa.
"""

from collections import deque
from typing import Dict, Optional, Tuple
import cv2
import numpy as np


# Kalibrasi Homografi Kamera 1: Underpass Unila (ID 312)
UNILA_SRC_POINTS = np.float32([
    [210, 80],   # Atas-Kiri (Pintu masuk turunan underpass)
    [350, 130],  # Atas-Kanan (Dinding batas kanan atas)
    [730, 710],  # Bawah-Kanan (Batas dinding kanan bawah)
    [220, 710],  # Bawah-Kiri (Batas median kiri bawah)
])
UNILA_DST_POINTS = np.float32([
    [0.0, 0.0],
    [7.0, 0.0],
    [7.0, 35.0],
    [0.0, 35.0],
])

# Kalibrasi Homografi Kamera 2: Perempatan Jl. Jendral Sudirman (ID 282)
# Simpang 4 lajur: lebar ~ 12 meter, panjang bidang pantau ~ 28 meter
SUDIRMAN_SRC_POINTS = np.float32([
    [380, 190],  # Atas-Kiri (Bawah flyover lajur kiri)
    [760, 190],  # Atas-Kanan (Bawah flyover lajur kanan)
    [800, 710],  # Bawah-Kanan (Depan kamera sisi kanan)
    [120, 710],  # Bawah-Kiri (Depan kamera sisi kiri)
])
SUDIRMAN_DST_POINTS = np.float32([
    [0.0, 0.0],
    [12.0, 0.0],
    [12.0, 28.0],
    [0.0, 28.0],
])

# Kalibrasi Homografi Kamera 3: CCTV Flyover Mall Boemi Kedaton (ID 190)
# Jembatan Layang (Flyover) 2 lajur terpisah elevasi: lebar ~ 10 meter, panjang bentang pantau ~ 30 meter
MBK_SRC_POINTS = np.float32([
    [410, 160],  # Atas-Kiri (Awal tanjakan / bentang atas flyover)
    [860, 160],  # Atas-Kanan (Batas parapet kanan atas)
    [930, 710],  # Bawah-Kanan (Batas parapet kanan turunan flyover)
    [180, 710],  # Bawah-Kiri (Batas parapet kiri turunan flyover)
])
MBK_DST_POINTS = np.float32([
    [0.0, 0.0],
    [10.0, 0.0],
    [10.0, 30.0],
    [0.0, 30.0],
])


class SpeedEstimator:
    """
    Mengestimasi kecepatan nyata kendaraan (km/jam) dan dimensi fisik (meter)
    dengan memetakan koordinat piksel ke koordinat bidang tanah melalui matriks Homografi.
    Mendukung kalibrasi tri-lokasi:
    1. CCTV Perempatan Jendral Sudirman (CCTV 282)
    2. CCTV Flyover Mall Boemi Kedaton (CCTV 190)
    3. CCTV Unila Arah Rajabasa (CCTV 312)
    """

    def __init__(
        self,
        location: str = "sudirman",
        src_points: Optional[np.ndarray] = None,
        dst_points: Optional[np.ndarray] = None,
        fps: float = 25.0,
        history_len: int = 10,
    ):
        self.location = location.lower()
        self.fps = fps
        self.history_len = history_len

        if src_points is not None:
            self.src_points = np.float32(src_points)
            self.dst_points = np.float32(dst_points)
        elif "mbk" in self.location or "kedaton" in self.location or "190" in self.location or "280" in self.location or "boemi" in self.location:
            self.src_points = MBK_SRC_POINTS
            self.dst_points = MBK_DST_POINTS
        elif "sudirman" in self.location or "282" in self.location:
            self.src_points = SUDIRMAN_SRC_POINTS
            self.dst_points = SUDIRMAN_DST_POINTS
        else:
            self.src_points = UNILA_SRC_POINTS
            self.dst_points = UNILA_DST_POINTS

        # Hitung matriks transformasi homografi
        self.H = cv2.getPerspectiveTransform(self.src_points, self.dst_points)

        # Riwayat pelacakan posisi meter tiap kendaraan: {track_id: deque([(t, x_m, y_m)])}
        self.track_history: Dict[int, deque] = {}
        # Kecepatan terestimasi terakhir yang sudah dihaluskan: {track_id: speed_kmh}
        self.speeds: Dict[int, float] = {}

    def pixel_to_meter(self, u: float, v: float) -> Tuple[float, float]:
        """Mentransformasikan koordinat piksel (u, v) ke bidang tanah (x_m, y_m)."""
        pt = np.array([[[u, v]]], dtype=np.float32)
        warped = cv2.perspectiveTransform(pt, self.H)
        x_m, y_m = warped[0][0]
        return float(x_m), float(y_m)

    # Alias untuk kompatibilitas suite pengujian
    image_to_ground = pixel_to_meter

    def estimate_dimensions_meter(self, bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        """
        Mengestimasi panjang (length) dan lebar (width) kendaraan dalam meter
        berdasarkan proyeksi 4 sudut bounding box ke bidang homografi.
        """
        x1, y1, x2, y2 = bbox
        # Titik dasar roda depan & belakang
        p_front_left = self.pixel_to_meter(x1, y1)
        p_front_right = self.pixel_to_meter(x2, y1)
        p_back_left = self.pixel_to_meter(x1, y2)
        p_back_right = self.pixel_to_meter(x2, y2)

        # Estimasi lebar (jarak transversal)
        w1 = np.hypot(p_front_right[0] - p_front_left[0], p_front_right[1] - p_front_left[1])
        w2 = np.hypot(p_back_right[0] - p_back_left[0], p_back_right[1] - p_back_left[1])
        width_m = (w1 + w2) / 2.0

        # Estimasi panjang (jarak longitudinal)
        l1 = np.hypot(p_back_left[0] - p_front_left[0], p_back_left[1] - p_front_left[1])
        l2 = np.hypot(p_back_right[0] - p_front_right[0], p_back_right[1] - p_front_right[1])
        length_m = (l1 + l2) / 2.0

        # Batasi ke rentang fisik yang masuk akal
        width_m = max(0.5, min(width_m, 3.5))
        length_m = max(1.2, min(length_m, 22.0))
        return round(float(length_m), 2), round(float(width_m), 2)

    def update(self, track_id: int, bottom_center: Tuple[float, float], timestamp: float) -> float:
        """
        Memperbarui posisi objek dan menghitung kecepatan sesaat dalam km/jam.
        bottom_center: (cx, y2) titik kontak roda kendaraan dengan jalan.
        """
        u, v = bottom_center
        x_m, y_m = self.pixel_to_meter(u, v)

        if track_id not in self.track_history:
            self.track_history[track_id] = deque(maxlen=self.history_len)

        history = self.track_history[track_id]
        history.append((timestamp, x_m, y_m, u, v))

        if len(history) < 3:
            return 0.0

        t_start, x_start, y_start, _, _ = history[0]
        t_end, x_end, y_end, _, _ = history[-1]

        dt = t_end - t_start
        if dt <= 0.04:
            return self.speeds.get(track_id, 0.0)

        dx = x_end - x_start
        dy = y_end - y_start
        dist_m = np.sqrt(dx * dx + dy * dy)

        raw_speed_kmh = (dist_m / dt) * 3.6
        raw_speed_kmh = max(0.0, min(raw_speed_kmh, 140.0))

        prev_speed = self.speeds.get(track_id, raw_speed_kmh)
        alpha = 0.4
        smoothed_speed = alpha * raw_speed_kmh + (1 - alpha) * prev_speed

        self.speeds[track_id] = smoothed_speed
        return round(smoothed_speed, 1)

    def get_speed(self, track_id: int) -> float:
        """Mengambil nilai kecepatan terkini untuk track_id tertentu."""
        return self.speeds.get(track_id, 0.0)

    def cleanup(self, active_track_ids: set):
        """Menghapus cache objek yang sudah keluar dari jangkauan kamera."""
        stale_ids = [tid for tid in self.track_history if tid not in active_track_ids]
        for tid in stale_ids:
            self.track_history.pop(tid, None)
            self.speeds.pop(tid, None)
