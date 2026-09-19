"""
detector.py
Modul deteksi dan klasifikasi kendaraan berdasarkan Sistem Penggolongan Standar Indonesia
(Kepmenhub / BPJT Jalan & Jembatan) dengan Penegakan Hukum Elektronik (ETLE):
- Golongan I    : Sedan, jip, pick-up, truk kecil, dan bus
- Golongan II   : Truk dengan 2 (dua) gandar (sumbu roda)
- Golongan III  : Truk dengan 3 (tiga) gandar
- Golongan IV   : Truk dengan 4 (empat) gandar
- Golongan V    : Truk dengan 5 (lima) gandar atau lebih
- Golongan VI-A : Kendaraan bermotor roda 2 (Sepeda Motor) - TAAT HUKUM (Pengendara Menggunakan Helm)
- Golongan VI-B : Kendaraan bermotor roda 2 (Sepeda Motor) - MELANGGAR ATURAN (Pengendara Tanpa Helm)
"""

from collections import deque
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

# Pemetaan ID kelas COCO ke nama dasar
COCO_VEHICLE_MAP = {
    0: "person",      # Pengendara motor yang terdeteksi terpisah / bersamaan
    1: "bicycle",     # Sepeda / moped listrik
    2: "car",         # Mobil penumpang
    3: "motorcycle",  # Sepeda motor
    5: "bus",         # Bus
    7: "truck",       # Truk
}

# Warna visualisasi bounding box per Golongan Kendaraan (BGR)
GOLONGAN_COLORS = {
    "Golongan I": (0, 215, 255),       # Kuning terang (Sedan/Jip/Pick-up/Bus)
    "Golongan II": (0, 255, 128),      # Hijau mint (Truk 2 Gandar)
    "Golongan III": (255, 191, 0),     # Biru muda (Truk 3 Gandar)
    "Golongan IV": (255, 105, 180),    # Pink keunguan (Truk 4 Gandar)
    "Golongan V": (200, 0, 255),       # Ungu tua (Truk 5+ Gandar / Trailer)
    "Golongan VI-A": (0, 220, 100),    # Hijau Emerald (Sepeda Motor Taat Helm)
    "Golongan VI-B": (0, 69, 255),     # Merah Terang (Sepeda Motor Melanggar / Tanpa Helm)
    "Golongan VI": (255, 140, 0),      # Oranye umum
}

# Polygon Area Badan Jalan (Road ROI) Underpass Unila Arah Rajabasa (1280x720)
# Dikalibrasi mencakup kedua lajur (lajur naik ke Rajabasa dan lajur turun underpass)
DEFAULT_ROAD_ROI = np.array([
    [20, 0],      # Sudut kiri atas turunan underpass (lajur Rajabasa)
    [650, 0],     # Sudut kanan atas turunan underpass (lajur masuk)
    [1050, 720],  # Sudut kanan bawah dekat dinding mural
    [20, 720],    # Sudut kiri bawah jalan
], np.int32)
UNILA_ROAD_ROI = DEFAULT_ROAD_ROI

# Polygon Area Badan Jalan (Road ROI) Perempatan Jl. Jendral Sudirman (1280x720)
# Mencakup simpang jalan di bawah flyover secara luas
SUDIRMAN_ROAD_ROI = np.array([
    [50, 120],   # Kiri atas di bawah flyover
    [980, 120],   # Kanan atas di bawah flyover
    [1250, 720],  # Kanan bawah
    [0, 720],     # Kiri bawah
    [0, 240],     # Kiri tengah
], np.int32)

# Polygon Area Badan Jalan (Road ROI) CCTV Flyover Mall Boemi Kedaton (1280x720) (ID 190)
# Mencakup badan jembatan layang (flyover) 2 lajur terpisah elevasi
MBK_ROAD_ROI = np.array([
    [320, 120],   # Kiri atas bentang tanjakan flyover
    [920, 120],   # Kanan atas bentang tanjakan flyover
    [1220, 720],  # Kanan bawah turunan flyover
    [80, 720],    # Kiri bawah turunan flyover
], np.int32)


def detect_helmet(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None) -> Tuple[bool, float]:
    """
    Algoritma Computer Vision untuk mendeteksi apakah pengendara sepeda motor mengenakan helm.
    Mengevaluasi:
    1. Region of Interest (ROI) kepala pada area atas sepeda motor (top 15-32%).
    2. Kurvatur elipsoid / konveksitas kubah helm (Convexity & Circularity).
    3. Reflektansi cahaya specular highlight pada permukaan batok helm (Gloss).
    4. Segmentasi warna kulit (Skin-Tone Masking di ruang warna HSV dan YCbCr).
    
    Mengembalikan: (has_helmet: bool, confidence_score: float)
    """
    if bbox is None:
        head_roi = frame
        roi_area = head_roi.shape[0] * head_roi.shape[1]
        if roi_area < 25:
            return True, 0.70
    else:
        x1, y1, x2, y2 = bbox
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        frame_h, frame_w = frame.shape[:2]

        # Ekstraksi ROI kepala pengendara (bagian atas objek motor/pengendara)
        head_y1 = max(0, y1)
        head_y2 = min(frame_h, y1 + int(h * 0.32))
        head_x1 = max(0, x1 + int(w * 0.15))
        head_x2 = min(frame_w, x2 - int(w * 0.15))

        if head_y2 - head_y1 < 6 or head_x2 - head_x1 < 6:
            # Resolusi terlalu kecil, asumsikan taat hukum sebagai prior konservatif
            return True, 0.70

        head_roi = frame[head_y1:head_y2, head_x1:head_x2]
        roi_area = (head_y2 - head_y1) * (head_x2 - head_x1)

    # 1. Analisis Warna Kulit (HSV & YCbCr) - Tanpa helm akan memiliki porsi kulit wajah/leher/telinga tinggi
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    lower_skin_hsv = np.array([0, 25, 40], dtype=np.uint8)
    upper_skin_hsv = np.array([25, 180, 255], dtype=np.uint8)
    skin_mask_hsv = cv2.inRange(hsv, lower_skin_hsv, upper_skin_hsv)

    ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
    lower_skin_ycb = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin_ycb = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask_ycb = cv2.inRange(ycbcr, lower_skin_ycb, upper_skin_ycb)

    skin_mask = cv2.bitwise_and(skin_mask_hsv, skin_mask_ycb)
    skin_pixels = cv2.countNonZero(skin_mask)
    skin_ratio = skin_pixels / float(roi_area)

    # 2. Analisis Reflektansi Specular (Batok helm mengkilap memantulkan cahaya lampu jalan)
    # Nilai V tinggi dan Saturasi rendah (putih/kilap terang)
    v_channel = hsv[:, :, 2]
    s_channel = hsv[:, :, 1]
    specular_mask = (v_channel > 175) & (s_channel < 90)
    specular_ratio = np.sum(specular_mask) / float(roi_area)

    # 3. Analisis Kontur & Konveksitas (Bentuk cangkang helm kubah/elips halus)
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    convexity_score = 0.5
    if contours:
        c = max(contours, key=cv2.contourArea)
        c_area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area > 10:
            solidity = c_area / float(hull_area)
            perimeter = cv2.arcLength(c, True)
            circularity = (4 * np.pi * c_area) / (perimeter ** 2 + 1e-5) if perimeter > 0 else 0
            convexity_score = min(1.0, 0.6 * solidity + 0.4 * min(1.0, circularity * 2.0))

    # 4. Fusi Skor Fitur Deteksi Helm
    # Helm: skin_ratio rendah (< 0.20), specular_ratio ada/cukup, convexity tinggi
    # Tanpa Helm: skin_ratio tinggi (> 0.25) atau rambut bertekstur kasar tanpa kubah kilap
    skin_penalty = max(0.0, 1.0 - skin_ratio * 3.5)
    specular_bonus = min(1.0, specular_ratio * 8.0)

    helmet_score = (
        0.35 * skin_penalty +
        0.35 * convexity_score +
        0.30 * specular_bonus
    )

    has_helmet = bool(helmet_score >= 0.45)
    confidence = float(np.clip(helmet_score, 0.05, 0.99))
    return has_helmet, confidence


def classify_indonesian_golongan(
    coco_cls_id: int,
    length_m: float = 4.0,
    width_m: float = 1.8,
    aspect_ratio: float = 0.8,
    has_helmet: bool = True,
) -> Tuple[str, str, str]:
    """
    Mengklasifikasikan objek ke dalam Sistem Penggolongan Kendaraan Standar Indonesia.
    Mendukung pemisahan resmi Golongan VI:
    - Golongan VI-A: Sepeda Motor (Taat Hukum / Pengendara Menggunakan Helm)
    - Golongan VI-B: Sepeda Motor (Melanggar / Pengendara Tanpa Helm)
    """
    # Deteksi apakah objek ini secara fisik adalah sepeda motor
    is_motorcycle = False
    if coco_cls_id in (1, 3):  # motorcycle atau bicycle
        is_motorcycle = True
    elif coco_cls_id == 0:  # person di badan jalan yang bergerak cepat
        is_motorcycle = True
    elif coco_cls_id == 2:
        # Heuristik koreksi: Objek COCO car yang memiliki dimensi fisik sepeda motor
        # (hanya jika seluruh dimensi sempit: rasio < 0.55, lebar < 1.35m, dan panjang < 2.5m)
        if aspect_ratio < 0.55 and width_m < 1.35 and length_m < 2.5:
            is_motorcycle = True

    if is_motorcycle:
        if has_helmet:
            return "Golongan VI-A", "Golongan VI-A: Sepeda Motor (Taat Helm)", "Sepeda Motor (Taat Helm)"
        else:
            return "Golongan VI-B", "Golongan VI-B: Sepeda Motor (Tanpa Helm)", "Sepeda Motor (Tanpa Helm)"

    elif coco_cls_id == 2:
        # Mobil penumpang: Sedan, Jip, Minibus, Pick-up
        return "Golongan I", "Golongan I: Sedan/Jip/Pick-up", "Sedan/Jip/Pick-up"

    elif coco_cls_id == 5:
        # Bus
        return "Golongan I", "Golongan I: Bus", "Bus"

    elif coco_cls_id == 7:
        # Truk: dibedakan berdasarkan panjang fisik (meter) hasil homografi
        if length_m < 5.0:
            return "Golongan I", "Golongan I: Truk Kecil", "Truk Kecil"
        elif 5.0 <= length_m < 8.0:
            return "Golongan II", "Golongan II: Truk 2 Gandar", "Truk 2 Gandar"
        elif 8.0 <= length_m < 11.5:
            return "Golongan III", "Golongan III: Truk 3 Gandar", "Truk 3 Gandar"
        elif 11.5 <= length_m < 14.0:
            return "Golongan IV", "Golongan IV: Truk 4 Gandar", "Truk 4 Gandar"
        else:
            return "Golongan V", "Golongan V: Truk 5+ Gandar", "Truk 5+ Gandar"

    return "Golongan I", "Golongan I: Kendaraan Umum", "Lainnya"


# Alias kompatibilitas fungsi penggolongan dan pengujian
classify_vehicle_indonesia = classify_indonesian_golongan
detect_helmet_heuristic = detect_helmet


_YOLO_CACHE: Dict[str, Any] = {}


def get_shared_yolo_model(model_path: str = "yolov8n.pt"):
    """Mengembalikan instance model YOLO singleton untuk efisiensi memori."""
    if model_path not in _YOLO_CACHE:
        import os
        os.environ["YOLO_OFFLINE"] = "1"
        os.environ["ULTRALYTICS_AUTOINSTALL"] = "0"
        from ultralytics import YOLO
        _YOLO_CACHE[model_path] = YOLO(model_path)
    return _YOLO_CACHE[model_path]


class VehicleDetector:
    """Mendeteksi, mengklasifikasikan ke Golongan I-VI (termasuk VI-A & VI-B), dan melacak kendaraan."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        conf_threshold: float = 0.20,
        road_roi: Optional[np.ndarray] = None,
        location: str = "sudirman",
    ):
        self.model = get_shared_yolo_model(model_path)
        self.conf_threshold = conf_threshold
        self.location = str(location).lower()
        if road_roi is not None:
            self.road_roi = road_roi
        elif "mbk" in self.location or "kedaton" in self.location or "190" in self.location or "280" in self.location or "boemi" in self.location:
            self.road_roi = MBK_ROAD_ROI
        elif "sudirman" in self.location or "282" in self.location:
            self.road_roi = SUDIRMAN_ROAD_ROI
        else:
            self.road_roi = DEFAULT_ROAD_ROI

        # Buffer konsistensi temporal helm per track_id
        # {track_id: deque([bool, bool, ...], maxlen=15)}
        self.helmet_history: Dict[int, deque] = {}

        # Fallback centroid tracker jika ByteTrack belum assign id
        self.next_fallback_id = 5000
        self.centroid_tracks: Dict[int, Tuple[float, float, int]] = {}
        self.current_frame_id = 0

        # Statistik total kendaraan unik per Golongan
        self.counted_ids = set()
        self.counts_by_golongan = {
            "Golongan I": 0,
            "Golongan II": 0,
            "Golongan III": 0,
            "Golongan IV": 0,
            "Golongan V": 0,
            "Golongan VI-A": 0,
            "Golongan VI-B": 0,
        }

    def is_in_roi(self, point: Tuple[float, float]) -> bool:
        """Memeriksa apakah titik kontak kendaraan berada di dalam polygon ROI jalan dengan toleransi margin."""
        if self.road_roi is None:
            return True
        dist = cv2.pointPolygonTest(self.road_roi, (float(point[0]), float(point[1])), True)
        return dist >= -45.0  # Toleransi margin 45 piksel dari tepi poligon jalan

    def _match_or_create_track(self, center: Tuple[float, float]) -> int:
        """Fallback tracker bila ByteTrack belum menetapkan id."""
        cx, cy = center
        best_id = -1
        min_dist = 70.0
        for tid, (tcx, tcy, t_frame) in list(self.centroid_tracks.items()):
            if (self.current_frame_id - t_frame) <= 4:
                dist = float(np.sqrt((cx - tcx)**2 + (cy - tcy)**2))
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
        if best_id != -1:
            self.centroid_tracks[best_id] = (cx, cy, self.current_frame_id)
            return best_id
        else:
            new_id = self.next_fallback_id
            self.next_fallback_id += 1
            self.centroid_tracks[new_id] = (cx, cy, self.current_frame_id)
            return new_id

    def detect_and_track(self, frame: np.ndarray, speed_estimator=None) -> List[Dict]:
        """
        Menjalankan deteksi dan pelacakan YOLOv8 pada frame.
        Mengklasifikasikan ke Golongan I s.d. VI-A/VI-B berdasarkan kelas COCO,
        koreksi perspektif geometri, dan deteksi helm.
        """
        self.current_frame_id += 1
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
            conf=self.conf_threshold,
            classes=list(COCO_VEHICLE_MAP.keys()),
        )

        detections = []
        if not results or len(results) == 0:
            return detections

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return detections

        active_track_ids = set()

        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = (int(x1), int(y1), int(x2), int(y2))
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            bottom_center = ((x1 + x2) / 2.0, y2)
            center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
            aspect_ratio = w / h

            if box.id is not None:
                track_id = int(box.id[0].item())
                self.centroid_tracks[track_id] = (center[0], center[1], self.current_frame_id)
            else:
                track_id = self._match_or_create_track(center)

            # Filter spasial: titik kontak roda atau pusat harus berada di badan jalan underpass
            if not self.is_in_roi(bottom_center) and not self.is_in_roi(center):
                continue

            # Estimasi dimensi fisik meter via Homografi
            if speed_estimator is not None:
                length_m, width_m = speed_estimator.estimate_dimensions_meter(bbox)
            else:
                length_m, width_m = 4.2, 1.8

            # Evaluasi deteksi helm jika berpotensi merupakan sepeda motor
            is_potential_motor = (
                cls_id in (1, 3) or
                cls_id == 0 or
                (cls_id == 2 and (aspect_ratio < 0.78 or width_m < 1.35))
            )

            has_helmet = True
            helmet_conf = 0.85
            if is_potential_motor:
                inst_helmet, helmet_conf = detect_helmet(frame, bbox)
                if track_id > 0:
                    if track_id not in self.helmet_history:
                        self.helmet_history[track_id] = deque(maxlen=15)
                    self.helmet_history[track_id].append(inst_helmet)
                    # Mayoritas voting temporal
                    has_helmet = (sum(self.helmet_history[track_id]) / len(self.helmet_history[track_id])) >= 0.5
                else:
                    has_helmet = inst_helmet

            golongan, golongan_label, subclass = classify_indonesian_golongan(
                coco_cls_id=cls_id,
                length_m=length_m,
                width_m=width_m,
                aspect_ratio=aspect_ratio,
                has_helmet=has_helmet,
            )

            if track_id > 0:
                active_track_ids.add(track_id)
                # Catat hitungan unik kendaraan
                if track_id not in self.counted_ids:
                    self.counted_ids.add(track_id)
                    self.counts_by_golongan[golongan] = self.counts_by_golongan.get(golongan, 0) + 1

            detections.append({
                "track_id": track_id,
                "class_id": cls_id,
                "golongan": golongan,
                "golongan_label": golongan_label,
                "class_name": subclass,
                "conf": conf,
                "bbox": bbox,
                "bottom_center": bottom_center,
                "center": center,
                "width": w,
                "height": h,
                "length_meter": length_m,
                "width_meter": width_m,
                "aspect_ratio": aspect_ratio,
                "has_helmet": has_helmet,
                "helmet_conf": helmet_conf,
            })

        # Bersihkan riwayat helm kendaraan yang sudah lewat
        stale_ids = [k for k in self.helmet_history if k not in active_track_ids and k not in self.counted_ids]
        for sid in stale_ids:
            self.helmet_history.pop(sid, None)

        return detections

    def draw_roi(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0), thickness: int = 2) -> np.ndarray:
        """Menggambar garis batas ROI jalan pada frame."""
        cv2.polylines(frame, [self.road_roi], isClosed=True, color=color, thickness=thickness)
        return frame

    def get_counts(self) -> Dict[str, int]:
        """Mengembalikan statistik hitungan kendaraan per Golongan I s.d. VI-A & VI-B."""
        return dict(self.counts_by_golongan)
