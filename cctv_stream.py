"""
cctv_stream.py
Modul untuk menangani pengambilan stream video langsung (HLS m3u8) dan snapshot
CCTV Kota Bandar Lampung:
1. CCTV ID 282: CCTV Perempatan Jendral Sudirman (Simpang Sebidang)
2. CCTV ID 190: CCTV Flyover Mall Boemi Kedaton (Jembatan Layang Arteri)
3. CCTV ID 312: CCTV Unila Arah Rajabasa (Terowongan Bebas Hambatan)

Dilengkapi LiveHLSStreamReader dengan background worker untuk streaming tanpa jeda.
"""

import os
import re
import time
import queue
import threading
import urllib.request
import http.cookiejar
from typing import Dict, Generator, Optional, Tuple
import cv2
import numpy as np

STREAM_BASE_URL = "https://seribuwajah.bandarlampungkota.go.id"
API_BASE_URL = "https://api-newseribuwajah.bandarlampungkota.go.id"
API_KEY = "ParkirIlegalDetectKey2026"
DEFAULT_CCTV_ID = 282

CCTV_LOCATIONS: Dict[int, str] = {
    282: "CCTV Perempatan Jendral Sudirman",
    190: "CCTV Flyover Mall Boemi Kedaton",
    312: "CCTV Unila Arah Rajabasa",
}


class CCTVStreamManager:
    """Mengelola koneksi sesi HLS dan snapshot CCTV Bandar Lampung."""

    def __init__(self, cctv_id: int = DEFAULT_CCTV_ID):
        self.cctv_id = cctv_id
        self.session_cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.session_cookie_jar)
        )
        self.opener.addheaders = [
            ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"),
            ("Referer", f"{STREAM_BASE_URL}/list"),
            ("Origin", STREAM_BASE_URL),
        ]
        self._session_initialized = False

    def init_hls_session(self) -> bool:
        """Melakukan handshake cookie session untuk HLS streaming."""
        init_url = f"{STREAM_BASE_URL}/cctv_{self.cctv_id}/index.m3u8"
        try:
            req = urllib.request.Request(init_url)
            with self.opener.open(req, timeout=10) as resp:
                _ = resp.read()
            self._session_initialized = True
            return True
        except Exception as e:
            print(f"[CCTVStreamManager] Gagal inisialisasi sesi HLS: {e}")
            return False

    def get_latest_playlist(self) -> Optional[str]:
        """Mengambil isi playlist m3u8 terbaru."""
        if not self._session_initialized:
            if not self.init_hls_session():
                return None

        url = f"{STREAM_BASE_URL}/cctv_{self.cctv_id}/main_stream.m3u8"
        try:
            req = urllib.request.Request(url)
            with self.opener.open(req, timeout=10) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            print(f"[CCTVStreamManager] Gagal mengambil playlist: {e}")
            self.init_hls_session()
            return None

    def download_segment(self, segment_filename: str) -> Optional[bytes]:
        """Mengunduh satu segmen TS."""
        url = f"{STREAM_BASE_URL}/cctv_{self.cctv_id}/{segment_filename}"
        try:
            req = urllib.request.Request(url)
            with self.opener.open(req, timeout=10) as resp:
                return resp.read()
        except Exception as e:
            print(f"[CCTVStreamManager] Gagal mengunduh segmen {segment_filename}: {e}")
            return None

    def fetch_snapshot(self) -> Optional[np.ndarray]:
        """Mengambil 1 frame snapshot real-time dari segmen HLS terbaru."""
        try:
            playlist = self.get_latest_playlist()
            if playlist:
                segs = re.findall(r"[\w-]+\.ts", playlist)
                if segs:
                    data = self.download_segment(segs[-1])
                    if data and len(data) > 1000:
                        import tempfile
                        tmp_seg = os.path.join(tempfile.gettempdir(), f"snap_{self.cctv_id}.ts")
                        with open(tmp_seg, "wb") as f:
                            f.write(data)
                        cap = cv2.VideoCapture(tmp_seg)
                        ret, frame = cap.read()
                        cap.release()
                        if os.path.exists(tmp_seg):
                            os.remove(tmp_seg)
                        if ret and frame is not None:
                            if frame.shape[:2] != (720, 1280):
                                frame = cv2.resize(frame, (1280, 720))
                            return frame
        except Exception as e:
            print(f"[CCTVStreamManager] Gagal mengambil snapshot dari HLS: {e}")
        return None

    def record_clip(self, output_path: str, num_segments: int = 3) -> bool:
        """Merekam beberapa segmen live HLS berurutan ke direktori lokal /tmp dan menyimpannya menjadi MP4."""
        import tempfile
        tmp_dir = tempfile.gettempdir()
        print(f"[CCTVStreamManager] Merekam {num_segments} segmen CCTV ke {output_path}...")
        downloaded_segs = set()
        ts_temp_files = []

        try:
            for i in range(num_segments):
                playlist = self.get_latest_playlist()
                if not playlist:
                    time.sleep(1.5)
                    continue

                segs = re.findall(r"[\w-]+\.ts", playlist)
                new_segs = [s for s in segs if s not in downloaded_segs]
                if not new_segs:
                    time.sleep(1.5)
                    continue

                target_seg = new_segs[-1]
                data = self.download_segment(target_seg)
                if data and len(data) > 1000:
                    downloaded_segs.add(target_seg)
                    temp_ts = os.path.join(tmp_dir, f"rec_seg_{self.cctv_id}_{i}.ts")
                    with open(temp_ts, "wb") as f:
                        f.write(data)
                    ts_temp_files.append(temp_ts)
                    print(f"  -> Terunduh segmen {i+1}/{num_segments}: {target_seg} ({len(data)} bytes)")

                if i < num_segments - 1:
                    time.sleep(3.5)

            if not ts_temp_files:
                print("[CCTVStreamManager] Tidak ada segmen yang berhasil diunduh.")
                return False

            concat_list_file = os.path.join(tmp_dir, f"concat_{self.cctv_id}.txt")
            with open(concat_list_file, "w") as f:
                for ts_file in ts_temp_files:
                    f.write(f"file '{ts_file}'\n")

            cmd = f"ffmpeg -y -f concat -safe 0 -i \"{concat_list_file}\" -c copy \"{output_path}\" 2>/dev/null"
            ret = os.system(cmd)

            for ts in ts_temp_files:
                if os.path.exists(ts):
                    os.remove(ts)
            if os.path.exists(concat_list_file):
                os.remove(concat_list_file)

            if ret == 0 and os.path.exists(output_path):
                print(f"[CCTVStreamManager] Sukses menyimpan rekaman: {output_path}")
                return True
            else:
                return False

        except Exception as e:
            print(f"[CCTVStreamManager] Kesalahan saat merekam: {e}")
            return False


class LiveHLSStreamReader:
    """
    Membaca aliran video HLS CCTV Kota Bandar Lampung secara real-time dan stabil.
    Menjalankan background worker dengan buffer di /tmp agar tidak membebani Google Drive.
    """

    def __init__(self, cctv_id: int = DEFAULT_CCTV_ID, queue_size: int = 120):
        import tempfile
        self.cctv_id = cctv_id
        self.mgr = CCTVStreamManager(cctv_id)
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = True
        self.downloaded_segs = set()
        self.tmp_dir = tempfile.gettempdir()
        self.last_valid_frame: Optional[np.ndarray] = None

        # Inisialisasi handshake sesi HLS
        self.mgr.init_hls_session()

        # Jalankan background worker thread
        self.worker_thread = threading.Thread(target=self._download_worker, daemon=True)
        self.worker_thread.start()

    def _download_worker(self):
        """Mengunduh segmen HLS baru secara kontinu dan mengekstrak frame ke antrean."""
        while self.running:
            try:
                playlist = self.mgr.get_latest_playlist()
                if not playlist:
                    time.sleep(1.0)
                    continue

                segs = re.findall(r"[\w-]+\.ts", playlist)
                if not segs:
                    time.sleep(1.0)
                    continue

                new_segs = [s for s in segs if s not in self.downloaded_segs]
                if not new_segs:
                    time.sleep(0.8)
                    continue

                # Ambil segmen terbaru
                target_seg = new_segs[-1]
                self.downloaded_segs.add(target_seg)
                if len(self.downloaded_segs) > 50:
                    self.downloaded_segs = set(list(self.downloaded_segs)[-25:])

                data = self.mgr.download_segment(target_seg)
                if data and len(data) > 1000:
                    seg_path = os.path.join(self.tmp_dir, f"cctv_{self.cctv_id}_{target_seg}")
                    with open(seg_path, "wb") as f:
                        f.write(data)

                    cap = cv2.VideoCapture(seg_path)
                    count = 0
                    while cap.isOpened() and self.running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        count += 1
                        # Sampling 1:2 untuk sinkronisasi 12.5 FPS real-time dengan inferensi AI
                        if count % 2 == 0:
                            self.last_valid_frame = frame
                            if self.frame_queue.full():
                                try:
                                    self.frame_queue.get_nowait()
                                except queue.Empty:
                                    pass
                            self.frame_queue.put(frame)
                    cap.release()
                    if os.path.exists(seg_path):
                        os.remove(seg_path)

                time.sleep(0.4)

            except Exception as e:
                print(f"[LiveHLSStreamReader] Peringatan worker CCTV {self.cctv_id}: {e}")
                time.sleep(1.2)

    def read(self, timeout: float = 4.0) -> Tuple[bool, Optional[np.ndarray]]:
        """Mengambil frame berikutnya dari antrean secara real-time tanpa hanging."""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame
        except queue.Empty:
            if self.last_valid_frame is not None:
                return True, self.last_valid_frame.copy()
            return False, None

    def stop(self):
        """Menghentikan thread pembaca live stream."""
        self.running = False


def get_video_stream(source: str) -> cv2.VideoCapture:
    """Membuka video stream dari file video lokal."""
    return cv2.VideoCapture(source)
