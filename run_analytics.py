"""
run_analytics.py
Skrip CLI untuk menjalankan pipeline analitik lalu lintas & deteksi kecelakaan secara langsung,
serta menyimpan hasil video teranotasi dan laporan insiden ke file JSON.
"""

import argparse
import json
import time
import cv2
from cctv_stream import CCTVStreamManager
from pipeline import UnderpassAnalyticsPipeline


def main():
    parser = argparse.ArgumentParser(description="Analitik CCTV Underpass Unila Arah Rajabasa")
    parser.add_argument("--source", type=str, default="underpass_live_sample.mp4",
                        help="Sumber video: path file mp4 atau 'live'")
    parser.add_argument("--output", type=str, default="output_annotated.mp4",
                        help="Path output video teranotasi")
    parser.add_argument("--duration", type=int, default=15,
                        help="Durasi pemrosesan dalam detik")
    parser.add_argument("--conf", type=float, default=0.35,
                        help="Confidence threshold YOLOv8")
    parser.add_argument("--save-json", type=str, default="incident_report.json",
                        help="File output JSON laporan insiden")
    args = parser.parse_args()

    print("=" * 65)
    print("🚦 SISTEM ANALITIK CCTV UNDERPASS UNILA ARAH RAJABASA (CCTV 312)")
    print(f"Sumber: {args.source} | Output Video: {args.output}")
    print("=" * 65)

    pipeline = UnderpassAnalyticsPipeline(model_path="yolov8n.pt", conf_threshold=args.conf)

    if args.source == "live":
        print("[INFO] Mengambil live stream langsung dari CCTV 312...")
        mgr = CCTVStreamManager(312)
        # Rekam segmen live sementara untuk diproses
        temp_file = "temp_live_run.mp4"
        num_segs = max(2, args.duration // 4)
        ok = mgr.record_clip(temp_file, num_segments=num_segs)
        if not ok:
            print("[ERROR] Gagal mengunduh live stream.")
            return
        cap = cv2.VideoCapture(temp_file)
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"[ERROR] Tidak dapat membuka file video: {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    frame_idx = 0
    start_time = time.time()
    max_frames = int(args.duration * fps)

    print(f"[INFO] Memproses video ({width}x{height} @ {fps:.1f} FPS, target {max_frames} frame)...")

    while cap.isOpened() and frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        annotated_frame, detections, new_alerts, stats = pipeline.process_frame(frame)
        out.write(annotated_frame)

        if new_alerts:
            for alt in new_alerts:
                print(f"  🚨 [{alt['severity']}] {alt['title']} - {alt['description']}")

        if frame_idx % 25 == 0:
            print(f"  -> Frame {frame_idx}/{max_frames} | Aktif: {stats['current_active_vehicles']} | Total: {stats['total_vehicles']} | Rerata Kecepatan: {stats['avg_speed_kmh']} km/h")

    cap.release()
    out.release()
    elapsed = time.time() - start_time

    print("-" * 65)
    print(f"✅ Pemrosesan Selesai dalam {elapsed:.2f} detik ({frame_idx / elapsed:.1f} FPS)")
    print(f"📁 Video teranotasi tersimpan di: {args.output}")

    # Simpan Laporan JSON
    incident_logs = pipeline.anomaly_detector.get_incident_log()
    report_data = {
        "cctv_name": "UNDERPASS UNILA ARAH RAJABASA",
        "cctv_id": 312,
        "processed_frames": frame_idx,
        "elapsed_seconds": round(elapsed, 2),
        "vehicle_counts": pipeline.detector.get_counts(),
        "total_unique_vehicles": len(pipeline.detector.counted_ids),
        "total_incidents": len(incident_logs),
        "incidents": [
            {
                "id": inc["id"],
                "time": inc["time"],
                "type": inc["type"],
                "severity": inc["severity"],
                "title": inc["title"],
                "description": inc["description"],
                "track_ids": inc["track_ids"],
            }
            for inc in incident_logs
        ],
    }

    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"📄 Laporan insiden tersimpan di: {args.save_json}")
    print(json.dumps(report_data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
