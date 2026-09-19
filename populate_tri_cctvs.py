"""
populate_tri_cctvs.py
Menghasilkan dan menyelaraskan dataset telemetri dan rekaman insiden untuk Tri-Lokasi:
1. CCTV ID 282: CCTV Perempatan Jendral Sudirman
2. CCTV ID 190: CCTV Flyover Mall Boemi Kedaton
3. CCTV ID 312: CCTV Unila Arah Rajabasa

Menjamin integritas 24 kolom telemetri dan 13 kolom insiden sesuai standar SDLC & STD.
"""

import os
import csv
import random
import time
from datetime import datetime, timedelta

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
TELEM_CSV = os.path.join(LOG_DIR, "traffic_telemetry.csv")
INCIDENT_CSV = os.path.join(LOG_DIR, "incident_records.csv")

EXPECTED_TELEM_HEADER = [
    "timestamp_epoch", "datetime_iso", "location_id", "location_name",
    "frame_id", "track_id", "golongan", "vehicle_subclass", "confidence",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "pixel_center_x", "pixel_center_y",
    "ground_x_meter", "ground_y_meter", "length_meter", "width_meter",
    "speed_kmh", "aspect_ratio", "helmet_status", "compliance_status", "status"
]

EXPECTED_INCIDENT_HEADER = [
    "incident_id", "timestamp_epoch", "datetime_iso", "location_id",
    "location_name", "incident_type", "severity", "involved_track_ids",
    "golongan_types", "speed_kmh", "location_description", "title", "description"
]


def generate_mbk_telemetry(num_records: int = 1200):
    """Menghasilkan data telemetri realistis untuk Flyover Mall Boemi Kedaton."""
    records = []
    base_time = datetime.now() - timedelta(hours=2)
    start_epoch = base_time.timestamp()

    # Model kendaraan jembatan layang MBK
    golongan_weights = [
        ("Golongan I", "Mobil Penumpang", 0.44, 4.2, 1.8, (38.0, 62.0), True),
        ("Golongan VI-A", "Sepeda Motor (Taat Helm)", 0.42, 1.9, 0.8, (35.0, 58.0), True),
        ("Golongan VI-B", "Sepeda Motor (Tanpa Helm)", 0.04, 1.9, 0.8, (32.0, 55.0), False),
        ("Golongan II", "Truk 2 Gandar / Box", 0.07, 6.8, 2.3, (30.0, 48.0), True),
        ("Golongan III", "Truk 3 Gandar", 0.02, 9.5, 2.5, (28.0, 42.0), True),
        ("Golongan IV", "Truk 4 Gandar", 0.01, 12.0, 2.5, (25.0, 40.0), True),
    ]
    gol_items = [g[0] for g in golongan_weights]
    gol_probs = [g[2] for g in golongan_weights]

    gol_map = {g[0]: g for g in golongan_weights}

    frame_id = 1
    current_epoch = start_epoch
    track_id_counter = 8000

    for i in range(num_records):
        current_epoch += random.uniform(0.1, 0.4)
        if i % 3 == 0:
            frame_id += 1
            track_id_counter += 1

        tid = track_id_counter
        chosen_gol = random.choices(gol_items, weights=gol_probs, k=1)[0]
        gol_info = gol_map[chosen_gol]

        subclass = gol_info[1]
        base_len = gol_info[3] + random.uniform(-0.3, 0.3)
        base_wid = gol_info[4] + random.uniform(-0.15, 0.15)
        spd_range = gol_info[5]
        speed_kmh = round(random.uniform(spd_range[0], spd_range[1]), 1)
        has_helmet = gol_info[6]

        confidence = round(random.uniform(0.68, 0.96), 3)

        # Koordinat proyeksi bentang flyover MBK (lebar 10m x panjang 30m)
        gx = round(random.uniform(1.2, 8.8), 2)
        gy = round(random.uniform(2.0, 28.0), 2)

        # Bounding box di resolusi 1280x720
        px = int(450 + (gx / 10.0) * 400 + random.uniform(-20, 20))
        py = int(200 + (gy / 30.0) * 450 + random.uniform(-15, 15))
        bw = int(35 + (base_wid * 18))
        bh = int(40 + (base_len * 12))
        x1 = max(0, px - bw // 2)
        y1 = max(0, py - bh // 2)
        x2 = min(1280, x1 + bw)
        y2 = min(720, y1 + bh)

        aspect_ratio = round(bw / float(max(1, bh)), 2)

        if chosen_gol == "Golongan VI-B":
            helm_st = "TANPA_HELM"
            comp_st = "MELANGGAR"
            status = "PELANGGARAN_HELM"
        elif chosen_gol == "Golongan VI-A":
            helm_st = "HELM"
            comp_st = "TAAT_HUKUM"
            status = "NORMAL"
        else:
            helm_st = "N/A"
            comp_st = "TAAT_HUKUM"
            status = "NORMAL"

        if speed_kmh > 65.0:
            status = "KECEPATAN_TINGGI"

        dt_iso = datetime.fromtimestamp(current_epoch).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        row = [
            f"{current_epoch:.3f}",
            dt_iso,
            190,
            "CCTV Flyover Mall Boemi Kedaton",
            frame_id,
            tid,
            chosen_gol,
            subclass,
            confidence,
            x1, y1, x2, y2,
            round((x1 + x2) / 2.0, 1),
            round((y1 + y2) / 2.0, 1),
            gx, gy,
            round(base_len, 2),
            round(base_wid, 2),
            speed_kmh,
            aspect_ratio,
            helm_st,
            comp_st,
            status
        ]
        records.append(row)

    return records


def generate_mbk_incidents(num_incidents: int = 35):
    """Menghasilkan insiden realistis spesifik karakteristik Flyover Mall Boemi Kedaton."""
    incidents = []
    base_time = datetime.now() - timedelta(hours=2)

    incident_templates = [
        ("PELANGGARAN_HELM", "WARNING", "ETLE: Pengendara Motor Tanpa Helm di Flyover MBK",
         "Pengendara melintasi bentang layang MBK tanpa mengenakan helm SNI berstandar keselamatan.",
         "Bentang Atas Flyover MBK", "Golongan VI-B", 42.0),
        ("KECEPATAN_TINGGI", "WARNING", "Overspeeding: Melebihi Batas Kecepatan Flyover",
         "Kendaraan melaju kencang di turunan flyover MBK melebihi ambang batas keselamatan 60 km/jam.",
         "Turunan Flyover MBK Arah Mall", "Golongan I", 71.5),
        ("KENDARAAN_MOGOK", "CRITICAL", "Bahaya: Kendaraan Mogok di Puncak Jembatan Layang",
         "Satu unit kendaraan terhenti statis lebih dari 15 detik di lajur utama jembatan layang memicu potensi tabrakan beruntun.",
         "Puncak Elevasi Flyover MBK", "Golongan I", 0.0),
        ("PENGEREMAN_MENDADAK", "WARNING", "Pengereman Mendadak di Zona Bottleneck Mall",
         "Penurunan kecepatan drastis akibat penyempitan arus lalu lintas di mulut turunan flyover samping MBK.",
         "Mulut Turunan Flyover Samping Mall", "Golongan I", 14.2),
        ("MOTOR_JATUH", "CRITICAL", "Insiden: Sepeda Motor Tergelincir di Siar Muai Flyover",
         "Pengendara sepeda motor tergelincir akibat permukaan licin pada sambungan dilatasi jembatan layang MBK.",
         "Sambungan Siar Muai Puncak Flyover", "Golongan VI-A", 8.0),
    ]

    for i in range(num_incidents):
        tmpl = random.choice(incident_templates)
        itype, sev, title, desc, loc_desc, gol_str, spd = tmpl
        t_offset = random.uniform(60, 7000)
        inc_dt = base_time + timedelta(seconds=t_offset)
        inc_epoch = inc_dt.timestamp()
        inc_id = f"INC_MBK_{int(inc_epoch)}_{i+1:03d}"
        tid = random.randint(8010, 8990)
        speed = round(spd + random.uniform(-4.0, 5.0), 1) if spd > 0 else 0.0

        row = [
            inc_id,
            f"{inc_epoch:.3f}",
            inc_dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            190,
            "CCTV Flyover Mall Boemi Kedaton",
            itype,
            sev,
            str(tid),
            gol_str,
            speed,
            loc_desc,
            title,
            desc
        ]
        incidents.append(row)

    return incidents


def main():
    print("=== Mensintesis Data Telemetri & Insiden Tri-Lokasi ===")
    
    # 1. Update Telemetry CSV
    mbk_rows = generate_mbk_telemetry(num_records=1500)
    
    # Baca data lama jika ada
    existing_rows = []
    if os.path.exists(TELEM_CSV):
        with open(TELEM_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) == len(EXPECTED_TELEM_HEADER):
                    # Filter out existing 280 and 190 to prevent duplicate accumulation
                    if r[2] not in ("280", "190"):
                        existing_rows.append(r)

    total_telem = existing_rows + mbk_rows
    # Urutkan berdasarkan timestamp
    total_telem.sort(key=lambda x: float(x[0]) if x[0].replace('.', '', 1).isdigit() else 0.0)

    with open(TELEM_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_TELEM_HEADER)
        writer.writerows(total_telem)

    print(f"-> Telemetry CSV diperbarui: {len(total_telem)} baris total (termasuk {len(mbk_rows)} data MBK).")

    # 2. Update Incident CSV
    mbk_incidents = generate_mbk_incidents(num_incidents=40)
    existing_incidents = []
    if os.path.exists(INCIDENT_CSV):
        with open(INCIDENT_CSV, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            for r in reader:
                if len(r) == len(EXPECTED_INCIDENT_HEADER):
                    if r[3] not in ("280", "190"):
                        existing_incidents.append(r)

    total_incidents = existing_incidents + mbk_incidents
    total_incidents.sort(key=lambda x: float(x[1]) if x[1].replace('.', '', 1).isdigit() else 0.0)

    with open(INCIDENT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EXPECTED_INCIDENT_HEADER)
        writer.writerows(total_incidents)

    print(f"-> Incident CSV diperbarui: {len(total_incidents)} baris total (termasuk {len(mbk_incidents)} insiden MBK).")


if __name__ == "__main__":
    main()
