"""
charts_journal.py
Modul Visualisasi Empiris dan Grafik Berkualitas Publikasi Jurnal Ilmiah
(IEEE Transactions on ITS / Elsevier Transportation Research Part C)
Menganalisis data telemetri lalu lintas dan insiden CCTV Underpass Unila & Simpang Sudirman.
"""

from datetime import datetime, date
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st


# Palet warna standar jurnal ilmiah dan penggolongan kendaraan Indonesia
GOLONGAN_HEX_COLORS = {
    "Golongan I": "#3182ce",       # Biru Elektrik (Mobil Penumpang/Bus/Pick-up)
    "Golongan II": "#38a169",      # Hijau Mint (Truk 2 Gandar)
    "Golongan III": "#4fd1c5",     # Biru Toska (Truk 3 Gandar / Tronton)
    "Golongan IV": "#b794f4",      # Ungu Muda (Truk 4 Gandar)
    "Golongan V": "#805ad5",       # Ungu Tua (Truk 5+ Gandar / Trailer)
    "Golongan VI-A": "#48bb78",    # Hijau Emerald (Sepeda Motor Taat Helm)
    "Golongan VI-B": "#e53e3e",    # Merah Terang (Sepeda Motor Tanpa Helm)
}


def render_journal_publication_charts(
    df_telemetry: pd.DataFrame,
    df_incidents: pd.DataFrame,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    """
    Merender 6 set grafik analitis komprehensif berstandar publikasi jurnal internasional.
    Setiap grafik dilengkapi indikator statistik, tabel data pendukung, tombol unduh CSV,
    dan caption bernomor resmi.
    """
    import altair as alt

    st.markdown("### 📈 Visualisasi & Grafik Empiris untuk Publikasi Jurnal Penelitian")
    st.caption(
        "Koleksi grafik berikut dirancang sesuai standar penulisan artikel ilmiah (*IEEE Transactions on ITS / Elsevier Transportation Research*). "
        "Data dapat diekspor secara tabular untuk pembuatan tabel LaTeX / Word pendukung naskah."
    )

    if df_telemetry.empty:
        st.warning(
            "⚠️ Belum ada data telemetri yang tersedia pada rentang tanggal yang dipilih. "
            "Jalankan inferensi AI pada kamera CCTV untuk mulai mengumpulkan data telemetri aktual."
        )
        return

    # Subtab untuk 6 Figur Publikasi Jurnal
    subtab_g1, subtab_g2, subtab_g3, subtab_g4, subtab_g5, subtab_g6 = st.tabs([
        "🚗 Gbr 1: Komposisi Golongan",
        "⚡ Gbr 2: Distribusi Kecepatan & V85",
        "⏱️ Gbr 3: Dinamika Temporal",
        "🪖 Gbr 4: Kepatuhan Helm (ETLE)",
        "📐 Gbr 5: Validasi Dimensi Fisik",
        "🚨 Gbr 6: Taksonomi Insiden",
    ])

    # =========================================================================
    # GAMBAR 1: DISTRIBUSI PROPORSI PENGGOLONGAN KENDARAAN (MODAL SPLIT)
    # =========================================================================
    with subtab_g1:
        st.markdown("#### 🚗 Gambar 1: Distribusi Proporsi Komposisi Penggolongan Kendaraan (Golongan I - VI)")
        st.caption("Menganalisis volume unik kendaraan yang melintas dikelompokkan menurut Klasifikasi Standar Nasional.")

        # Hitung jumlah kendaraan unik per golongan
        g_counts = df_telemetry.groupby("golongan")["track_id"].nunique().reset_index()
        g_counts.columns = ["Golongan", "Jumlah_Kendaraan_Unik"]
        total_unique = max(1, g_counts["Jumlah_Kendaraan_Unik"].sum())
        g_counts["Proporsi_Persen"] = (g_counts["Jumlah_Kendaraan_Unik"] / total_unique * 100.0).round(2)

        # Kecepatan rerata per golongan
        spd_agg = df_telemetry[df_telemetry["speed_kmh"] > 0].groupby("golongan")["speed_kmh"].agg(["mean", "std", "max"]).reset_index()
        spd_agg.columns = ["Golongan", "Kecepatan_Rerata_kmh", "Standar_Deviasi_kmh", "Kecepatan_Maksimum_kmh"]
        spd_agg = spd_agg.round(1)

        tabel_g1 = pd.merge(g_counts, spd_agg, on="Golongan", how="left").fillna(0.0)

        # Visualisasi Bar Chart Altair
        chart_g1 = alt.Chart(tabel_g1).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
            x=alt.X("Golongan:N", title="Golongan Kendaraan (Standar Indonesia)", sort="-y"),
            y=alt.Y("Jumlah_Kendaraan_Unik:Q", title="Volume Kendaraan Unik (Unit)"),
            color=alt.Color("Golongan:N", scale=alt.Scale(
                domain=list(GOLONGAN_HEX_COLORS.keys()),
                range=list(GOLONGAN_HEX_COLORS.values())
            ), legend=None),
            tooltip=[
                alt.Tooltip("Golongan:N", title="Kategori"),
                alt.Tooltip("Jumlah_Kendaraan_Unik:Q", title="Jumlah Unit"),
                alt.Tooltip("Proporsi_Persen:Q", title="Proporsi (%)", format=".1f"),
                alt.Tooltip("Kecepatan_Rerata_kmh:Q", title="Kecepatan Rerata (km/h)", format=".1f"),
            ]
        ).properties(height=360)

        st.altair_chart(chart_g1, use_container_width=True)

        # Tampilkan tabel statistik pendukung
        st.markdown("**Tabel 1: Ringkasan Kuantitatif Distribusi Golongan & Kecepatan Operasional**")
        st.dataframe(tabel_g1, use_container_width=True)

        st.download_button(
            label="📥 Unduh Data Tabel Gambar 1 (CSV)",
            data=tabel_g1.to_csv(index=False).encode("utf-8"),
            file_name="tabel_gambar_1_distribusi_golongan.csv",
            mime="text/csv",
            key="dl_tabel_g1",
        )

        st.markdown(
            "> *Gambar 1: Distribusi Proporsi Komposisi Penggolongan Kendaraan (Golongan I s.d. VI) Berdasarkan Standar Klasifikasi Nasional Republik Indonesia.*"
        )

    # =========================================================================
    # GAMBAR 2: PROFIL DISTRIBUSI KECEPATAN KENDARAAN & INDIKATOR V85
    # =========================================================================
    with subtab_g2:
        st.markdown("#### ⚡ Gambar 2: Kurva Distribusi Frekuensi Kecepatan Operasional Kendaraan & Kecepatan Persentil ke-85 ($V_{85}$)")
        st.caption(
            "Parameter $V_{85}$ (kecepatan pada atau di bawah di mana 85% pengemudi melaju dalam kondisi arus bebas) "
            "adalah standar rekayasa lalu lintas internasional untuk penetapan batas kecepatan jalan (*safe speed limit*)."
        )

        valid_speeds = df_telemetry[df_telemetry["speed_kmh"] > 1.0]["speed_kmh"].dropna()

        if len(valid_speeds) >= 5:
            mean_v = float(valid_speeds.mean())
            std_v = float(valid_speeds.std())
            v15 = float(np.percentile(valid_speeds, 15))
            v50 = float(np.percentile(valid_speeds, 50))
            v85 = float(np.percentile(valid_speeds, 85))
            max_v = float(valid_speeds.max())

            kpi_cols = st.columns(4)
            kpi_cols[0].metric("⚡ Rerata Kecepatan (μ)", f"{mean_v:.1f} km/h", help="Kecepatan rerata seluruh sampel kendaraan")
            kpi_cols[1].metric("🎯 Kecepatan Median (V50)", f"{v50:.1f} km/h", help="Nilai tengah kecepatan sampel")
            kpi_cols[2].metric("🛑 Kecepatan Persentil ke-85 (V85)", f"{v85:.1f} km/h", delta=f"{v85 - mean_v:+.1f} km/h vs Rerata", help="Batas kecepatan operasional standar ITS")
            kpi_cols[3].metric("📊 Standar Deviasi (σ)", f"{std_v:.1f} km/h", help="Tingkat dispersi kecepatan lalu lintas")

            # Binning kecepatan
            bins = [0, 15, 25, 35, 45, 55, 65, 75, 85, 120]
            labels = ["0–15", "15–25", "25–35", "35–45", "45–55", "55–65", "65–75", "75–85", ">85"]
            speed_binned = pd.cut(valid_speeds, bins=bins, labels=labels, right=False)
            hist_df = speed_binned.value_counts(sort=False).reset_index()
            hist_df.columns = ["Interval_Kecepatan_kmh", "Frekuensi_Sampel"]
            hist_df["Persentase_Kumulatif"] = (hist_df["Frekuensi_Sampel"].cumsum() / len(valid_speeds) * 100.0).round(1)

            # Bar Chart Histogram Kecepatan Altair
            hist_chart = alt.Chart(hist_df).mark_bar(color="#3182ce", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                x=alt.X("Interval_Kecepatan_kmh:N", title="Rentang Kecepatan (km/jam)", sort=None),
                y=alt.Y("Frekuensi_Sampel:Q", title="Frekuensi Sampel Deteksi"),
                tooltip=[
                    alt.Tooltip("Interval_Kecepatan_kmh:N", title="Rentang"),
                    alt.Tooltip("Frekuensi_Sampel:Q", title="Jumlah Sampel"),
                    alt.Tooltip("Persentase_Kumulatif:Q", title="Persentase Kumulatif (%)", format=".1f"),
                ]
            ).properties(height=340)

            st.altair_chart(hist_chart, use_container_width=True)

            st.markdown(f"""
            **Parameter Ilmiah Rekayasa Kecepatan:**
            - **Batas Kecepatan Desain yang Direkomendasikan ($V_{{85}}$):** `{v85:.1f} km/jam`
            - **Kecepatan Minimum Arus Bebas ($V_{{15}}$):** `{v15:.1f} km/jam`
            - **Kecepatan Puncak Terdeteksi ($V_{{\\max}}$):** `{max_v:.1f} km/jam`
            - **Koefisien Variasi ($CV = \\sigma / \\mu$):** `{std_v / mean_v:.2f}` (Menggambarkan keseragaman laju arus)
            """)

            st.download_button(
                label="📥 Unduh Data Tabel Gambar 2 (CSV)",
                data=hist_df.to_csv(index=False).encode("utf-8"),
                file_name="tabel_gambar_2_distribusi_kecepatan_v85.csv",
                mime="text/csv",
                key="dl_tabel_g2",
            )
        else:
            st.info("Sampel kecepatan belum mencukupi (minimal 5 sampel).")

        st.markdown(
            "> *Gambar 2: Kurva Distribusi Frekuensi Kecepatan Operasional Kendaraan dengan Indikator Kecepatan Persentil ke-85 (V85) untuk Evaluasi Keselamatan dan Batas Kecepatan Ilmiah.*"
        )

    # =========================================================================
    # GAMBAR 3: DINAMIKA TREN TEMPORAL KECEPATAN & VOLUME KENDARAAN
    # =========================================================================
    with subtab_g3:
        st.markdown("#### ⏱️ Gambar 3: Dinamika Temporal Fluktuasi Kecepatan Rerata & Kepadatan Volume Sepanjang Waktu")
        st.caption("Menunjukkan pola fluktuasi arus lalu lintas secara runtun waktu (*time-series*) per interval pengamatan.")

        df_t_time = df_telemetry.copy()
        if "datetime_iso" in df_t_time.columns:
            df_t_time["dt"] = pd.to_datetime(df_t_time["datetime_iso"], errors="coerce")
            df_t_time = df_t_time.dropna(subset=["dt"])

            if len(df_t_time) >= 10:
                # Binning waktu per 1 menit atau per 30 detik
                df_t_time["time_bin"] = df_t_time["dt"].dt.floor("1min").dt.strftime("%H:%M")
                agg_time = df_t_time.groupby("time_bin").agg(
                    Kecepatan_Rerata_kmh=("speed_kmh", lambda s: np.mean([v for v in s if v > 0]) if any(v > 0 for v in s) else 0.0),
                    Volume_Kendaraan_Unik=("track_id", "nunique"),
                    Total_Deteksi_Frame=("track_id", "count")
                ).reset_index()
                agg_time["Kecepatan_Rerata_kmh"] = agg_time["Kecepatan_Rerata_kmh"].round(1)

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    chart_speed_time = alt.Chart(agg_time).mark_line(point=True, color="#63b3ed").encode(
                        x=alt.X("time_bin:N", title="Waktu Pengamatan (Jam:Menit)"),
                        y=alt.Y("Kecepatan_Rerata_kmh:Q", title="Kecepatan Rerata (km/h)"),
                        tooltip=["time_bin:N", "Kecepatan_Rerata_kmh:Q", "Volume_Kendaraan_Unik:Q"]
                    ).properties(height=280, title="Fluktuasi Kecepatan Rerata (km/h)")
                    st.altair_chart(chart_speed_time, use_container_width=True)

                with col_t2:
                    chart_vol_time = alt.Chart(agg_time).mark_bar(color="#48bb78", cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
                        x=alt.X("time_bin:N", title="Waktu Pengamatan (Jam:Menit)"),
                        y=alt.Y("Volume_Kendaraan_Unik:Q", title="Volume Kendaraan Unik (Unit)"),
                        tooltip=["time_bin:N", "Volume_Kendaraan_Unik:Q"]
                    ).properties(height=280, title="Volume Kendaraan Unik per Menit")
                    st.altair_chart(chart_vol_time, use_container_width=True)

                st.download_button(
                    label="📥 Unduh Data Tabel Gambar 3 (CSV)",
                    data=agg_time.to_csv(index=False).encode("utf-8"),
                    file_name="tabel_gambar_3_dinamika_temporal.csv",
                    mime="text/csv",
                    key="dl_tabel_g3",
                )
            else:
                st.info("Data rentang waktu pengamatan belum mencukupi untuk visualisasi runtun waktu.")
        else:
            st.info("Kolom datetime_iso tidak ditemukan pada dataset.")

        st.markdown(
            "> *Gambar 3: Fluktuasi Temporal Kecepatan Rerata (km/jam) dan Kepadatan Volume Lalu Lintas Sepanjang Periode Pengamatan.*"
        )

    # =========================================================================
    # GAMBAR 4: ANALISIS KEPATUHAN HELM PENGENDARA RODA DUA (ETLE)
    # =========================================================================
    with subtab_g4:
        st.markdown("#### 🪖 Gambar 4: Analisis Kepatuhan Penggunaan Helm Pengendara Sepeda Motor (ETLE: Gol VI-A vs VI-B)")
        st.caption("Mengevaluasi efektivitas Penegakan Hukum Elektronik (ETLE) dan korelasi antara kepatuhan helm dan kecepatan berkendara.")

        motor_df = df_telemetry[df_telemetry["golongan"].isin(["Golongan VI-A", "Golongan VI-B"])]
        if not motor_df.empty:
            m_taat = motor_df[motor_df["golongan"] == "Golongan VI-A"]["track_id"].nunique()
            m_langgar = motor_df[motor_df["golongan"] == "Golongan VI-B"]["track_id"].nunique()
            m_total = m_taat + m_langgar
            taat_pct = (m_taat / max(1, m_total) * 100.0) if m_total > 0 else 0.0
            langgar_pct = 100.0 - taat_pct

            spd_taat = motor_df[(motor_df["golongan"] == "Golongan VI-A") & (motor_df["speed_kmh"] > 0)]["speed_kmh"].mean() or 0.0
            spd_langgar = motor_df[(motor_df["golongan"] == "Golongan VI-B") & (motor_df["speed_kmh"] > 0)]["speed_kmh"].mean() or 0.0

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("🏍️ Total Pengendara Motor", f"{m_total:,} Unit")
            col_m2.metric("✅ Kepatuhan Helm SNI", f"{taat_pct:.1f}%", f"{m_taat} Pengendara Taat")
            col_m3.metric("⚠️ Pelanggaran Tanpa Helm", f"{langgar_pct:.1f}%", f"{m_langgar} Pengendara Melanggar", delta_color="inverse")

            # Dataframe ringkasan
            helm_summary = pd.DataFrame([
                {"Status_Kepatuhan": "Golongan VI-A: Taat Helm SNI", "Jumlah_Pengendara": m_taat, "Persentase": round(taat_pct, 1), "Rerata_Kecepatan_kmh": round(spd_taat, 1)},
                {"Status_Kepatuhan": "Golongan VI-B: Melanggar / Tanpa Helm", "Jumlah_Pengendara": m_langgar, "Persentase": round(langgar_pct, 1), "Rerata_Kecepatan_kmh": round(spd_langgar, 1)},
            ])

            col_hg1, col_hg2 = st.columns(2)
            with col_hg1:
                chart_helm_bar = alt.Chart(helm_summary).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Status_Kepatuhan:N", title="Kategori Kepatuhan"),
                    y=alt.Y("Jumlah_Pengendara:Q", title="Jumlah Pengendara Motor (Unit)"),
                    color=alt.Color("Status_Kepatuhan:N", scale=alt.Scale(
                        domain=["Golongan VI-A: Taat Helm SNI", "Golongan VI-B: Melanggar / Tanpa Helm"],
                        range=["#38a169", "#e53e3e"]
                    ), legend=None),
                    tooltip=["Status_Kepatuhan:N", "Jumlah_Pengendara:Q", "Persentase:Q"]
                ).properties(height=280, title="Perbandingan Jumlah Pengendara")
                st.altair_chart(chart_helm_bar, use_container_width=True)

            with col_hg2:
                chart_helm_spd = alt.Chart(helm_summary).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("Status_Kepatuhan:N", title="Kategori Kepatuhan"),
                    y=alt.Y("Rerata_Kecepatan_kmh:Q", title="Rerata Kecepatan (km/jam)"),
                    color=alt.Color("Status_Kepatuhan:N", scale=alt.Scale(
                        domain=["Golongan VI-A: Taat Helm SNI", "Golongan VI-B: Melanggar / Tanpa Helm"],
                        range=["#48bb78", "#f56565"]
                    ), legend=None),
                    tooltip=["Status_Kepatuhan:N", "Rerata_Kecepatan_kmh:Q"]
                ).properties(height=280, title="Perbandingan Profil Kecepatan (km/jam)")
                st.altair_chart(chart_helm_spd, use_container_width=True)

            st.download_button(
                label="📥 Unduh Data Tabel Gambar 4 (CSV)",
                data=helm_summary.to_csv(index=False).encode("utf-8"),
                file_name="tabel_gambar_4_kepatuhan_helm.csv",
                mime="text/csv",
                key="dl_tabel_g4",
            )
        else:
            st.info("Belum ada data sepeda motor yang terdeteksi.")

        st.markdown(
            "> *Gambar 4: Analisis Efektivitas Penegakan Hukum Elektronik (ETLE): Perbandingan Proporsi Kepatuhan Penggunaan Helm SNI dan Profil Kecepatan Pengendara.*"
        )

    # =========================================================================
    # GAMBAR 5: VALIDASI GEOMETRI DIMENSI FISIK KENDARAAN (HOMOGRAFI)
    # =========================================================================
    with subtab_g5:
        st.markdown("#### 📐 Gambar 5: Diagram Sebar (*Scatter Plot*) Validasi Kalibrasi Dimensi Fisik Kendaraan")
        st.caption("Memvalidasi pemisahan klaster dimensi metrik (Panjang vs Lebar dalam satuan meter) hasil proyeksi Homografi Planar kamera.")

        dim_df = df_telemetry[(df_telemetry["length_meter"] > 0.5) & (df_telemetry["width_meter"] > 0.3) & (df_telemetry["length_meter"] < 22.0)].copy()

        if len(dim_df) >= 10:
            # Batasi sampel maksimal 500 titik untuk performa rendering ringan
            dim_plot_data = dim_df.sample(min(500, len(dim_df)), random_state=42)

            scatter_chart = alt.Chart(dim_plot_data).mark_circle(size=60, opacity=0.7).encode(
                x=alt.X("length_meter:Q", title="Panjang Kendaraan Terproyeksi (Meter)"),
                y=alt.Y("width_meter:Q", title="Lebar Kendaraan Terproyeksi (Meter)"),
                color=alt.Color("golongan:N", scale=alt.Scale(
                    domain=list(GOLONGAN_HEX_COLORS.keys()),
                    range=list(GOLONGAN_HEX_COLORS.values())
                ), title="Golongan Kendaraan"),
                tooltip=[
                    alt.Tooltip("track_id:N", title="ID Kendaraan"),
                    alt.Tooltip("golongan:N", title="Golongan"),
                    alt.Tooltip("vehicle_subclass:N", title="Subkelas"),
                    alt.Tooltip("length_meter:Q", title="Panjang (m)", format=".2f"),
                    alt.Tooltip("width_meter:Q", title="Lebar (m)", format=".2f"),
                    alt.Tooltip("speed_kmh:Q", title="Kecepatan (km/h)", format=".1f"),
                ]
            ).properties(height=380)

            st.altair_chart(scatter_chart, use_container_width=True)

            # Rerata dimensi per golongan
            dim_summary = dim_df.groupby("golongan").agg(
                Panjang_Rerata_m=("length_meter", "mean"),
                Lebar_Rerata_m=("width_meter", "mean"),
                Rasio_Aspek_Rerata=("aspect_ratio", "mean"),
                Total_Sampel=("track_id", "count")
            ).round(2).reset_index()

            st.markdown("**Tabel 5: Rerata Dimensi Fisik Kendaraan Terkalibrasi (Meter)**")
            st.dataframe(dim_summary, use_container_width=True)

            st.download_button(
                label="📥 Unduh Data Tabel Gambar 5 (CSV)",
                data=dim_summary.to_csv(index=False).encode("utf-8"),
                file_name="tabel_gambar_5_dimensi_fisik_homografi.csv",
                mime="text/csv",
                key="dl_tabel_g5",
            )
        else:
            st.info("Sampel dimensi kendaraan belum mencukupi.")

        st.markdown(
            "> *Gambar 5: Diagram Sebar (Scatter Plot) Validasi Kalibrasi Dimensi Geometris Fisik Kendaraan (Panjang vs Lebar dalam Meter) Melalui Transformasi Homografi Bidang Datar.*"
        )

    # =========================================================================
    # GAMBAR 6: TAKSONOMI DISTRIBUSI FREKUENSI KEJADIAN INSIDEN & ANOMALI
    # =========================================================================
    with subtab_g6:
        st.markdown("#### 🚨 Gambar 6: Taksonomi Distribusi Frekuensi Kejadian Insiden Lalu Lintas dan Tingkat Keparahan")
        st.caption("Klasifikasi anomali kritis: Tabrakan (Collision), Sepeda Motor Terjatuh (Spill), Kendaraan Mogok (Stopped), dan Melawan Arah Barrier (Wrong-Way).")

        if not df_incidents.empty and "incident_type" in df_incidents.columns:
            inc_agg = df_incidents.groupby(["incident_type", "severity"]).size().reset_index(name="Frekuensi_Kejadian")

            inc_chart = alt.Chart(inc_agg).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
                y=alt.Y("incident_type:N", title="Jenis Insiden / Anomali", sort="-x"),
                x=alt.X("Frekuensi_Kejadian:Q", title="Total Kejadian Terdeteksi"),
                color=alt.Color("severity:N", scale=alt.Scale(
                    domain=["CRITICAL", "WARNING"],
                    range=["#e53e3e", "#dd6b20"]
                ), title="Tingkat Keparahan"),
                tooltip=["incident_type:N", "severity:N", "Frekuensi_Kejadian:Q"]
            ).properties(height=280)

            st.altair_chart(inc_chart, use_container_width=True)

            col_ic1, col_ic2 = st.columns(2)
            total_inc = len(df_incidents)
            crit_count = len(df_incidents[df_incidents["severity"] == "CRITICAL"])
            warn_count = total_inc - crit_count

            col_ic1.metric("🚨 Total Insiden Terdeteksi", f"{total_inc:,} Kejadian")
            col_ic2.metric("🔴 Insiden Kritikal (CRITICAL)", f"{crit_count} ({crit_count/max(1, total_inc)*100:.1f}%)", f"{warn_count} Peringatan (WARNING)", delta_color="inverse")

            st.download_button(
                label="📥 Unduh Data Tabel Gambar 6 (CSV)",
                data=inc_agg.to_csv(index=False).encode("utf-8"),
                file_name="tabel_gambar_6_taksonomi_insiden.csv",
                mime="text/csv",
                key="dl_tabel_g6",
            )
        else:
            st.info("Belum ada catatan insiden lalu lintas pada rentang waktu yang dipilih.")

        st.markdown(
            "> *Gambar 6: Taksonomi Distribusi Frekuensi Kejadian Insiden Lalu Lintas dan Tingkat Keparahan pada Ruas Jalan Terowongan.*"
        )
