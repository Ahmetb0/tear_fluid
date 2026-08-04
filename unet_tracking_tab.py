"""Streamlit tab: U-Net tear-film particle tracking."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st
import torch

from track_particles import (
    DEFAULT_MODEL_OUT,
    THRESHOLD,
    enrich_tracking_dataframe,
    load_model,
    process_video_file,
    trackpoints_to_dataframe,
)
from ui_video import UPLOAD_DIR
from medical_report import compute_medical_report
from medical_report_ui import render_medical_report_section

PROJECT_ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner="U-Net modeli yükleniyor...")
def get_unet_model(model_path: str):
    """Load and cache the U-Net model (GPU if available)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(Path(model_path), device)
    return model, device


def render_unet_tracking_tab() -> None:
    st.header("🧬 U-Net Gözyaşı Takibi")
    st.markdown(
        "Sol panelden yüklenen videoda U-Net segmentasyonu + parçacık takibi yapılır. "
        "**Blink Detection** ile aynı `safe frame` listesi burada da kullanılabilir."
    )

    model_path = PROJECT_ROOT / "unet_tear_film.pth"
    if not model_path.exists():
        st.error(f"Model bulunamadı: `{model_path}` — önce `python train_unet.py` çalıştırın.")
        return

    if not st.session_state.get("video_loaded"):
        st.warning("⚠️ Sol panelden video yükleyip **Load Video** butonuna basın.")
        if "unet_track_df" in st.session_state:
            _show_results(
                st.session_state.unet_track_df,
                st.session_state.get("unet_output_video"),
                st.session_state.get("unet_video_name", "tracked_output.mp4"),
                mm_per_pixel=st.session_state.get("unet_mm_per_pixel"),
                fps=st.session_state.get("unet_fps"),
                epochs=st.session_state.get("epochs"),
                fdm_enabled=st.session_state.get("fdm_enabled", False),
            )
        return

    st.success(f"📹 Aktif video: **{st.session_state.get('video_display_name', '—')}**")

    n_total = st.session_state.get("num_frames", 0)
    n_safe = len(st.session_state.get("safe_frames") or [])
    n_epochs = len(st.session_state.get("epochs") or [])
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.metric("Toplam kare", n_total)
    with col_s2:
        st.metric("Safe kare (blink filtresi)", n_safe)
    with col_s3:
        st.metric("Epoch", n_epochs)

    use_safe_frames = st.checkbox(
        "Sadece safe frame'leri işle (blink filtresi)",
        value=True,
        help="Kapalıysa videonun tüm kareleri işlenir (göz kırpma dönemleri dahil).",
    )
    if use_safe_frames and n_safe == 0:
        st.error("Safe frame yok — **Blink Detection** sekmesinde parametreleri ayarlayın.")
        return

    default_fps = float(st.session_state.config.fps or 30.0)
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        fps = st.number_input("FPS", min_value=1.0, max_value=120.0, value=default_fps, step=1.0)
    with col_b:
        threshold = st.slider("Maske eşiği", 0.05, 0.9, THRESHOLD, 0.05)
    with col_c:
        max_dist = st.slider("Max takip mesafesi (px)", 10, 150, 50, 5)

    mm_per_pixel = st.number_input(
        "mm/piksel (opsiyonel, mm/s için)",
        min_value=0.0,
        value=0.0,
        step=0.001,
        format="%.4f",
        help="0 bırakılırsa yalnızca px/s hesaplanır.",
    )
    mm_per_pixel_val = mm_per_pixel if mm_per_pixel > 0 else None

    frame_indices = st.session_state.safe_frames if use_safe_frames else None
    if frame_indices is not None:
        st.caption(f"İşlenecek kare sayısı: **{len(frame_indices)}** / {n_total}")

    if st.button("🚀 U-Net ile İşle ve Takip Et", type="primary"):
        video_path = Path(st.session_state.config.video_path)
        if not video_path.exists():
            st.error("Video dosyası bulunamadı. Sol panelden videoyu yeniden yükleyin.")
            return

        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        output_video = UPLOAD_DIR / f"{video_path.stem}_unet_tracked.mp4"

        model, device = get_unet_model(str(model_path))
        device_label = "CUDA 🟢" if device.type == "cuda" else "CPU 🟡"
        st.caption(f"Cihaz: **{device_label}** — {device}")

        progress = st.progress(0, text="Kareler işleniyor...")
        total_expected = len(frame_indices) if frame_indices else n_total

        def on_progress(current: int, total: int) -> None:
            progress.progress(
                min(current / max(total, 1), 1.0),
                text=f"Kare {current}/{total} işlendi...",
            )

        with st.spinner("Segmentasyon + takip devam ediyor..."):
            records = process_video_file(
                video_path=video_path,
                output_path=output_video,
                model=model,
                device=device,
                video_id=video_path.stem,
                fps=float(fps),
                threshold=float(threshold),
                max_track_distance=float(max_dist),
                mm_per_pixel=mm_per_pixel_val,
                progress_callback=on_progress,
                frame_indices=frame_indices,
            )

        df = trackpoints_to_dataframe(records)
        epochs = st.session_state.get("epochs")
        fps_val = float(fps)
        if epochs:
            df = enrich_tracking_dataframe(df, epochs=epochs, fps=fps_val)

        st.session_state.unet_track_df = df
        st.session_state.unet_output_video = output_video.read_bytes()
        st.session_state.unet_video_name = output_video.name
        st.session_state.unet_used_safe_frames = use_safe_frames
        st.session_state.unet_mm_per_pixel = mm_per_pixel_val
        st.session_state.unet_fps = fps_val

        progress.progress(1.0, text="Tamamlandı!")
        mode = "safe frame" if use_safe_frames else "tüm kare"
        st.success(
            f"İşlem bitti ({mode}) — {len(records)} kayıt, "
            f"{df['particle_id'].nunique()} parçacık ID."
        )

    if "unet_track_df" in st.session_state:
        if st.session_state.get("unet_used_safe_frames"):
            st.info("ℹ️ Son işlem **blink filtreli safe frame** modunda yapıldı.")
        _show_results(
            st.session_state.unet_track_df,
            st.session_state.get("unet_output_video"),
            st.session_state.get("unet_video_name", "tracked_output.mp4"),
            mm_per_pixel=st.session_state.get("unet_mm_per_pixel"),
            fps=st.session_state.get("unet_fps"),
            epochs=st.session_state.get("epochs"),
            fdm_enabled=st.session_state.get("fdm_enabled", False),
        )


def _show_results(
    df,
    video_bytes: bytes | None,
    video_name: str = "tracked_output.mp4",
    *,
    mm_per_pixel=None,
    fps=None,
    epochs=None,
    fdm_enabled: bool = False,
) -> None:
    st.markdown("---")
    st.subheader("🎬 Takip Videosu")
    if video_bytes:
        st.video(video_bytes)
    else:
        st.warning("Video çıktısı bulunamadı.")

    st.subheader("📈 Ortalama Akış Hızı")
    if df.empty:
        st.warning("Takip verisi yok.")
        return

    vel = df[df["velocity_px_per_sec"] > 0].copy()
    if vel.empty:
        st.info("Henüz hız hesaplanabilen eşleşme yok (ilk kareler hariç).")
    else:
        chart_df = vel.groupby("time_sec")["velocity_px_per_sec"].mean().reset_index()
        chart_df.columns = ["time_sec", "mean_velocity_px_per_sec"]
        st.line_chart(chart_df.set_index("time_sec"))

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Ort. hız (px/s)", f"{vel['velocity_px_per_sec'].mean():.2f}")
        with col2:
            st.metric("Maks. hız (px/s)", f"{vel['velocity_px_per_sec'].max():.2f}")
        with col3:
            if vel["velocity_mm_per_sec"].sum() > 0:
                st.metric("Ort. hız (mm/s)", f"{vel['velocity_mm_per_sec'].mean():.3f}")
            else:
                st.metric("Parçacık ID sayısı", str(df["particle_id"].nunique()))

        fig, ax = plt.subplots(figsize=(10, 4))
        for pid, grp in vel.groupby("particle_id"):
            ax.plot(grp["time_sec"], grp["velocity_px_per_sec"], alpha=0.5, linewidth=1)
        ax.set_xlabel("Zaman (s)")
        ax.set_ylabel("Hız (px/s)")
        ax.set_title("Parçacık bazlı hız — zaman")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

    if not df.empty and epochs and "time_since_blink_s" in df.columns:
        fps_val = float(fps or 30.0)
        medical_report = compute_medical_report(
            df,
            epochs=epochs,
            fps=fps_val,
            fdm_enabled=fdm_enabled,
            mm_per_pixel=mm_per_pixel,
        )
        render_medical_report_section(
            medical_report,
            mm_per_pixel=mm_per_pixel,
        )
    elif not df.empty and not epochs:
        st.caption(
            "Tıbbi rapor için sol panelden video yükleyip blink analizi yapın "
            "(epoch / time_since_blink_s gerekli)."
        )

    st.subheader("📋 Veri İndir")
    csv_data = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ CSV İndir (yörünge + hız)",
        data=csv_data,
        file_name="unet_particle_tracks.csv",
        mime="text/csv",
    )
    st.dataframe(df.head(100), width="stretch")
    if len(df) > 100:
        st.caption(f"Toplam {len(df)} satır — tablo ilk 100 satırı gösteriyor.")
