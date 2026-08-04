"""Streamlit UI helpers for tear-film medical report."""

from __future__ import annotations

from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import streamlit as st


def render_medical_report_section(
    report: Dict[str, Any],
    *,
    mm_per_pixel: Optional[float] = None,
    show_fit_plot: bool = True,
) -> None:
    """Render 'Tıbbi Tanı Raporu - Gözyaşı Dinamikleri' below velocity charts."""
    st.subheader("Tıbbi Tanı Raporu - Gözyaşı Dinamikleri")

    if report.get("fdm_enabled"):
        dur = report.get("fdm_duration_s", 1.0)
        st.caption(
            f"**FDM aktif:** Yalnızca her göz kırpmasından sonraki ilk **{dur:.1f} s** "
            f"(t=0 = ilk net kare) kullanıldı."
        )
    else:
        st.caption(
            "Standart mod: Tüm post-blink epoch verileri birleştirildi. "
            "FDM için sol paneldeki kutuyu işaretleyin."
        )

    if not report.get("fit_success"):
        st.warning(
            f"Güç yasası eğrisi uydurulamadı. "
            f"{report.get('error') or 'Yetersiz veya uyumsuz veri.'}"
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("alpha (Ölçekleme Faktörü)", "—")
        with col2:
            st.metric("beta (Sönümleme Faktörü)", "—")
        with col3:
            st.metric("eMMSi (0.1. Sn Başlangıç Hızı)", "—")
        with col4:
            st.metric("eMMSf (2.0. Sn Stabilizasyon Hızı)", "—")
        return

    unit = report.get("velocity_unit", "px/s")
    if unit == "MMS":
        speed_suffix = " MMS"
    else:
        speed_suffix = f" {unit}"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("alpha (Ölçekleme Faktörü)", f"{report['alpha']:.4f}")
    with col2:
        st.metric("beta (Sönümleme Faktörü)", f"{report['beta']:.4f}")
    with col3:
        st.metric(
            f"eMMSi (0.1. Sn Başlangıç Hızı)",
            f"{report['eMMSi']:.4f}{speed_suffix}",
        )
    with col4:
        st.metric(
            f"eMMSf (2.0. Sn Stabilizasyon Hızı)",
            f"{report['eMMSf']:.4f}{speed_suffix}",
        )

    if mm_per_pixel and mm_per_pixel > 0 and unit == "px/s":
        st.caption(
            f"Kalibrasyon: **{mm_per_pixel:.4f} mm/piksel** — "
            f"hız birimi px/s (mm/s için U-Net sekmesinde kalibrasyon girin)."
        )

    st.caption(
        f"Model: **MMS(t) = α × t^(-β)** | R² = {report['r_squared']:.4f} | "
        f"{report.get('equation', '')}"
    )

    if show_fit_plot and report.get("binned_time") is not None:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(
            report["binned_time"],
            report["binned_velocity"],
            s=60,
            alpha=0.65,
            color="steelblue",
            label="Binned median MMS",
            zorder=3,
        )
        ax.plot(
            report["binned_time"],
            report["fitted_curve"],
            linewidth=2.5,
            color="crimson",
            label=report.get("equation", "Fit"),
            zorder=4,
        )
        ax.set_xlabel("Blink sonrası süre t (s)")
        ylabel = f"MMS / Hız ({unit})"
        ax.set_ylabel(ylabel)
        title = "Güç Yasası Uyumu: MMS(t) = α × t^(-β)"
        if report.get("fdm_enabled"):
            title += f" [FDM ≤ {report.get('fdm_duration_s', 1.0):.1f}s]"
        ax.set_title(title)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="upper right")
        st.pyplot(fig)
        plt.close(fig)

    if report.get("eMMSi") and report.get("eMMSf") and report["eMMSf"] > 0:
        ratio = report["eMMSi"] / report["eMMSf"]
        st.info(
            f"**Klinik özet:** eMMSi/eMMSf = **{ratio:.2f}×** "
            f"(başlangıç hızının stabilizasyon hızına oranı)."
        )
