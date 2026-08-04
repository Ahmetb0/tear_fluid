"""
Medical tear-film dynamics report (literature-based).

Reference model: MMS(t) = alpha * t^(-beta)
FDM: Fixed-Duration Model — first 1 s after each blink (t=0 = first clear frame).
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

FDM_DEFAULT_DURATION_S = 1.0
EMMSI_TIME_S = 0.1
EMMSF_TIME_S = 2.0


def power_law(t: np.ndarray | float, alpha: float, beta: float) -> np.ndarray | float:
    """MMS(t) = alpha * t^(-beta)."""
    return alpha * np.power(t, -beta)


def compute_emmsi_emmsf(alpha: float, beta: float) -> tuple[float, float]:
    """Estimated initial (t=0.1 s) and stabilization (t=2.0 s) speeds."""
    return float(power_law(EMMSI_TIME_S, alpha, beta)), float(power_law(EMMSF_TIME_S, alpha, beta))


def _epoch_bounds(epoch: Any) -> tuple[int, int]:
    if hasattr(epoch, "start_frame"):
        return int(epoch.start_frame), int(epoch.end_frame)
    return int(epoch["start_frame"]), int(epoch["end_frame"])


def enrich_tracks_with_blink_timing(
    df: "pd.DataFrame",
    epochs: Sequence[Any],
    fps: float,
    frame_col: str = "frame_number",
) -> "pd.DataFrame":
    """
    Add epoch-relative timing columns for U-Net (or generic) tracking CSVs.

    t=0 at the first clear frame after each blink (epoch start).
    """
    import pandas as pd

    if df.empty or not epochs:
        out = df.copy()
        out["time_since_blink_s"] = np.nan
        out["epoch"] = -1
        out["include_in_power_law_fit"] = False
        return out

    out = df.copy()
    out["time_since_blink_s"] = np.nan
    out["epoch"] = -1
    out["include_in_power_law_fit"] = False

    for epoch_idx, epoch in enumerate(epochs):
        start, end = _epoch_bounds(epoch)
        mask = (out[frame_col] >= start) & (out[frame_col] < end)
        if not mask.any():
            continue
        out.loc[mask, "time_since_blink_s"] = (out.loc[mask, frame_col] - start) / fps
        out.loc[mask, "epoch"] = epoch_idx
        if start > 0:
            out.loc[mask, "include_in_power_law_fit"] = True

    return out


def apply_fdm_filter(
    df: "pd.DataFrame",
    *,
    time_col: str = "time_since_blink_s",
    duration_s: float = FDM_DEFAULT_DURATION_S,
) -> "pd.DataFrame":
    """Keep only rows within the first *duration_s* seconds after each blink."""
    if df.empty or time_col not in df.columns:
        return df.copy()
    valid = df[time_col].notna()
    window = (df[time_col] >= 0) & (df[time_col] <= duration_s)
    return df[valid & window].copy()


def _prepare_fit_source(
    df: "pd.DataFrame",
    *,
    time_col: str,
    velocity_col: str,
    fdm_enabled: bool,
    fdm_duration_s: float,
) -> "pd.DataFrame":
    import pandas as pd

    fit_source = df.copy()
    if "include_in_power_law_fit" in fit_source.columns:
        fit_source = fit_source[fit_source["include_in_power_law_fit"].astype(bool)]

    if fdm_enabled:
        fit_source = apply_fdm_filter(
            fit_source, time_col=time_col, duration_s=fdm_duration_s
        )

    return fit_source


def compute_power_law_biomarkers(
    df: "pd.DataFrame",
    *,
    time_col: str = "time_since_blink_s",
    velocity_col: str = "mms_velocity",
    bin_size: float = 0.15,
    fdm_enabled: bool = False,
    fdm_duration_s: float = FDM_DEFAULT_DURATION_S,
    mm_per_pixel: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Fit MMS(t) = alpha * t^(-beta) and compute alpha, beta, eMMSi, eMMSf.

    Returns a dict with fit_success=False and empty biomarkers on failure.
    """
    import pandas as pd
    from scipy.optimize import curve_fit

    empty: Dict[str, Any] = {
        "fit_success": False,
        "alpha": None,
        "beta": None,
        "eMMSi": None,
        "eMMSf": None,
        "r_squared": None,
        "equation": None,
        "binned_time": None,
        "binned_velocity": None,
        "fitted_curve": None,
        "num_bins": 0,
        "bin_size": bin_size,
        "fdm_enabled": fdm_enabled,
        "fdm_duration_s": fdm_duration_s if fdm_enabled else None,
        "velocity_unit": _velocity_unit_label(velocity_col, mm_per_pixel),
        "error": None,
    }

    if df is None or df.empty:
        empty["error"] = "Veri yok"
        return empty

    if time_col not in df.columns:
        empty["error"] = f"'{time_col}' sütunu bulunamadı"
        return empty
    if velocity_col not in df.columns:
        empty["error"] = f"'{velocity_col}' sütunu bulunamadı"
        return empty

    fit_source = _prepare_fit_source(
        df,
        time_col=time_col,
        velocity_col=velocity_col,
        fdm_enabled=fdm_enabled,
        fdm_duration_s=fdm_duration_s,
    )

    valid_data = fit_source[
        (fit_source[time_col] > 0) & (fit_source[velocity_col] > 0)
    ].copy()

    if len(valid_data) == 0:
        empty["error"] = "Güç yasası için geçerli hız verisi yok"
        return empty

    max_time = valid_data[time_col].max()
    bins = np.arange(0, max_time + bin_size, bin_size)
    valid_data["time_bin"] = pd.cut(valid_data[time_col], bins=bins)

    binned_stats = valid_data.groupby("time_bin", observed=True)[velocity_col].agg(
        ["median", "mean", "count"]
    )
    binned_stats = binned_stats[binned_stats["count"] >= 3]

    if len(binned_stats) < 4:
        empty["error"] = "Eğri uydurma için yeterli zaman dilimi yok (≥4 bin, bin başına ≥3 nokta)"
        return empty

    bin_centers = np.array([interval.mid for interval in binned_stats.index])
    median_velocities = binned_stats["median"].values

    valid_mask = (
        (~np.isnan(bin_centers))
        & (~np.isnan(median_velocities))
        & (bin_centers > 0)
        & (median_velocities > 0)
    )
    bin_centers = bin_centers[valid_mask]
    median_velocities = median_velocities[valid_mask]

    if len(bin_centers) < 4:
        empty["error"] = "Filtreleme sonrası yeterli bin kalmadı"
        return empty

    try:
        initial_guess = [float(median_velocities[0]), 0.5]
        params, _ = curve_fit(
            power_law,
            bin_centers,
            median_velocities,
            p0=initial_guess,
            bounds=([0.01, 0.01], [1e6, 3.0]),
            maxfev=5000,
        )
        alpha, beta = float(params[0]), float(params[1])
        fitted_velocities = power_law(bin_centers, alpha, beta)

        ss_res = np.sum((median_velocities - fitted_velocities) ** 2)
        ss_tot = np.sum((median_velocities - np.mean(median_velocities)) ** 2)
        r_squared = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

        eMMSi, eMMSf = compute_emmsi_emmsf(alpha, beta)
        if mm_per_pixel and mm_per_pixel > 0 and velocity_col.endswith("px_per_sec"):
            eMMSi *= mm_per_pixel
            eMMSf *= mm_per_pixel

        return {
            "fit_success": True,
            "alpha": alpha,
            "beta": beta,
            "eMMSi": eMMSi,
            "eMMSf": eMMSf,
            "r_squared": r_squared,
            "equation": f"MMS = {alpha:.3f} × t^(-{beta:.3f})",
            "binned_time": bin_centers,
            "binned_velocity": median_velocities,
            "fitted_curve": fitted_velocities,
            "num_bins": len(bin_centers),
            "bin_size": bin_size,
            "fdm_enabled": fdm_enabled,
            "fdm_duration_s": fdm_duration_s if fdm_enabled else None,
            "velocity_unit": _velocity_unit_label(velocity_col, mm_per_pixel),
            "error": None,
        }
    except Exception as exc:
        empty["error"] = str(exc)
        return empty


def _velocity_unit_label(velocity_col: str, mm_per_pixel: Optional[float]) -> str:
    if velocity_col == "mms_velocity":
        return "MMS"
    if velocity_col == "velocity_mm_per_sec" or (
        mm_per_pixel and mm_per_pixel > 0 and "px" in velocity_col
    ):
        return "mm/s"
    return "px/s"


def resolve_velocity_column(
    df: "pd.DataFrame",
    mm_per_pixel: Optional[float] = None,
) -> str:
    """Pick velocity column for biomarker fitting."""
    if "mms_velocity" in df.columns:
        return "mms_velocity"
    if mm_per_pixel and mm_per_pixel > 0 and "velocity_mm_per_sec" in df.columns:
        if df["velocity_mm_per_sec"].fillna(0).sum() > 0:
            return "velocity_mm_per_sec"
    if "velocity_px_per_sec" in df.columns:
        return "velocity_px_per_sec"
    raise ValueError("Uygun hız sütunu bulunamadı")


def build_analysis_dataframe(
    df: "pd.DataFrame",
    *,
    epochs: Optional[Sequence[Any]] = None,
    fps: float = 30.0,
) -> "pd.DataFrame":
    """Ensure blink-relative columns exist (enrich U-Net tracks when needed)."""
    if "time_since_blink_s" in df.columns:
        return df
    if epochs:
        return enrich_tracks_with_blink_timing(df, epochs, fps)
    return df


def compute_medical_report(
    df: "pd.DataFrame",
    *,
    epochs: Optional[Sequence[Any]] = None,
    fps: float = 30.0,
    fdm_enabled: bool = False,
    fdm_duration_s: float = FDM_DEFAULT_DURATION_S,
    bin_size: float = 0.15,
    mm_per_pixel: Optional[float] = None,
) -> Dict[str, Any]:
    """End-to-end biomarker computation for classic or U-Net tracking data."""
    work = build_analysis_dataframe(df, epochs=epochs, fps=fps)
    velocity_col = resolve_velocity_column(work, mm_per_pixel)
    return compute_power_law_biomarkers(
        work,
        time_col="time_since_blink_s",
        velocity_col=velocity_col,
        bin_size=bin_size,
        fdm_enabled=fdm_enabled,
        fdm_duration_s=fdm_duration_s,
        mm_per_pixel=mm_per_pixel,
    )
