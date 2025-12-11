# climate_analyzer.py

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ========= CONFIGURABLE THRESHOLDS (EDIT FOR YOUR CROP) ========= #

THRESHOLDS = {
    "VPD": {
        "stress_high": 1.5,    # kPa
        "optimal_low": 0.5,    # kPa
        "spike_delta": 0.2     # kPa
    },
    "PPFD": {
        "stress_high": 1000.0, # µmol m⁻² s⁻¹ (very high PPFD -> photoinhibition risk)
        "optimal_low": 200.0,  # µmol m⁻² s⁻¹
        "spike_delta": 200.0   # µmol m⁻² s⁻¹
    },
    "Temperature": {
        "stress_high": 30.0,   # °C
        "optimal_low": 18.0,   # °C
        "spike_delta": 2.0     # °C
    }
}

# Stress window detection parameters (hours)
STRESS_WINDOWS = {
    "high_vpd_min_run": 6,   # consecutive hours of high VPD to flag tipburn risk
    "low_vpd_min_run": 12,   # consecutive hours of low VPD to flag glassiness/humidity risk
    "high_ppfd_min_run": 4   # consecutive hours of very high PPFD to flag photoinhibition risk
}


# ----------------- Core analysis functions (logic) ----------------- #

def analyze_series(series: pd.Series, var_label: str, config: dict):
    """Compute variability and stress metrics for a 1D time series."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    n = len(s)
    if n == 0:
        return {"error": f"No numeric data for {var_label}."}

    mean = s.mean()
    sd = s.std()
    cv = sd / mean if mean != 0 else np.nan

    stress_high = config.get("stress_high", None)
    optimal_low = config.get("optimal_low", None)
    spike_delta = config.get("spike_delta", None)

    # Hours above stress threshold
    if stress_high is not None:
        hours_above_stress = int((s > stress_high).sum())
        pct_above_stress = 100.0 * hours_above_stress / n
    else:
        hours_above_stress = np.nan
        pct_above_stress = np.nan

    # Hours below optimal lower bound
    if optimal_low is not None:
        hours_below_optimal = int((s < optimal_low).sum())
        pct_below_optimal = 100.0 * hours_below_optimal / n
    else:
        hours_below_optimal = np.nan
        pct_below_optimal = np.nan

    if n > 1:
        diff = s.diff().dropna()
        mean_abs_delta = diff.abs().mean()
        max_abs_delta = diff.abs().max()
        try:
            lag1_autocorr = s.autocorr(lag=1)
        except Exception:
            lag1_autocorr = np.nan

        if spike_delta is not None:
            spike_mask = diff.abs() > spike_delta
            n_spikes = int(spike_mask.sum())
            max_spike = diff.abs().max()
        else:
            n_spikes = np.nan
            max_spike = np.nan
    else:
        mean_abs_delta = np.nan
        max_abs_delta = np.nan
        lag1_autocorr = np.nan
        n_spikes = np.nan
        max_spike = np.nan

    return {
        "n_hours": n,
        "mean": mean,
        "sd": sd,
        "cv": cv,
        "hours_above_stress": hours_above_stress,
        "pct_above_stress": pct_above_stress,
        "hours_below_optimal": hours_below_optimal,
        "pct_below_optimal": pct_below_optimal,
        "lag1_autocorr": lag1_autocorr,
        "mean_abs_delta": mean_abs_delta,
        "max_abs_delta": max_abs_delta,
        "n_spikes": n_spikes,
        "max_spike": max_spike,
    }


def compute_dli_metrics(ppfd_series: pd.Series, timestamp: pd.Series | None):
    """
    Compute daily light integral (DLI) metrics from PPFD.
    Assumes PPFD is in µmol m⁻² s⁻¹ and samples are hourly.
    DLI_day = sum(PPFD_hour * 3600) / 1e6 [mol m⁻² d⁻¹]
    """
    s = pd.to_numeric(ppfd_series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean_dli": np.nan, "sd_dli": np.nan, "n_days": 0}

    # If we have timestamps, group by date; otherwise approximate by 24-hour blocks.
    if timestamp is not None and not timestamp.isna().all():
        ts_valid = timestamp.loc[s.index]
        df = pd.DataFrame({"ts": ts_valid, "ppfd": s}).dropna(subset=["ts", "ppfd"])
        if df.empty:
            return {"mean_dli": np.nan, "sd_dli": np.nan, "n_days": 0}
        df["date"] = df["ts"].dt.date
        daily = df.groupby("date")["ppfd"].sum()
        dli_per_day = daily * 3600.0 / 1e6
        return {
            "mean_dli": float(dli_per_day.mean()),
            "sd_dli": float(dli_per_day.std()),
            "n_days": int(dli_per_day.shape[0]),
        }
    else:
        # No timestamps: assume hourly data and approximate days by 24 measurements.
        n = len(s)
        if n >= 24:
            day_index = np.floor(np.arange(n) / 24).astype(int)
            df = pd.DataFrame({"day": day_index, "ppfd": s.values})
            daily = df.groupby("day")["ppfd"].sum()
            dli_per_day = daily * 3600.0 / 1e6
            return {
                "mean_dli": float(dli_per_day.mean()),
                "sd_dli": float(dli_per_day.std()),
                "n_days": int(dli_per_day.shape[0]),
            }
        else:
            # Short segment: treat as representative "day" using mean PPFD
            dli_est = float(s.mean() * 3600.0 * 24.0 / 1e6)
            return {"mean_dli": dli_est, "sd_dli": 0.0, "n_days": 1}


def _find_segments(cond: pd.Series, min_run: int, ts: pd.Series | None):
    """
    Find continuous segments where cond==True for at least min_run consecutive points.
    Returns list of (start_x, end_x) where x is timestamp or index.
    """
    if cond.empty:
        return []

    cond = cond.fillna(False).to_numpy()
    n = len(cond)
    segments = []
    start = None

    for i in range(n):
        if cond[i]:
            if start is None:
                start = i
        else:
            if start is not None:
                if i - start >= min_run:
                    segments.append((start, i - 1))
                start = None

    # Handle run till end
    if start is not None and n - start >= min_run:
        segments.append((start, n - 1))

    if ts is not None:
        x = ts.reset_index(drop=True)
        return [(x[s], x[e]) for (s, e) in segments]
    else:
        # Use index positions
        return segments


def detect_stress_segments(df_work: pd.DataFrame, ts: pd.Series | None, vpd_col: str, ppfd_col: str):
    """
    Detect stress segments for:
    - High VPD (tipburn risk)
    - Low VPD (glassiness / disease risk)
    - Very high PPFD (photoinhibition risk)
    Returns a list of dicts: {"start": ..., "end": ..., "label": ..., "color": ...}
    """
    segments = []

    vpd_cfg = THRESHOLDS["VPD"]
    ppfd_cfg = THRESHOLDS["PPFD"]

    vpd = pd.to_numeric(df_work[vpd_col], errors="coerce")
    ppfd = pd.to_numeric(df_work[ppfd_col], errors="coerce")

    # High VPD stress (tipburn risk)
    high_vpd_cond = vpd > vpd_cfg["stress_high"]
    high_vpd_runs = _find_segments(high_vpd_cond, STRESS_WINDOWS["high_vpd_min_run"], ts)
    for start, end in high_vpd_runs:
        segments.append({
            "start": start,
            "end": end,
            "label": "High VPD stress (tipburn risk)",
            "color": "red"
        })

    # Low VPD / very humid (glassiness / disease risk)
    low_vpd_cond = vpd < vpd_cfg["optimal_low"]
    low_vpd_runs = _find_segments(low_vpd_cond, STRESS_WINDOWS["low_vpd_min_run"], ts)
    for start, end in low_vpd_runs:
        segments.append({
            "start": start,
            "end": end,
            "label": "Low VPD / humid (glassiness risk)",
            "color": "blue"
        })

    # Very high PPFD (photoinhibition risk)
    high_ppfd_cond = ppfd > ppfd_cfg["stress_high"]
    high_ppfd_runs = _find_segments(high_ppfd_cond, STRESS_WINDOWS["high_ppfd_min_run"], ts)
    for start, end in high_ppfd_runs:
        segments.append({
            "start": start,
            "end": end,
            "label": "Very high PPFD (photoinhibition risk)",
            "color": "orange"
        })

    return segments


def classify_environment(vpd: dict, ppfd: dict, temp: dict, dli: dict):
    """
    Classify environment into gear (by DLI) and stability (by VPD + temp).
    Returns:
      - env_label: e.g. "Mid gear, stable" / "High gear, unstable"
      - tags: list of descriptive strings
    """
    tags = []

    # ---------- GEAR: based on DLI ----------
    dli_mean = dli.get("mean_dli", np.nan)
    if pd.isna(dli_mean):
        gear = "?"
        gear_tag = "DLI unknown (insufficient data)"
    elif dli_mean < 14:
        gear = "L"
        gear_tag = f"Low gear DLI (<14): {dli_mean:.1f} mol/m²/day"
    elif dli_mean <= 24:
        gear = "M"
        gear_tag = f"Mid gear DLI (14–24): {dli_mean:.1f} mol/m²/day"
    else:
        gear = "H"
        gear_tag = f"High gear DLI (>24): {dli_mean:.1f} mol/m²/day"
    tags.append(gear_tag)

    # ---------- VPD behaviour tags ----------
    pct_above = vpd.get("pct_above_stress", np.nan)
    pct_below = vpd.get("pct_below_optimal", np.nan)
    vpd_cv = vpd.get("cv", np.nan)
    vpd_spikes = vpd.get("n_spikes", 0) if vpd.get("n_spikes", None) is not None else 0

    if not pd.isna(pct_above) and not pd.isna(pct_below):
        pct_target = max(0.0, 100.0 - pct_above - pct_below)
    else:
        pct_target = np.nan

    if not pd.isna(pct_above) and pct_above > 15:
        tags.append("Frequent high VPD stress")
    elif not pd.isna(pct_below) and pct_below > 40:
        tags.append("Mostly low VPD / humid")
    else:
        tags.append("VPD mostly in target range")

    if not pd.isna(vpd_cv) and (vpd_cv > 0.7 or vpd_spikes > 20):
        tags.append("VPD highly variable")
    elif not pd.isna(vpd_cv) and vpd_cv < 0.4:
        tags.append("VPD relatively stable")

    # ---------- Light level tags ----------
    ppfd_mean = ppfd.get("mean", np.nan)
    if not pd.isna(ppfd_mean):
        if ppfd_mean > 600:
            tags.append("High average PPFD")
        elif ppfd_mean < 250:
            tags.append("Low average PPFD")
        else:
            tags.append("Moderate average PPFD")

    # ---------- Temperature behaviour tags ----------
    t_mean = temp.get("mean", np.nan)
    t_cv = temp.get("cv", np.nan)
    t_spikes = temp.get("n_spikes", 0) if temp.get("n_spikes", None) is not None else 0

    if not pd.isna(t_mean):
        if t_mean > 27:
            tags.append("Warm to hot temperatures")
        elif t_mean < 20:
            tags.append("Cool temperatures")
        else:
            tags.append("Moderate temperatures")

    if not pd.isna(t_cv) and t_cv > 0.06:
        tags.append("Temperature quite variable")
    else:
        tags.append("Temperature fairly stable")

    # ---------- STABILITY flag (VPD + Temp) ----------
    vpd_stable = True
    if not pd.isna(pct_target) and pct_target < 60:
        vpd_stable = False
    if not pd.isna(pct_above) and pct_above > 15:
        vpd_stable = False
    if not pd.isna(vpd_cv) and vpd_cv > 0.6:
        vpd_stable = False
    if vpd_spikes is not None and vpd_spikes > 30:
        vpd_stable = False

    temp_stable = True
    if not pd.isna(t_cv) and t_cv > 0.06:
        temp_stable = False
    if t_spikes is not None and t_spikes > 30:
        temp_stable = False

    stable = vpd_stable and temp_stable

    # ---------- Simple label ----------
    gear_word = {
        "L": "Low gear",
        "M": "Mid gear",
        "H": "High gear",
        "?": "Unknown gear",
    }.get(gear, "Unknown gear")

    stability_word = "stable" if stable else "unstable"

    env_label = f"{gear_word}, {stability_word}"

    return env_label, tags


def plot_time_series(timestamp, vpd, ppfd, temp, stress_segments):
    """Create interactive Plotly time series with shaded stress periods."""
    # Shared x-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("VPD", "PPFD", "Temperature"),
    )

    x = timestamp if timestamp is not None else list(range(len(vpd)))

    # VPD
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pd.to_numeric(vpd, errors="coerce"),
            mode="lines",
            name="VPD (kPa)",
            hovertemplate="Time: %{x}<br>VPD: %{y:.2f} kPa<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # PPFD
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pd.to_numeric(ppfd, errors="coerce"),
            mode="lines",
            name="PPFD (µmol m⁻² s⁻¹)",
            hovertemplate="Time: %{x}<br>PPFD: %{y:.0f} µmol m⁻² s⁻¹<extra></extra>",
        ),
        row=2,
        col=1,
    )

    # Temperature
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pd.to_numeric(temp, errors="coerce"),
            mode="lines",
            name="Temperature (°C)",
            hovertemplate="Time: %{x}<br>Temp: %{y:.2f} °C<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Shade stress segments
    for seg in stress_segments:
        start = seg["start"]
        end = seg["end"]
        color = seg["color"]

        # Add a semi-transparent vertical band across all rows
        for row in [1, 2, 3]:
            fig.add_vrect(
                x0=start,
                x1=end,
                fillcolor=color,
                opacity=0.15,
                line_width=0,
                row=row,
                col=1,
            )

    fig.update_yaxes(title_text="VPD (kPa)", row=1, col=1)
    fig.update_yaxes(title_text="PPFD (µmol m⁻² s⁻¹)", row=2, col=1)
    fig.update_yaxes(title_text="Temp (°C)", row=3, col=1)

    fig.update_xaxes(title_text="Time", row=3, col=1)

    fig.update_layout(
        height=700,
        margin=dict(l=60, r=20, t=40, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # 🔥 Remove the default "new text" labels on the vrects
    fig.update_layout(annotations=[])

    return fig


# ----------------- Streamlit UI wrapper class ----------------- #

class ClimateAnalyzerUI:
    """Streamlit UI wrapper for the greenhouse climate analyzer."""

    @classmethod
    def render(cls):
        st.subheader("Greenhouse Climate Analyzer")

        st.markdown(
            """
            Upload a climate log (Excel) to get a quick overview of how your environment behaves:
            - VPD, PPFD and Temperature stability
            - Simple gear classification based on DLI
            - Shaded time series showing stress windows
            """
        )

        uploaded_file = st.file_uploader(
            "Upload climate Excel file",
            type=["xlsx", "xls"],
            help="Typical export with timestamp, VPD, PPFD and temperature columns.",
        )

        if uploaded_file is None:
            st.info("Upload an Excel file to begin.")
            return

        try:
            df = pd.read_excel(uploaded_file)
        except Exception as e:
            st.error(f"Could not read Excel file: {e}")
            return

        if df.empty:
            st.warning("The uploaded file appears to be empty.")
            return

        st.markdown("#### Preview")
        st.dataframe(df.head(), use_container_width=True)

        cols = list(df.columns.astype(str))

        st.markdown("#### Column mapping")

        c1, c2 = st.columns(2)
        with c1:
            ts_col = st.selectbox(
                "Timestamp column (optional but recommended)",
                options=["<none>"] + cols,
                index=0,
            )

            vpd_col = st.selectbox(
                "VPD column",
                options=cols,
                help="kPa",
            )

        with c2:
            ppfd_col = st.selectbox(
                "PPFD column",
                options=cols,
                help="µmol m⁻² s⁻¹",
            )

            temp_col = st.selectbox(
                "Temperature column",
                options=cols,
                help="°C",
            )

        # Optional analysis window if timestamp selected
        use_ts = ts_col != "<none>"

        start_dt = None
        end_dt = None
        ts_series = None

        if use_ts:
            try:
                ts_series = pd.to_datetime(df[ts_col], errors="coerce")
            except Exception:
                st.warning("Could not parse timestamp column; proceeding without timestamps.")
                ts_series = None
                use_ts = False

        if use_ts and ts_series is not None and not ts_series.isna().all():
            ts_min = ts_series.min()
            ts_max = ts_series.max()

            st.markdown("#### Analysis window (optional)")

            cw1, cw2 = st.columns(2)
            with cw1:
                start_dt = st.date_input(
                    "Start date",
                    value=ts_min.date(),
                    min_value=ts_min.date(),
                    max_value=ts_max.date(),
                )
            with cw2:
                end_dt = st.date_input(
                    "End date",
                    value=ts_max.date(),
                    min_value=ts_min.date(),
                    max_value=ts_max.date(),
                )

            # Convert dates to datetimes spanning full days
            start_dt = pd.to_datetime(start_dt)
            end_dt = pd.to_datetime(end_dt) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

        else:
            st.markdown(
                "<small>No valid timestamp selected; analysis uses the full dataset and assumes hourly spacing.</small>",
                unsafe_allow_html=True,
            )

        if st.button("Run climate analysis", type="primary"):
            cls._run_analysis(
                df=df,
                ts_series=ts_series if use_ts else None,
                vpd_col=vpd_col,
                ppfd_col=ppfd_col,
                temp_col=temp_col,
                start_dt=start_dt,
                end_dt=end_dt,
            )

    @classmethod
    def _run_analysis(cls, df, ts_series, vpd_col, ppfd_col, temp_col, start_dt, end_dt):
        # Work on filtered copy
        df_work = df.copy()

        if ts_series is not None:
            df_work["_ts_"] = ts_series

            if start_dt is not None:
                df_work = df_work[df_work["_ts_"] >= start_dt]
            if end_dt is not None:
                df_work = df_work[df_work["_ts_"] <= end_dt]

            if df_work.empty:
                st.warning("No data points fall inside the selected time window.")
                return

            ts_for_analysis = df_work["_ts_"]
            window_desc = (
                f"{start_dt.date()} → {end_dt.date()}"
                if start_dt and end_dt
                else "Full timestamp range"
            )
        else:
            ts_for_analysis = None
            window_desc = "Full dataset (no timestamp)"

        st.markdown(f"**Analysis window:** {window_desc}")

        results = {}
        metrics_text = []

        # Analyze each variable
        for label, col in [("VPD", vpd_col), ("PPFD", ppfd_col), ("Temperature", temp_col)]:
            config = THRESHOLDS.get(label, {})
            m = analyze_series(df_work[col], label, config)
            results[label] = m
            metrics_text.append(cls._format_metrics(label, m, config))

        # Compute DLI metrics from PPFD
        dli_metrics = compute_dli_metrics(df_work[ppfd_col], ts_for_analysis)

        # Classification
        env_type, tags = classify_environment(
            results.get("VPD", {}),
            results.get("PPFD", {}),
            results.get("Temperature", {}),
            dli_metrics,
        )

        st.markdown("### Environment summary")
        if any("error" in m for m in results.values()):
            st.warning("Not enough numeric data to fully classify environment.")
        st.write(f"**{env_type}**")
        for t in tags:
            st.write(f"- {t}")

        # Metrics block
        with st.expander("Show detailed metrics", expanded=False):
            st.text("\n\n".join(metrics_text))

            st.markdown("**DLI overview**")
            mean_dli = dli_metrics.get("mean_dli", np.nan)
            if not np.isnan(mean_dli):
                st.write(f"- Mean DLI: {mean_dli:.2f} mol/m²/day")
            else:
                st.write("- Mean DLI: NA")
            st.write(f"- Days in dataset: {dli_metrics.get('n_days', 0)}")

        # Stress segments + interactive plot
        stress_segments = detect_stress_segments(
            df_work,
            ts_for_analysis,
            vpd_col=vpd_col,
            ppfd_col=ppfd_col,
        )

        fig = plot_time_series(
            timestamp=ts_for_analysis,
            vpd=df_work[vpd_col],
            ppfd=df_work[ppfd_col],
            temp=df_work[temp_col],
            stress_segments=stress_segments,
        )

        st.markdown("### Time series view")
        st.plotly_chart(fig, use_container_width=True)

    @staticmethod
    def _format_metrics(label: str, m: dict, config: dict) -> str:
        if "error" in m:
            return f"--- {label} ---\n{m['error']}\n"

        def fmt(x, digits=3):
            try:
                if pd.isna(x):
                    return "NA"
                return f"{x:.{digits}f}"
            except Exception:
                return str(x)

        lines = [f"--- {label} ---"]
        lines.append(f" n_hours: {m['n_hours']}  (number of hourly records in this window)")
        lines.append(f" mean: {fmt(m['mean'])}  (average {label} over the window)")
        lines.append(f" sd: {fmt(m['sd'])}  (standard deviation — how spread out values are)")
        lines.append(f" cv: {fmt(m['cv'])}  (relative variability; >0.5 means quite variable)")

        thresh = config.get("stress_high", None)
        if thresh is not None:
            lines.append(
                f" hours_above_stress (>{thresh}): {m['hours_above_stress']} "
                f"({fmt(m['pct_above_stress'])} %)  "
                f"(time spent above the stress limit)"
            )
        else:
            lines.append(" hours_above_stress: NA")

        opt_low = config.get("optimal_low", None)
        if opt_low is not None:
            lines.append(
                f" hours_below_optimal (<{opt_low}): {m['hours_below_optimal']} "
                f"({fmt(m['pct_below_optimal'])} %)  "
                f"(time spent below the lower comfort zone)"
            )
        else:
            lines.append(" hours_below_optimal: NA")

        lines.append(
            f" lag1_autocorr: {fmt(m['lag1_autocorr'])}  "
            f"(how similar each hour is to the previous one; closer to 1 = smoother trends)"
        )
        lines.append(
            f" mean_abs_delta: {fmt(m['mean_abs_delta'])}  "
            f"(average hour-to-hour change in {label})"
        )
        lines.append(
            f" max_abs_delta: {fmt(m['max_abs_delta'])}  "
            f"(largest single jump between two hours)"
        )

        spike_delta = config.get("spike_delta", None)
        if spike_delta is not None:
            lines.append(
                f" n_spikes (|Δ|>{spike_delta}): {m['n_spikes']}, "
                f"max_spike={fmt(m['max_spike'])}  "
                f"(number and size of large jumps)"
            )
        else:
            lines.append(" spikes: NA")

        return "\n".join(lines)

