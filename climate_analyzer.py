# climate_analyzer.py

import io
import math
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
    "PAR": {
        "stress_high": 1000.0, # µmol m⁻² s⁻¹ (same as PPFD)
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

# Detection constants
RH_FRACTION_MAX = 1.2  # Max value for RH as fraction (0-1); allows slight overshoot
RH_PERCENTAGE_MAX = 105  # Max value for RH as percentage (0-100); allows slight overshoot
PAR_LOW_THRESHOLD = 100  # PAR values below this might be in W/m² not µmol m⁻² s⁻¹
PAR_MAX_WM2 = 600  # Typical max for PAR in W/m²



# ----------------- File loading and inference helpers ----------------- #

def load_climate_file(uploaded_file):
    """
    Load climate data from uploaded CSV or Excel file.
    Returns (df, source_type) where source_type is "csv" or "excel".
    """
    filename = uploaded_file.name.lower()
    
    try:
        if filename.endswith('.csv'):
            # Try common CSV separators
            content = uploaded_file.read()
            uploaded_file.seek(0)
            
            # Try comma first (most common)
            try:
                df = pd.read_csv(io.BytesIO(content), sep=',')
                if len(df.columns) > 1 and len(df) > 0:
                    return df, "csv"
            except Exception:
                pass
            
            # Try semicolon
            try:
                df = pd.read_csv(io.BytesIO(content), sep=';')
                if len(df.columns) > 1 and len(df) > 0:
                    return df, "csv"
            except Exception:
                pass
            
            # Try tab
            try:
                df = pd.read_csv(io.BytesIO(content), sep='\t')
                if len(df.columns) > 1 and len(df) > 0:
                    return df, "csv"
            except Exception:
                pass
            
            # Fall back to default pandas detection
            df = pd.read_csv(io.BytesIO(content))
            if len(df.columns) == 1:
                raise ValueError("Could not detect CSV separator. File may have only one column or use an unusual separator.")
            return df, "csv"
            
        else:  # Excel
            df = pd.read_excel(uploaded_file)
            return df, "excel"
            
    except Exception as e:
        raise ValueError(f"Could not read file: {e}")


def infer_timestamp_column(df):
    """
    Infer which column is most likely the timestamp.
    Returns (col_name_or_none, confidence, explanation).
    """
    candidates = []
    
    for col in df.columns:
        col_lower = str(col).lower()
        score = 0
        reasons = []
        
        # Name-based matching
        if any(keyword in col_lower for keyword in ['timestamp', 'time', 'date', 'datetime']):
            score += 50
            reasons.append("name matches timestamp pattern")
        
        # Try parsing as datetime
        try:
            parsed = pd.to_datetime(df[col], errors='coerce')
            valid_count = parsed.notna().sum()
            valid_pct = valid_count / len(df) if len(df) > 0 else 0
            
            if valid_pct > 0.8:
                score += 40
                reasons.append(f"{valid_pct*100:.0f}% of values parse as datetime")
            elif valid_pct > 0.5:
                score += 20
                reasons.append(f"{valid_pct*100:.0f}% of values parse as datetime")
        except Exception:
            pass
        
        if score > 0:
            candidates.append({
                'col': col,
                'score': score,
                'reasons': reasons
            })
    
    if not candidates:
        return None, 0, "No timestamp column detected"
    
    # Pick best candidate
    best = max(candidates, key=lambda x: x['score'])
    confidence = min(100, best['score'])
    explanation = "; ".join(best['reasons'])
    
    return best['col'], confidence, explanation


def infer_env_columns(df):
    """
    Infer VPD, PPFD, Temperature, and RH columns using name matching + value range checks.
    Returns dict with keys 'vpd', 'ppfd', 'temp', 'rh', each containing:
      {'col': name, 'score': confidence, 'explanation': why}
    """
    result = {
        'vpd': {'col': None, 'score': 0, 'explanation': ''},
        'ppfd': {'col': None, 'score': 0, 'explanation': ''},
        'temp': {'col': None, 'score': 0, 'explanation': ''},
        'rh': {'col': None, 'score': 0, 'explanation': ''}
    }
    
    for col in df.columns:
        col_lower = str(col).lower()
        col_data = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(col_data) == 0:
            continue
        
        mean_val = col_data.mean()
        min_val = col_data.min()
        max_val = col_data.max()
        
        # VPD detection
        vpd_score = 0
        vpd_reasons = []
        if any(keyword in col_lower for keyword in ['vpd', 'vapor pressure deficit']):
            vpd_score += 50
            vpd_reasons.append("name matches 'VPD'")
        
        # VPD typically 0-3 kPa, rarely > 5
        if 0 <= mean_val <= 3 and min_val >= 0 and max_val < 6:
            vpd_score += 30
            vpd_reasons.append(f"values in range 0-3 kPa (mean={mean_val:.2f})")
        
        if vpd_score > result['vpd']['score']:
            result['vpd'] = {
                'col': col,
                'score': vpd_score,
                'explanation': "; ".join(vpd_reasons)
            }
        
        # PPFD/PAR/Light detection
        ppfd_score = 0
        ppfd_reasons = []
        if any(keyword in col_lower for keyword in ['ppfd', 'par', 'light', 'photon', 'umol', 'µmol']):
            ppfd_score += 50
            ppfd_reasons.append(f"name matches 'PPFD/PAR/light'")
        
        # PPFD typically 0-2000 µmol m⁻² s⁻¹
        if 0 <= mean_val <= 1500 and min_val >= 0 and max_val < 2500:
            ppfd_score += 30
            ppfd_reasons.append(f"values in range 0-2000 (mean={mean_val:.0f})")
        
        if ppfd_score > result['ppfd']['score']:
            result['ppfd'] = {
                'col': col,
                'score': ppfd_score,
                'explanation': "; ".join(ppfd_reasons)
            }
        
        # Temperature detection
        temp_score = 0
        temp_reasons = []
        if any(keyword in col_lower for keyword in ['temp', 'temperature', 'tair', 't_air']):
            temp_score += 50
            temp_reasons.append("name matches 'temperature'")
        
        # Greenhouse temp typically 0-45 °C
        if 0 <= mean_val <= 45 and min_val >= -5 and max_val < 55:
            temp_score += 30
            temp_reasons.append(f"values in range 0-45°C (mean={mean_val:.1f})")
        
        if temp_score > result['temp']['score']:
            result['temp'] = {
                'col': col,
                'score': temp_score,
                'explanation': "; ".join(temp_reasons)
            }
        
        # RH detection
        rh_score = 0
        rh_reasons = []
        if any(keyword in col_lower for keyword in ['rh', 'humidity', 'relative humidity', 'hum']):
            rh_score += 50
            rh_reasons.append("name matches 'RH/humidity'")
        
        # RH can be 0-1 (fraction) or 0-100 (percentage)
        if (0 <= mean_val <= 1 and min_val >= 0 and max_val <= RH_FRACTION_MAX) or \
           (0 <= mean_val <= 100 and min_val >= 0 and max_val <= RH_PERCENTAGE_MAX):
            rh_score += 30
            rh_reasons.append(f"values in RH range (mean={mean_val:.1f})")
        
        if rh_score > result['rh']['score']:
            result['rh'] = {
                'col': col,
                'score': rh_score,
                'explanation': "; ".join(rh_reasons)
            }
    
    return result


def compute_vpd_from_temp_rh(temp_c: pd.Series, rh: pd.Series) -> pd.Series:
    """
    Compute VPD (kPa) from temperature (°C) and relative humidity.
    Uses Magnus-Tetens formula for saturation vapor pressure.
    
    Args:
        temp_c: Temperature in Celsius
        rh: Relative humidity (0-1 or 0-100, will be normalized)
    
    Returns:
        VPD in kPa
    """
    # Convert to numeric
    temp = pd.to_numeric(temp_c, errors='coerce')
    rh_vals = pd.to_numeric(rh, errors='coerce')
    
    # Normalize RH to 0-100 range
    # If values are between 0-1, multiply by 100
    if rh_vals.max() <= RH_FRACTION_MAX:  # Assume it's fractional
        rh_vals = rh_vals * 100.0
    
    # Clamp RH to valid range
    rh_vals = rh_vals.clip(0, 100)
    
    # Magnus-Tetens formula for saturation vapor pressure (kPa)
    # es = 0.6108 * exp((17.27*T) / (T + 237.3))
    es = 0.6108 * np.exp((17.27 * temp) / (temp + 237.3))
    
    # Actual vapor pressure
    ea = es * (rh_vals / 100.0)
    
    # VPD = es - ea, but ensure non-negative
    vpd = (es - ea).clip(lower=0)
    
    return vpd


def infer_time_step_seconds(ts_series):
    """
    Infer the time step from a timestamp series.
    Returns (step_seconds, label, explanation).
    """
    if ts_series is None or ts_series.isna().all():
        return None, None, "No timestamp data"
    
    # Parse and clean
    ts_clean = pd.to_datetime(ts_series, errors='coerce').dropna().drop_duplicates().sort_values()
    
    if len(ts_clean) < 2:
        return None, None, "Not enough unique timestamps"
    
    # Compute deltas
    deltas = ts_clean.diff().dropna()
    delta_seconds = deltas.dt.total_seconds()
    
    # Use median to be robust against outliers
    median_seconds = delta_seconds.median()
    
    # Guard against zero or negative values
    if median_seconds <= 0:
        return None, None, "Invalid time deltas detected"
    
    # Create human-readable label
    if median_seconds < 60:  # Less than 1 minute
        label = f"{int(median_seconds)} sec"
    elif median_seconds < 90:  # Less than 1.5 minutes
        label = "1 min"
    elif median_seconds < 3600:  # Less than 1 hour
        step_minutes = round(median_seconds / 60)
        label = f"{step_minutes} min"
    elif median_seconds < 86400:  # Less than 1 day
        step_hours = round(median_seconds / 3600)
        label = f"{step_hours} hour" if step_hours == 1 else f"{step_hours} hours"
    else:
        step_days = round(median_seconds / 86400)
        label = f"{step_days} day" if step_days == 1 else f"{step_days} days"
    
    explanation = f"Median time delta: {median_seconds:.0f} seconds ({label})"
    
    return median_seconds, label, explanation


def resample_df(df_work, ts_col, interval, step_seconds):
    """
    Resample dataframe to the specified interval.
    interval: 'raw', '1min', '5min', '1hour', '1day'
    Returns resampled dataframe with _ts_ column.
    """
    if interval == 'raw' or ts_col is None:
        return df_work
    
    # Map interval to pandas resample rule
    interval_map = {
        '1min': '1min',
        '5min': '5min',
        '1hour': '1h',
        '1day': '1D'
    }
    
    rule = interval_map.get(interval)
    if rule is None:
        return df_work
    
    # Set timestamp as index
    df_temp = df_work.copy()
    df_temp = df_temp.set_index(ts_col)
    
    # Resample numeric columns using mean
    df_resampled = df_temp.select_dtypes(include=[np.number]).resample(rule).mean()
    
    # Reset index to restore timestamp as column named _ts_
    df_resampled = df_resampled.reset_index()
    df_resampled = df_resampled.rename(columns={ts_col: '_ts_'})
    
    return df_resampled


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


def compute_dli_metrics(ppfd_series: pd.Series, timestamp: pd.Series | None, step_seconds: float | None = None):
    """
    Compute daily light integral (DLI) metrics from PPFD.
    PPFD is in µmol m⁻² s⁻¹.
    DLI_day = sum(PPFD_sample * step_seconds) / 1e6 [mol m⁻² d⁻¹]
    
    Args:
        ppfd_series: PPFD data
        timestamp: Timestamp series (if available)
        step_seconds: Time step in seconds (if known)
    """
    s = pd.to_numeric(ppfd_series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean_dli": np.nan, "sd_dli": np.nan, "n_days": 0}

    # If we have timestamps, group by date and integrate properly
    if timestamp is not None and not timestamp.isna().all():
        ts_valid = timestamp.loc[s.index]
        df = pd.DataFrame({"ts": ts_valid, "ppfd": s}).dropna(subset=["ts", "ppfd"])
        if df.empty:
            return {"mean_dli": np.nan, "sd_dli": np.nan, "n_days": 0}
        
        df["date"] = df["ts"].dt.date
        
        # Use provided step_seconds or infer it
        if step_seconds is None:
            # Try to infer from timestamps
            ts_sorted = df["ts"].sort_values()
            if len(ts_sorted) > 1:
                deltas = ts_sorted.diff().dropna().dt.total_seconds()
                step_seconds = deltas.median()
            else:
                step_seconds = 3600.0  # Default to hourly
        
        daily = df.groupby("date")["ppfd"].sum()
        dli_per_day = daily * step_seconds / 1e6
        return {
            "mean_dli": float(dli_per_day.mean()),
            "sd_dli": float(dli_per_day.std()),
            "n_days": int(dli_per_day.shape[0]),
        }
    else:
        # No timestamps: use step_seconds if provided, otherwise assume hourly
        if step_seconds is None:
            step_seconds = 3600.0  # Default to hourly
        
        n = len(s)
        samples_per_day = int(86400 / step_seconds)
        
        if n >= samples_per_day:
            day_index = np.floor(np.arange(n) / samples_per_day).astype(int)
            df = pd.DataFrame({"day": day_index, "ppfd": s.values})
            daily = df.groupby("day")["ppfd"].sum()
            dli_per_day = daily * step_seconds / 1e6
            return {
                "mean_dli": float(dli_per_day.mean()),
                "sd_dli": float(dli_per_day.std()),
                "n_days": int(dli_per_day.shape[0]),
            }
        else:
            # Short segment: estimate using mean PPFD over 24 hours
            dli_est = float(s.mean() * 86400.0 / 1e6)
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


def detect_stress_segments(df_work: pd.DataFrame, ts: pd.Series | None, vpd_col: str, ppfd_col: str, step_seconds: float | None = None):
    """
    Detect stress segments for:
    - High VPD (tipburn risk)
    - Low VPD (glassiness / disease risk)
    - Very high PPFD (photoinhibition risk)
    
    Args:
        step_seconds: Time step in seconds; used to convert hour-based windows to point counts
        
    Returns a list of dicts: {"start": ..., "end": ..., "label": ..., "color": ...}
    """
    segments = []

    vpd_cfg = THRESHOLDS["VPD"]
    ppfd_cfg = THRESHOLDS["PPFD"]

    vpd = pd.to_numeric(df_work[vpd_col], errors="coerce")
    ppfd = pd.to_numeric(df_work[ppfd_col], errors="coerce")

    # Convert hour-based windows to point counts based on step_seconds
    # If step_seconds is None or invalid, default to hourly (3600s)
    if step_seconds is None or step_seconds <= 0:
        step_seconds = 3600.0
    
    # Calculate conversion factor: points per hour
    points_per_hour = 3600.0 / step_seconds
    
    high_vpd_min_points = max(1, math.ceil(STRESS_WINDOWS["high_vpd_min_run"] * points_per_hour))
    low_vpd_min_points = max(1, math.ceil(STRESS_WINDOWS["low_vpd_min_run"] * points_per_hour))
    high_ppfd_min_points = max(1, math.ceil(STRESS_WINDOWS["high_ppfd_min_run"] * points_per_hour))

    # High VPD stress (tipburn risk)
    high_vpd_cond = vpd > vpd_cfg["stress_high"]
    high_vpd_runs = _find_segments(high_vpd_cond, high_vpd_min_points, ts)
    for start, end in high_vpd_runs:
        segments.append({
            "start": start,
            "end": end,
            "label": "High VPD stress (tipburn risk)",
            "color": "red"
        })

    # Low VPD / very humid (glassiness / disease risk)
    low_vpd_cond = vpd < vpd_cfg["optimal_low"]
    low_vpd_runs = _find_segments(low_vpd_cond, low_vpd_min_points, ts)
    for start, end in low_vpd_runs:
        segments.append({
            "start": start,
            "end": end,
            "label": "Low VPD / humid (glassiness risk)",
            "color": "blue"
        })

    # Very high PPFD (photoinhibition risk)
    high_ppfd_cond = ppfd > ppfd_cfg["stress_high"]
    high_ppfd_runs = _find_segments(high_ppfd_cond, high_ppfd_min_points, ts)
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


def plot_time_series(timestamp, vpd, ppfd, temp, stress_segments, light_label="PPFD (µmol m⁻² s⁻¹)"):
    """Create interactive Plotly time series with shaded stress periods."""
    # Shared x-axis
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=("VPD", light_label.split("(")[0].strip(), "Temperature"),
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

    # Light (PPFD or PAR)
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pd.to_numeric(ppfd, errors="coerce"),
            mode="lines",
            name=light_label,
            hovertemplate=f"Time: %{{x}}<br>{light_label}: %{{y:.0f}}<extra></extra>",
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
    fig.update_yaxes(title_text=light_label, row=2, col=1)
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
            Upload a climate log (CSV or Excel) to get a quick overview of how your environment behaves:
            - VPD, PPFD and Temperature stability
            - Simple gear classification based on DLI
            - Shaded time series showing stress windows
            """
        )

        uploaded_file = st.file_uploader(
            "Upload climate file",
            type=["xlsx", "xls", "csv"],
            help="CSV or Excel file with timestamp, VPD, PPFD and temperature columns.",
        )

        if uploaded_file is None:
            st.info("Upload a CSV or Excel file to begin.")
            return

        try:
            df, source_type = load_climate_file(uploaded_file)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            return

        if df.empty:
            st.warning("The uploaded file appears to be empty.")
            return

        st.markdown("#### Preview")
        st.dataframe(df.head(), use_container_width=True)

        cols = list(df.columns.astype(str))

        # Auto-detect columns (silently)
        ts_col_inferred, _, _ = infer_timestamp_column(df)
        env_cols = infer_env_columns(df)

        st.markdown("#### Column mapping")
        st.caption("Review and adjust auto-selected columns:")

        c1, c2 = st.columns(2)
        with c1:
            # Pre-select timestamp if detected
            ts_options = ["<none>"] + cols
            if ts_col_inferred and ts_col_inferred in cols:
                ts_default_idx = ts_options.index(ts_col_inferred)
            else:
                ts_default_idx = 0
            
            ts_col = st.selectbox(
                "Timestamp column (optional but recommended)",
                options=ts_options,
                index=ts_default_idx,
            )

            # Pre-select Temperature if detected
            if env_cols['temp']['col'] and env_cols['temp']['col'] in cols:
                temp_default_idx = cols.index(env_cols['temp']['col'])
            else:
                temp_default_idx = 0
            
            temp_col = st.selectbox(
                "Temperature column",
                options=cols,
                index=temp_default_idx,
                help="°C",
            )
            
            # Pre-select VPD if detected (optional now)
            vpd_options = ["<none>"] + cols
            if env_cols['vpd']['col'] and env_cols['vpd']['col'] in cols:
                vpd_default_idx = vpd_options.index(env_cols['vpd']['col'])
            else:
                vpd_default_idx = 0
            
            vpd_col = st.selectbox(
                "VPD column (optional if RH provided)",
                options=vpd_options,
                index=vpd_default_idx,
                help="kPa - will be computed from Temperature + RH if not provided",
            )

        with c2:
            # Pre-select Light column if detected
            if env_cols['ppfd']['col'] and env_cols['ppfd']['col'] in cols:
                light_default_idx = cols.index(env_cols['ppfd']['col'])
            else:
                light_default_idx = 0
            
            light_col = st.selectbox(
                "Light column",
                options=cols,
                index=light_default_idx,
                help="PPFD or PAR",
            )
            
            # Light type selector
            light_type = st.selectbox(
                "Light type",
                options=["PPFD", "PAR"],
                index=0,
                help="PPFD: Photosynthetic Photon Flux Density (µmol m⁻² s⁻¹) | PAR: Photosynthetically Active Radiation"
            )
            
            # Pre-select RH if detected (optional)
            rh_options = ["<none>"] + cols
            if env_cols['rh']['col'] and env_cols['rh']['col'] in cols:
                rh_default_idx = rh_options.index(env_cols['rh']['col'])
            else:
                rh_default_idx = 0
            
            rh_col = st.selectbox(
                "Relative Humidity column (optional)",
                options=rh_options,
                index=rh_default_idx,
                help="% RH - used to compute VPD if VPD column not provided",
            )

        # Optional analysis window if timestamp selected
        use_ts = ts_col != "<none>"

        start_dt = None
        end_dt = None
        ts_series = None
        step_seconds = None
        step_label = None

        if use_ts:
            try:
                ts_series = pd.to_datetime(df[ts_col], errors="coerce")
            except Exception:
                st.warning("Could not parse timestamp column; proceeding without timestamps.")
                ts_series = None
                use_ts = False

        if use_ts and ts_series is not None and not ts_series.isna().all():
            # Detect time step
            step_seconds, step_label, step_explanation = infer_time_step_seconds(ts_series)
            
            if step_seconds:
                st.markdown(f"**Time step detected:** {step_label} ({step_explanation})")
            else:
                st.info("Could not detect time step from timestamps")
            
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
            
            # Add resolution selector
            st.markdown("#### View resolution")
            resolution = st.selectbox(
                "Graph and metrics resolution",
                options=["Raw", "1 min", "5 min", "1 hour", "1 day"],
                index=0,
                help="Resample data to this interval for plotting and metrics calculation"
            )

        else:
            st.markdown(
                "<small>No valid timestamp selected; analysis uses the full dataset and assumes hourly spacing.</small>",
                unsafe_allow_html=True,
            )
            resolution = "Raw"

        if st.button("Run climate analysis", type="primary"):
            cls._run_analysis(
                df=df,
                ts_series=ts_series if use_ts else None,
                vpd_col=vpd_col,
                light_col=light_col,
                light_type=light_type,
                temp_col=temp_col,
                rh_col=rh_col,
                start_dt=start_dt,
                end_dt=end_dt,
                resolution=resolution,
                step_seconds=step_seconds,
            )

    @classmethod
    def _run_analysis(cls, df, ts_series, vpd_col, light_col, light_type, temp_col, rh_col, start_dt, end_dt, resolution="Raw", step_seconds=None):
        # Work on filtered copy
        df_work = df.copy()
        
        # Handle VPD computation if needed
        use_vpd_col = vpd_col
        if vpd_col == "<none>" or vpd_col not in df_work.columns:
            # Try to compute VPD from temperature + RH
            if rh_col != "<none>" and rh_col in df_work.columns and temp_col in df_work.columns:
                try:
                    df_work["_vpd_calc_"] = compute_vpd_from_temp_rh(df_work[temp_col], df_work[rh_col])
                    use_vpd_col = "_vpd_calc_"
                    st.info("VPD computed from Temperature and Relative Humidity.")
                except Exception as e:
                    st.warning(f"Could not compute VPD from Temperature and RH: {e}")
                    st.warning("Please select a VPD column or provide both Temperature and RH columns.")
                    return
            else:
                st.warning("VPD column not selected and cannot be computed. Please select a VPD column or provide both Temperature and RH columns.")
                return
        
        # Determine light label based on type
        if light_type == "PPFD":
            light_label = "PPFD (µmol m⁻² s⁻¹)"
        else:  # PAR
            light_label = "PAR (µmol m⁻² s⁻¹)"

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
            
            # Apply resampling if requested
            if resolution != "Raw":
                interval_map = {
                    "1 min": "1min",
                    "5 min": "5min",
                    "1 hour": "1hour",
                    "1 day": "1day"
                }
                interval = interval_map.get(resolution, "raw")
                
                # Update step_seconds based on resolution
                resolution_seconds = {
                    "1 min": 60,
                    "5 min": 300,
                    "1 hour": 3600,
                    "1 day": 86400
                }
                step_seconds = resolution_seconds.get(resolution, step_seconds)
                
                df_work = resample_df(df_work, "_ts_", interval, step_seconds)
                ts_for_analysis = df_work["_ts_"]
                window_desc += f" (resampled to {resolution})"
        else:
            ts_for_analysis = None
            window_desc = "Full dataset (no timestamp)"

        st.markdown(f"**Analysis window:** {window_desc}")

        results = {}
        metrics_text = []

        # Analyze each variable
        for label, col in [("VPD", use_vpd_col), (light_type, light_col), ("Temperature", temp_col)]:
            config = THRESHOLDS.get(label, {})
            m = analyze_series(df_work[col], label, config)
            results[label] = m
            metrics_text.append(cls._format_metrics(label, m, config))

        # Compute DLI metrics from light column (only if appropriate)
        # DLI is only valid for photon flux measurements (µmol m⁻² s⁻¹)
        light_data = pd.to_numeric(df_work[light_col], errors='coerce').dropna()
        light_mean = light_data.mean() if len(light_data) > 0 else 0
        
        # Check if PAR might be in W/m² (typical range 0-500) vs µmol m⁻² s⁻¹ (typical 0-2000)
        if light_type == "PAR" and light_mean > 0:
            if light_mean < PAR_LOW_THRESHOLD and light_data.max() < PAR_MAX_WM2:
                st.warning("⚠️ PAR values appear to be in W/m² rather than µmol m⁻² s⁻¹. DLI calculation may not be accurate. Consider converting to photon flux units or selecting a different light type.")
                # Still compute DLI but it will be less meaningful
                dli_metrics = compute_dli_metrics(df_work[light_col], ts_for_analysis, step_seconds)
            else:
                dli_metrics = compute_dli_metrics(df_work[light_col], ts_for_analysis, step_seconds)
        else:
            dli_metrics = compute_dli_metrics(df_work[light_col], ts_for_analysis, step_seconds)

        # Classification (use PPFD key for light data regardless of type)
        light_results = results.get(light_type, {})
        env_type, tags = classify_environment(
            results.get("VPD", {}),
            light_results,  # Pass light results
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
            vpd_col=use_vpd_col,
            ppfd_col=light_col,
            step_seconds=step_seconds,
        )

        fig = plot_time_series(
            timestamp=ts_for_analysis,
            vpd=df_work[use_vpd_col],
            ppfd=df_work[light_col],
            temp=df_work[temp_col],
            stress_segments=stress_segments,
            light_label=light_label,
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
        lines.append(f" n_hours: {m['n_hours']}  (number of data points in this window)")
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
            f"(how similar each point is to the previous one; closer to 1 = smoother trends)"
        )
        lines.append(
            f" mean_abs_delta: {fmt(m['mean_abs_delta'])}  "
            f"(average point-to-point change in {label})"
        )
        lines.append(
            f" max_abs_delta: {fmt(m['max_abs_delta'])}  "
            f"(largest single jump between two points)"
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


# ----------------- Climate Feedback Loop Builder ----------------- #
# REMOVED: The ClimateFeedbackLoopUI class has been fully removed.
# If you need to restore it, check the git history for this file.
