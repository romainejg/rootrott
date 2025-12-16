"""
Canopy Closure Estimator Module

This module provides functions to estimate days until canopy closure based on:
- Plant density (plants/m²)
- Environmental averages: Temperature, DLI, CO2, VPD
- Normalized PDI (Plant Development Index)
- Density buffering (high density less sensitive to environment)
- Optional variety/stress modifiers

Model is calibrated to specific anchor points at standard environment.
"""

import numpy as np
import pandas as pd


# Standard reference environment for PDI normalization
T_REF = 23.0      # °C
DLI_REF = 18.0    # mol m⁻² d⁻¹
CO2_REF = 800.0   # ppm
VPD_REF = 0.8     # kPa


def compute_pdi(T, DLI, CO2, VPD):
    """
    Compute normalized Plant Development Index (PDI) from environmental averages.
    
    PDI is anchored so that standard environment (T=23, DLI=18, CO2=800, VPD=0.8)
    gives PDI = 1.0. Higher PDI means better growing conditions.
    
    Args:
        T: Average temperature (°C)
        DLI: Average Daily Light Integral (mol m⁻² d⁻¹)
        CO2: Average CO2 concentration (ppm)
        VPD: Average Vapor Pressure Deficit (kPa)
    
    Returns:
        tuple: (pdi_raw, pdi) where pdi is clipped to [0.70, 1.30]
    
    Quick checks at standard env (23, 18, 800, 0.8):
        PDI should be ~1.0
    """
    # Protect against division by zero
    if DLI_REF <= 0 or CO2_REF <= 0 or VPD_REF <= 0:
        return 0.0, 0.0
    
    # Light term: weighted combination of DLI and CO2
    light_term = 0.8 * (DLI / DLI_REF) + 0.2 * (CO2 / CO2_REF)
    
    # Temperature term: Gaussian response centered at T_ref
    temp_term = np.exp(-((T - T_REF) ** 2) / (2 * 8 ** 2))
    
    # VPD penalty: increases linearly when VPD exceeds reference
    vpd_penalty = 1 + 0.8 * max(0, (VPD / VPD_REF) - 1)
    
    # Raw PDI
    pdi_raw = (light_term * temp_term) / vpd_penalty
    
    # Clip to reasonable range
    pdi = np.clip(pdi_raw, 0.70, 1.30)
    
    return float(pdi_raw), float(pdi)


def t90_anchor_from_density(D):
    """
    Compute days to 90% closure at standard environment (PDI=1).
    
    This is a piecewise function calibrated to hit exact anchor points:
        D=1200 -> 19 days
        D=600  -> 21 days
        D=350  -> 27 days
        D=50   -> 37 days
        D=36   -> 44 days
    
    Args:
        D: Plant density (plants/m²)
    
    Returns:
        float: Days to 90% closure at PDI=1
    
    Quick checks at standard env (PDI=1):
        D=600  -> ~21 days
        D=1200 -> ~19 days
        D=350  -> ~27 days
        D=50   -> ~37 days
        D=36   -> ~44 days
    """
    # Clamp density to minimum
    D = max(D, 1.0)
    
    if D >= 600:
        # High density: saturating asymptote
        tmin = 14.2
        t600 = 21.0
        kh = 0.5
        t90 = tmin + (t600 - tmin) * (600 / D) ** kh
    
    elif D >= 350:
        # 350-600 range: power interpolation
        k = np.log(27 / 21) / np.log(600 / 350)
        t90 = 21 * (D / 600) ** (-k)
    
    elif D >= 50:
        # 50-350 range: power interpolation
        k = np.log(37 / 27) / np.log(350 / 50)
        t90 = 27 * (D / 350) ** (-k)
    
    elif D >= 36:
        # 36-50 range: power interpolation
        k = np.log(44 / 37) / np.log(50 / 36)
        t90 = 37 * (D / 50) ** (-k)
    
    else:
        # Below 36: extrapolate using 36-50 segment
        k = np.log(44 / 37) / np.log(50 / 36)
        t90 = 37 * (D / 50) ** (-k)
    
    return float(t90)


def alpha_from_density(D):
    """
    Compute density buffering factor alpha(D).
    
    This controls how sensitive the model is to PDI changes:
    - At high density: alpha is small (~0.09), so environment barely affects days
    - At low density: alpha is large (~0.75), so environment strongly affects days
    
    Args:
        D: Plant density (plants/m²)
    
    Returns:
        float: Buffering factor alpha
    """
    alpha = 0.09 + 0.66 / (1 + (D / 150) ** 2)
    return float(alpha)


def canopy_days_to_target(D, T, DLI, CO2, VPD, target_pct=90.0, speed_mult=1.0):
    """
    Compute days to reach target canopy closure percentage.
    
    This is the main calculation function that combines all components:
    1. Compute PDI from environment
    2. Get t90_anchor from density
    3. Get alpha (buffering) from density
    4. Adjust t90 based on PDI and alpha
    5. Apply speed multipliers (variety/stress modifiers)
    6. Convert to target closure percentage
    
    Args:
        D: Plant density (plants/m²)
        T: Average temperature (°C)
        DLI: Average Daily Light Integral (mol m⁻² d⁻¹)
        CO2: Average CO2 concentration (ppm)
        VPD: Average Vapor Pressure Deficit (kPa)
        target_pct: Target closure percentage (0-100, default 90)
        speed_mult: Speed multiplier from variety/stress modifiers (default 1.0)
    
    Returns:
        dict with keys:
            - pdi_raw: Raw PDI before clipping
            - pdi: Clipped PDI value
            - t90_anchor: Days to 90% at PDI=1
            - alpha: Density buffering factor
            - t90: Days to 90% after PDI adjustment
            - speed_multiplier: Applied speed multiplier
            - effective_t90: Days to 90% after all modifiers
            - t_target: Days to target closure %
            - k: Growth rate constant (for curve generation)
    """
    # Clamp inputs
    D = max(D, 1.0)
    target_pct = np.clip(target_pct, 1.0, 99.9)
    speed_mult = max(speed_mult, 0.01)  # Avoid division by zero
    
    # Step 1: Compute PDI
    pdi_raw, pdi = compute_pdi(T, DLI, CO2, VPD)
    
    # Step 2: Get anchor from density
    t90_anchor = t90_anchor_from_density(D)
    
    # Step 3: Get buffering factor
    alpha = alpha_from_density(D)
    
    # Step 4: Adjust for environment (PDI)
    # t90 = t90_anchor * PDI^(-alpha)
    t90 = t90_anchor * (pdi ** (-alpha))
    
    # Step 5: Apply speed multipliers (slower = more days)
    # effective_t90 = t90 / speed_multiplier
    effective_t90 = t90 / speed_mult
    
    # Step 6: Compute k from effective_t90
    # For saturating exponential: CC(t) = 1 - exp(-k*t)
    # At t=t90, CC=0.90, so k = ln(10)/t90
    k = np.log(10) / effective_t90
    
    # Step 7: Compute t_target for generic target
    # CC(t_target) = target_pct/100
    # target_pct/100 = 1 - exp(-k*t_target)
    # t_target = -ln(1 - target_pct/100) / k
    p_frac = target_pct / 100.0
    t_target = -np.log(1 - p_frac) / k
    
    return {
        'pdi_raw': pdi_raw,
        'pdi': pdi,
        't90_anchor': t90_anchor,
        'alpha': alpha,
        't90': t90,
        'speed_multiplier': speed_mult,
        'effective_t90': effective_t90,
        't_target': t_target,
        'k': k,
    }


def closure_curve(t90, target_pct=90.0, max_days=None):
    """
    Generate closure curve data (closure % vs day).
    
    Uses saturating exponential model: CC(t) = 1 - exp(-k*t)
    where k is calibrated so CC(t90) = 0.90
    
    Args:
        t90: Days to 90% closure (after all adjustments)
        target_pct: Target closure percentage (for determining plot range)
        max_days: Maximum days to plot (default: ceil(t_target * 1.2))
    
    Returns:
        pandas.DataFrame with columns 'day' and 'closure_pct'
    """
    # Compute k from t90
    k = np.log(10) / t90
    
    # Determine plot range
    if max_days is None:
        p_frac = target_pct / 100.0
        t_target = -np.log(1 - p_frac) / k
        max_days = int(np.ceil(t_target * 1.2))
    
    # Generate curve
    days = np.linspace(0, max_days, max(100, max_days * 2))
    closure_pct = 100 * (1 - np.exp(-k * days))
    
    return pd.DataFrame({
        'day': days,
        'closure_pct': closure_pct
    })


def canopy_days_multistage(stages, T, DLI, CO2, VPD, target_pct=90.0, speed_mult=1.0):
    """
    Compute days to target closure with transplant stages.
    
    This function handles multiple density stages (e.g., plug -> transplant -> final).
    Each stage is at a different density, and we compute partial closure at each stage.
    
    Args:
        stages: List of tuples [(density1, days1), (density2, days2), ...]
                Last stage has days=None and goes until target closure
        T: Average temperature (°C)
        DLI: Average Daily Light Integral (mol m⁻² d⁻¹)
        CO2: Average CO2 concentration (ppm)
        VPD: Average Vapor Pressure Deficit (kPa)
        target_pct: Target closure percentage (0-100)
        speed_mult: Speed multiplier from variety/stress modifiers
    
    Returns:
        dict with keys:
            - stages_info: List of dicts with info for each stage
            - total_days: Total days to reach target
            - final_closure_pct: Final closure percentage
    """
    # Validate inputs
    if not stages or len(stages) == 0:
        return {
            'stages_info': [],
            'total_days': 0.0,
            'final_closure_pct': 0.0
        }
    
    stages_info = []
    cumulative_days = 0.0
    current_closure = 0.0
    
    for i, stage in enumerate(stages):
        if len(stage) == 2:
            stage_density, stage_days = stage
        else:
            # Invalid stage format
            continue
        
        # Get growth parameters for this density
        result = canopy_days_to_target(
            stage_density, T, DLI, CO2, VPD, 
            target_pct=90.0,  # Always use 90 as reference
            speed_mult=speed_mult
        )
        
        k = result['k']
        
        # Determine days for this stage
        is_last_stage = (i == len(stages) - 1) or (stage_days is None)
        
        if is_last_stage:
            # Last stage: compute days to reach target from current closure
            if current_closure >= target_pct:
                actual_days = 0.0
            else:
                # Solve: current_closure + 100*(1-exp(-k*t)) = target
                # But we need to account for starting point
                # If starting at current_closure%, we need additional coverage
                remaining = (target_pct - current_closure) / 100.0
                # This is approximate; exact formula depends on combining closures
                # For simplicity: assume additive in the exponential space
                if remaining <= 0:
                    actual_days = 0.0
                else:
                    # Use simple approach: days to reach target from 0, minus days already achieved
                    t_target_total = -np.log(1 - target_pct/100.0) / k
                    t_current = -np.log(1 - current_closure/100.0) / k if current_closure > 0 else 0
                    actual_days = max(0, t_target_total - t_current)
        else:
            actual_days = stage_days
        
        # Compute closure gained in this stage
        if actual_days > 0:
            # New closure from this stage
            stage_closure = 100 * (1 - np.exp(-k * actual_days))
            # Combine with current (simplified: additive up to 100)
            new_closure = min(100.0, current_closure + stage_closure * (1 - current_closure/100.0))
        else:
            new_closure = current_closure
        
        stages_info.append({
            'stage': i + 1,
            'density': stage_density,
            'days': actual_days,
            'k': k,
            'closure_start': current_closure,
            'closure_end': new_closure,
            'cumulative_days': cumulative_days + actual_days
        })
        
        cumulative_days += actual_days
        current_closure = new_closure
        
        if current_closure >= target_pct:
            break
    
    return {
        'stages_info': stages_info,
        'total_days': cumulative_days,
        'final_closure_pct': current_closure
    }
