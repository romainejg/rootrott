import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import canopy_closure as cc


class DLICalculator:
    """Daily Light Integral calculator."""

    @staticmethod
    def compute_dli(ppfd: float, hours: float) -> float:
        """
        Compute DLI (mol·m⁻²·day⁻¹) from PPFD (µmol·m⁻²·s⁻¹)
        and photoperiod in hours.
        """
        if ppfd <= 0 or hours <= 0:
            return 0.0
        return ppfd * hours * 3600.0 / 1_000_000.0

    @classmethod
    def render(cls):
        st.subheader("Daily Light Integral (DLI)")

        st.markdown(
            """
            DLI describes the total amount of photosynthetically active light a crop
            receives over a day, in **mol·m⁻²·day⁻¹**.

            This calculator uses:

            - **PPFD** in µmol·m⁻²·s⁻¹  
            - **Photoperiod** in hours per day  
            """
        )

        col1, col2 = st.columns(2)

        with col1:
            ppfd = st.number_input(
                "Average PPFD (µmol·m⁻²·s⁻¹)",
                min_value=0.0,
                max_value=5000.0,
                value=200.0,
                step=10.0,
            )

        with col2:
            hours = st.number_input(
                "Photoperiod (hours per day)",
                min_value=0.0,
                max_value=24.0,
                value=16.0,
                step=0.5,
            )

        if ppfd > 0 and hours > 0:
            dli = cls.compute_dli(ppfd, hours)
            st.markdown("### Result")
            st.write(f"**DLI: {dli:.2f} mol·m⁻²·day⁻¹**")
        else:
            st.info("Enter PPFD and photoperiod above zero to see the DLI.")


class VPDCalculator:
    """Vapor Pressure Deficit calculator."""

    @staticmethod
    def compute_vpd_kpa(temp_c: float, rh: float) -> float:
        """
        Compute VPD in kPa from temperature in °C and RH in %.
        Uses standard Tetens formula for saturation vapor pressure.
        """
        if rh <= 0 or rh > 100:
            return 0.0

        # Saturation vapor pressure (kPa) at temp_c
        es = 0.6108 * np.exp((17.27 * temp_c) / (temp_c + 237.3))
        # Actual vapor pressure (kPa)
        ea = es * (rh / 100.0)
        vpd = es - ea
        return max(vpd, 0.0)

    @classmethod
    def render(cls):
        st.subheader("Vapor Pressure Deficit (VPD)")

        st.markdown(
            """
            VPD describes the drying power of the air and is closely tied to crop transpiration.

            This calculator uses:

            - **Air temperature** (°C or °F)  
            - **Relative humidity** (%)  

            and returns VPD in **kPa**.
            """
        )

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            temp_unit = st.radio(
                "Temperature unit",
                ["°C", "°F"],
                index=0,
                horizontal=True,
            )

        with col2:
            temp_input = st.number_input(
                f"Air temperature ({temp_unit})",
                value=25.0,
                step=0.5,
            )

        with col3:
            rh = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=70.0,
                step=1.0,
            )

        # Convert to °C if user entered °F
        if temp_unit == "°F":
            temp_c = (temp_input - 32.0) * 5.0 / 9.0
        else:
            temp_c = temp_input

        if 0 < rh <= 100:
            vpd = cls.compute_vpd_kpa(temp_c, rh)
            st.markdown("### Result")
            st.write(f"**VPD: {vpd:.2f} kPa**")
        else:
            st.info("Set relative humidity between 0 and 100% to calculate VPD.")


class GutterPlantDensityCalculator:
    """
    Calculator for plants per square meter in a gutter system.

    Inputs:
    - Gutter width
    - Gutter length
    - Spacing between gutters
    - Plants per gutter

    All length inputs can be in m, cm, ft, or in.
    """

    @staticmethod
    def to_meters(value: float, unit: str) -> float:
        """Convert a length to meters based on unit."""
        if value is None:
            return 0.0

        unit = unit.lower()
        if unit == "m":
            return value
        elif unit == "cm":
            return value / 100.0
        elif unit == "ft":
            return value * 0.3048
        elif unit == "in":
            return value * 0.0254
        else:
            return value  # fallback (treat as meters)

    @staticmethod
    def compute_plants_per_m2(
        gutter_length_m: float,
        gutter_width_m: float,
        gutter_spacing_m: float,
        plants_per_gutter: float,
    ) -> float:
        """
        Compute plants per m².

        Assumes:
        - Each gutter occupies width = gutter_width + spacing_between_gutters
        - Area per gutter = gutter_length * (gutter_width + spacing)
        """
        if gutter_length_m <= 0 or plants_per_gutter <= 0:
            return 0.0

        pitch_m = gutter_width_m + gutter_spacing_m  # total width per gutter
        if pitch_m <= 0:
            return 0.0

        area_per_gutter_m2 = gutter_length_m * pitch_m
        if area_per_gutter_m2 <= 0:
            return 0.0

        return plants_per_gutter / area_per_gutter_m2

    @classmethod
    def render(cls):
        st.subheader("Plants per m² (Gutter Layout)")

        st.markdown(
            """
            This calculator estimates **plants per square meter** for a gutter-based system.

            It uses:

            - **Gutter width**  
            - **Gutter length**  
            - **Spacing between gutters** (gap between gutters)  
            - **Plants per gutter**  

            All length inputs can be in **m, cm, ft, or in**.
            """
        )

        # Layout inputs
        col1, col2 = st.columns(2)

        with col1:
            gutter_length_val = st.number_input(
                "Gutter length",
                min_value=0.0,
                value=100.0,
                step=10.0,
                key="gutter_length_val",
            )
            gutter_length_unit = st.selectbox(
                "Length unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_length_unit",
            )

            gutter_width_val = st.number_input(
                "Gutter width",
                min_value=0.0,
                value=0.2,
                step=0.05,
                key="gutter_width_val",
            )
            gutter_width_unit = st.selectbox(
                "Width unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_width_unit",
            )

        with col2:
            spacing_val = st.number_input(
                "Spacing between gutters",
                min_value=0.0,
                value=0.5,
                step=0.05,
                key="gutter_spacing_val",
            )
            spacing_unit = st.selectbox(
                "Spacing unit",
                ["m", "cm", "ft", "in"],
                index=0,
                key="gutter_spacing_unit",
            )

            plants_per_gutter = st.number_input(
                "Plants per gutter",
                min_value=0.0,
                value=30.0,
                step=1.0,
                key="plants_per_gutter",
            )

        # Convert to meters
        length_m = cls.to_meters(gutter_length_val, gutter_length_unit)
        width_m = cls.to_meters(gutter_width_val, gutter_width_unit)
        spacing_m = cls.to_meters(spacing_val, spacing_unit)

        if plants_per_gutter > 0 and length_m > 0 and (width_m + spacing_m) > 0:
            plants_per_m2 = cls.compute_plants_per_m2(
                length_m,
                width_m,
                spacing_m,
                plants_per_gutter,
            )

            st.markdown("### Result")
            st.write(f"**Plants per m²: {plants_per_m2:.2f}**")

            # Optional: show intermediate values
            with st.expander("Show details", expanded=False):
                st.write(f"Gutter length (m): `{length_m:.3f}`")
                st.write(f"Gutter width (m): `{width_m:.3f}`")
                st.write(f"Spacing between gutters (m): `{spacing_m:.3f}`")
                pitch_m = width_m + spacing_m
                st.write(f"Total pitch (width per gutter) (m): `{pitch_m:.3f}`")
                area_per_gutter = length_m * pitch_m
                st.write(f"Area per gutter (m²): `{area_per_gutter:.3f}`")
        else:
            st.info("Enter non-zero values for length, widths, spacing and plants per gutter to see the result.")


class CanopyClosureCalculator:
    """
    Canopy Closure Estimator based on calibrated density anchor + normalized PDI 
    with density buffering and optional demerit modifiers.
    
    Model features:
    - Normalized PDI from environmental averages (T, DLI, CO2, VPD)
    - Calibrated anchor points at standard environment
    - Density buffering (high density less sensitive to environment)
    - Optional variety/climate/stress modifiers
    - Multi-stage transplant support
    """

    @classmethod
    def render(cls):
        st.subheader("Canopy Closure Estimator")
        
        st.markdown(
            """
            **⚠️ This tool is still under construction** but provides estimates based on:
            
            - **Plant density** (plants/m²)
            - **Environmental averages**: Temperature, DLI, CO2, VPD
            - **Target closure %** (default 90%)
            - **Optional modifiers**: variety, climate consistency, stress periods
            - **Optional transplant stages**: multiple density stages if needed
            
            The model uses a calibrated anchor approach with density buffering, meaning high 
            densities (≥600 plants/m²) are less sensitive to environmental changes, while low 
            densities show more environmental sensitivity.
            """
        )
        
        # Create tabs for single-stage vs multi-stage
        mode_tabs = st.tabs(["Single Stage", "Multi-Stage (Transplants)"])
        
        # ===== SINGLE STAGE MODE =====
        with mode_tabs[0]:
            st.markdown("### Environmental Inputs")
            
            col1, col2 = st.columns(2)
            
            with col1:
                density = st.number_input(
                    "Density (plants/m²)",
                    min_value=1.0,
                    max_value=2000.0,
                    value=600.0,
                    step=10.0,
                    key="single_density"
                )
                
                temp = st.number_input(
                    "Avg Temperature (°C)",
                    min_value=-10.0,
                    max_value=45.0,
                    value=23.0,
                    step=0.5,
                    key="single_temp"
                )
                
                dli = st.number_input(
                    "Avg DLI (mol m⁻² d⁻¹)",
                    min_value=0.1,
                    max_value=60.0,
                    value=18.0,
                    step=0.5,
                    key="single_dli"
                )
            
            with col2:
                co2 = st.number_input(
                    "Avg CO2 (ppm)",
                    min_value=300.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0,
                    key="single_co2"
                )
                
                vpd = st.number_input(
                    "Avg VPD (kPa)",
                    min_value=0.1,
                    max_value=3.0,
                    value=0.8,
                    step=0.1,
                    key="single_vpd"
                )
                
                target_pct = st.number_input(
                    "Target Closure (%)",
                    min_value=10.0,
                    max_value=99.0,
                    value=90.0,
                    step=5.0,
                    key="single_target"
                )
            
            st.markdown("### Modifiers (Optional)")
            
            mod1, mod2, mod3 = st.columns(3)
            
            with mod1:
                variety = st.selectbox(
                    "Variety",
                    ["Green (default)", "Red variety", "Romaine/Cos"],
                    index=0,
                    key="single_variety"
                )
            
            with mod2:
                inconsistent_climate = st.checkbox(
                    "Inconsistent climate",
                    value=False,
                    help="Check if climate conditions are inconsistent (10% slower)",
                    key="single_climate"
                )
            
            with mod3:
                known_stress = st.checkbox(
                    "Known stress period",
                    value=False,
                    help="Check if there's a known stress period (15% slower)",
                    key="single_stress"
                )
            
            # Compute speed multiplier
            variety_mult = {
                "Green (default)": 1.00,
                "Red variety": 0.80,
                "Romaine/Cos": 0.95
            }[variety]
            
            climate_mult = 0.90 if inconsistent_climate else 1.00
            stress_mult = 0.85 if known_stress else 1.00
            speed_multiplier = variety_mult * climate_mult * stress_mult
            
            # Compute results
            result = cc.canopy_days_to_target(
                D=density,
                T=temp,
                DLI=dli,
                CO2=co2,
                VPD=vpd,
                target_pct=target_pct,
                speed_mult=speed_multiplier
            )
            
            # Display results
            st.markdown("### Results")
            
            res1, res2, res3 = st.columns(3)
            
            with res1:
                st.metric("PDI (raw)", f"{result['pdi_raw']:.3f}")
                st.metric("PDI (clipped)", f"{result['pdi']:.3f}")
            
            with res2:
                st.metric("t90 anchor (days)", f"{result['t90_anchor']:.1f}")
                st.metric("Alpha (buffering)", f"{result['alpha']:.3f}")
            
            with res3:
                st.metric("Speed multiplier", f"{result['speed_multiplier']:.2f}")
                st.metric("Effective t90 (days)", f"{result['effective_t90']:.1f}")
            
            st.markdown("---")
            
            st.markdown(f"### **Days to {target_pct:.0f}% Closure: {result['t_target']:.1f} days**")
            
            # Generate and plot curve
            curve_data = cc.closure_curve(
                t90=result['effective_t90'],
                target_pct=target_pct
            )
            
            fig = go.Figure()
            
            # Main curve
            fig.add_trace(go.Scatter(
                x=curve_data['day'],
                y=curve_data['closure_pct'],
                mode='lines',
                name='Closure %',
                line=dict(color='#45C96B', width=3)
            ))
            
            # Target line
            fig.add_hline(
                y=target_pct,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Target: {target_pct:.0f}%",
                annotation_position="right"
            )
            
            # Target point
            fig.add_trace(go.Scatter(
                x=[result['t_target']],
                y=[target_pct],
                mode='markers',
                name=f'{target_pct:.0f}% closure',
                marker=dict(size=10, color='#ED695D')
            ))
            
            # 90% reference if different from target
            if abs(target_pct - 90.0) > 1.0:
                fig.add_trace(go.Scatter(
                    x=[result['effective_t90']],
                    y=[90.0],
                    mode='markers',
                    name='90% closure (ref)',
                    marker=dict(size=8, color='#8C8BFF', symbol='diamond')
                ))
            
            fig.update_layout(
                title=f"Canopy Closure Curve (Density: {density:.0f} plants/m²)",
                xaxis_title="Days",
                yaxis_title="Canopy Closure (%)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show intermediate calculations
            with st.expander("Show intermediate calculations", expanded=False):
                st.markdown(f"""
                **Step 1: Normalized PDI**
                - Standard env: T={cc.T_REF}°C, DLI={cc.DLI_REF}, CO2={cc.CO2_REF}ppm, VPD={cc.VPD_REF}kPa
                - Your env: T={temp}°C, DLI={dli}, CO2={co2}ppm, VPD={vpd}kPa
                - PDI raw: {result['pdi_raw']:.3f}
                - PDI clipped [0.70, 1.30]: {result['pdi']:.3f}
                
                **Step 2: Density anchor at PDI=1**
                - Density: {density:.0f} plants/m²
                - t90_anchor: {result['t90_anchor']:.1f} days (days to 90% at standard env)
                
                **Step 3: Density buffering**
                - Alpha: {result['alpha']:.3f} (sensitivity to environment)
                - Higher density → lower alpha → less sensitive
                
                **Step 4: Environment adjustment**
                - t90 = t90_anchor × PDI^(-alpha)
                - t90 = {result['t90_anchor']:.1f} × {result['pdi']:.3f}^(-{result['alpha']:.3f})
                - t90 = {result['t90']:.1f} days
                
                **Step 5: Modifiers**
                - Variety: {variety} → {variety_mult:.2f}
                - Climate: {"Inconsistent" if inconsistent_climate else "Consistent"} → {climate_mult:.2f}
                - Stress: {"Yes" if known_stress else "No"} → {stress_mult:.2f}
                - Combined: {speed_multiplier:.2f}
                - Effective t90 = {result['t90']:.1f} / {speed_multiplier:.2f} = {result['effective_t90']:.1f} days
                
                **Step 6: Target closure**
                - Growth constant k = ln(10) / t90 = {result['k']:.4f}
                - Days to {target_pct:.0f}%: {result['t_target']:.1f} days
                """)
        
        # ===== MULTI-STAGE MODE =====
        with mode_tabs[1]:
            st.markdown("### Multi-Stage Transplant Mode")
            st.markdown(
                """
                Use this mode when you have multiple transplant stages at different densities.
                For example: plug stage → first transplant → final spacing.
                """
            )
            
            # Environment inputs (shared for all stages)
            st.markdown("#### Environmental Conditions (applies to all stages)")
            
            col1, col2 = st.columns(2)
            
            with col1:
                temp_multi = st.number_input(
                    "Avg Temperature (°C)",
                    min_value=-10.0,
                    max_value=45.0,
                    value=23.0,
                    step=0.5,
                    key="multi_temp"
                )
                
                dli_multi = st.number_input(
                    "Avg DLI (mol m⁻² d⁻¹)",
                    min_value=0.1,
                    max_value=60.0,
                    value=18.0,
                    step=0.5,
                    key="multi_dli"
                )
            
            with col2:
                co2_multi = st.number_input(
                    "Avg CO2 (ppm)",
                    min_value=300.0,
                    max_value=2000.0,
                    value=800.0,
                    step=50.0,
                    key="multi_co2"
                )
                
                vpd_multi = st.number_input(
                    "Avg VPD (kPa)",
                    min_value=0.1,
                    max_value=3.0,
                    value=0.8,
                    step=0.1,
                    key="multi_vpd"
                )
            
            target_pct_multi = st.number_input(
                "Target Closure (%)",
                min_value=10.0,
                max_value=99.0,
                value=90.0,
                step=5.0,
                key="multi_target"
            )
            
            # Modifiers
            st.markdown("#### Modifiers")
            
            mod1, mod2, mod3 = st.columns(3)
            
            with mod1:
                variety_multi = st.selectbox(
                    "Variety",
                    ["Green (default)", "Red variety", "Romaine/Cos"],
                    index=0,
                    key="multi_variety"
                )
            
            with mod2:
                inconsistent_climate_multi = st.checkbox(
                    "Inconsistent climate",
                    value=False,
                    key="multi_climate"
                )
            
            with mod3:
                known_stress_multi = st.checkbox(
                    "Known stress period",
                    value=False,
                    key="multi_stress"
                )
            
            # Compute speed multiplier
            variety_mult_multi = {
                "Green (default)": 1.00,
                "Red variety": 0.80,
                "Romaine/Cos": 0.95
            }[variety_multi]
            
            climate_mult_multi = 0.90 if inconsistent_climate_multi else 1.00
            stress_mult_multi = 0.85 if known_stress_multi else 1.00
            speed_multiplier_multi = variety_mult_multi * climate_mult_multi * stress_mult_multi
            
            # Stage inputs
            st.markdown("#### Transplant Stages")
            st.markdown("Define each stage with density and days. Last stage goes until target closure.")
            
            num_stages = st.number_input(
                "Number of stages",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
                key="num_stages"
            )
            
            stages = []
            for i in range(num_stages):
                st.markdown(f"**Stage {i+1}**")
                c1, c2 = st.columns(2)
                
                with c1:
                    stage_density = st.number_input(
                        f"Density (plants/m²)",
                        min_value=1.0,
                        max_value=2000.0,
                        value=1200.0 if i == 0 else (600.0 if i == 1 else 350.0),
                        step=10.0,
                        key=f"stage_{i}_density"
                    )
                
                with c2:
                    if i < num_stages - 1:
                        stage_days = st.number_input(
                            f"Days at this stage",
                            min_value=0.0,
                            max_value=100.0,
                            value=7.0 if i == 0 else 14.0,
                            step=1.0,
                            key=f"stage_{i}_days"
                        )
                    else:
                        st.markdown("**(Final stage: until target)**")
                        stage_days = None
                
                stages.append((stage_density, stage_days))
            
            # Compute multi-stage results
            multi_result = cc.canopy_days_multistage(
                stages=stages,
                T=temp_multi,
                DLI=dli_multi,
                CO2=co2_multi,
                VPD=vpd_multi,
                target_pct=target_pct_multi,
                speed_mult=speed_multiplier_multi
            )
            
            # Display results
            st.markdown("### Results")
            
            st.markdown(f"### **Total Days to {target_pct_multi:.0f}% Closure: {multi_result['total_days']:.1f} days**")
            st.markdown(f"**Final Closure: {multi_result['final_closure_pct']:.1f}%**")
            
            st.markdown("#### Stage Breakdown")
            
            # Create table of stages
            stage_table = []
            for stage_info in multi_result['stages_info']:
                stage_table.append({
                    'Stage': stage_info['stage'],
                    'Density (plants/m²)': f"{stage_info['density']:.0f}",
                    'Days': f"{stage_info['days']:.1f}",
                    'Closure Start (%)': f"{stage_info['closure_start']:.1f}",
                    'Closure End (%)': f"{stage_info['closure_end']:.1f}",
                    'Cumulative Days': f"{stage_info['cumulative_days']:.1f}"
                })
            
            st.dataframe(pd.DataFrame(stage_table), use_container_width=True)
            
            # Plot multi-stage curve
            fig_multi = go.Figure()
            
            for stage_info in multi_result['stages_info']:
                days_start = stage_info['cumulative_days'] - stage_info['days']
                days_end = stage_info['cumulative_days']
                
                # Generate curve for this stage
                k = stage_info['k']
                days_range = np.linspace(0, stage_info['days'], 50)
                closure_base = 100 * (1 - np.exp(-k * days_range))
                
                # Adjust for starting point
                closure_adjusted = stage_info['closure_start'] + \
                                 closure_base * (1 - stage_info['closure_start']/100.0)
                days_absolute = days_start + days_range
                
                fig_multi.add_trace(go.Scatter(
                    x=days_absolute,
                    y=closure_adjusted,
                    mode='lines',
                    name=f'Stage {stage_info["stage"]} (D={stage_info["density"]:.0f})',
                    line=dict(width=3)
                ))
            
            # Target line
            fig_multi.add_hline(
                y=target_pct_multi,
                line_dash="dash",
                line_color="gray",
                annotation_text=f"Target: {target_pct_multi:.0f}%",
                annotation_position="right"
            )
            
            fig_multi.update_layout(
                title="Multi-Stage Canopy Closure",
                xaxis_title="Days",
                yaxis_title="Canopy Closure (%)",
                hovermode='x unified',
                height=450
            )
            
            st.plotly_chart(fig_multi, use_container_width=True)


class UnitConverterCalculator:
    """
    Generic unit converter for common greenhouse / agronomy-relevant quantities:
    - Length
    - Area
    - Volume
    - Mass
    - Temperature
    """

    # Base-unit scaling factors for multiplicative conversions
    LENGTH_FACTORS = {
        "m": 1.0,
        "cm": 0.01,
        "mm": 0.001,
        "ft": 0.3048,
        "in": 0.0254,
    }

    AREA_FACTORS = {
        "m²": 1.0,
        "ft²": 0.09290304,
        "ha": 10_000.0,
        "acre": 4046.8564224,
    }

    VOLUME_FACTORS = {
        "L": 0.001,           # 1 L = 0.001 m³
        "m³": 1.0,
        "gal (US)": 0.00378541,
        "ft³": 0.0283168,
    }

    MASS_FACTORS = {
        "g": 0.001,    # 1 g = 0.001 kg
        "kg": 1.0,
        "lb": 0.45359237,
        "oz": 0.0283495231,
    }

    @staticmethod
    def convert_length(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.LENGTH_FACTORS[from_unit]
        return base / UnitConverterCalculator.LENGTH_FACTORS[to_unit]

    @staticmethod
    def convert_area(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.AREA_FACTORS[from_unit]
        return base / UnitConverterCalculator.AREA_FACTORS[to_unit]

    @staticmethod
    def convert_volume(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.VOLUME_FACTORS[from_unit]
        return base / UnitConverterCalculator.VOLUME_FACTORS[to_unit]

    @staticmethod
    def convert_mass(value: float, from_unit: str, to_unit: str) -> float:
        base = value * UnitConverterCalculator.MASS_FACTORS[from_unit]
        return base / UnitConverterCalculator.MASS_FACTORS[to_unit]

    @staticmethod
    def convert_temperature(value: float, from_unit: str, to_unit: str) -> float:
        # Convert to °C first
        if from_unit == "°C":
            c = value
        elif from_unit == "°F":
            c = (value - 32.0) * 5.0 / 9.0
        elif from_unit == "K":
            c = value - 273.15
        else:
            c = value

        # Convert from °C to target
        if to_unit == "°C":
            return c
        elif to_unit == "°F":
            return c * 9.0 / 5.0 + 32.0
        elif to_unit == "K":
            return c + 273.15
        else:
            return c

    @classmethod
    def render(cls):
        st.subheader("Unit Converter")

        st.markdown(
            """
            A small helper for converting common greenhouse and agronomy units.

            Choose a quantity type, enter a value, and select the units to convert from and to.
            """
        )

        quantity_type = st.selectbox(
            "Quantity type",
            ["Length", "Area", "Volume", "Mass", "Temperature"],
            index=0,
        )

        if quantity_type == "Length":
            units = list(cls.LENGTH_FACTORS.keys())
        elif quantity_type == "Area":
            units = list(cls.AREA_FACTORS.keys())
        elif quantity_type == "Volume":
            units = list(cls.VOLUME_FACTORS.keys())
        elif quantity_type == "Mass":
            units = list(cls.MASS_FACTORS.keys())
        else:
            units = ["°C", "°F", "K"]

        col1, col2, col3 = st.columns([1, 1, 1.2])

        with col1:
            value = st.number_input("Value", value=1.0, step=0.1)

        with col2:
            from_unit = st.selectbox("From", units, index=0)

        with col3:
            to_unit = st.selectbox("To", units, index=min(1, len(units) - 1))

        result = None
        if quantity_type == "Length":
            result = cls.convert_length(value, from_unit, to_unit)
        elif quantity_type == "Area":
            result = cls.convert_area(value, from_unit, to_unit)
        elif quantity_type == "Volume":
            result = cls.convert_volume(value, from_unit, to_unit)
        elif quantity_type == "Mass":
            result = cls.convert_mass(value, from_unit, to_unit)
        elif quantity_type == "Temperature":
            result = cls.convert_temperature(value, from_unit, to_unit)

        if result is not None:
            st.markdown("### Result")
            st.write(f"**{value} {from_unit} = {result:.4g} {to_unit}**")
