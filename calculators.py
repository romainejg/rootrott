# calculators.py

import streamlit as st
import numpy as np


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
