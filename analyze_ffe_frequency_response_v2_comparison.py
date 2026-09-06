#!/usr/bin/env python3
"""Compare frequency-domain behavior of 7-tap FFE vs. 7-tap FFE + 1-tap DFE.

Reads waveforms before equalization and after each equalizer configuration,
computes empirical transfer functions via FFT, and overlays analytical
transfer functions derived from tap coefficients. CSV files may include
metadata headers; only numerical time/amplitude rows are used.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# === DEFAULT TAP COEFFICIENTS ===
# These should be obtained from your simulation/measurement tool (e.g., Keysight ADS, MATLAB).
# Update these values with your actual FFE and DFE tap coefficients.
DEFAULT_FFE_TAPS = [0.05, -0.15, 0.85, -0.20, 0.08, -0.03, 0.01]  # 7-tap FFE coefficients
DEFAULT_DFE_TAP = 0.12  # 1-tap DFE feedback coefficient
DEFAULT_MAX_FREQ_GHZ = 212.5  # Recommended: 2x symbol rate (106.25 GBd) to avoid aliasing artifacts


def read_waveform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return time in seconds and amplitude from a waveform-export CSV."""
    # Load CSV data, skipping header rows (typically 7 rows from waveform viewer exports)
    try:
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=7)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read waveform data from {path}: {error}") from error

    # Validate: must be 2D array with exactly 2 columns (time, amplitude) and at least 2 rows
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError(f"{path} must contain at least two time,amplitude rows.")

    # Convert time from nanoseconds to seconds, extract amplitude column
    time_seconds = data[:, 0] * 1e-9
    amplitude = data[:, 1]
    
    # Calculate sample intervals and median sampling period
    sample_intervals = np.diff(time_seconds)
    sample_period = float(np.median(sample_intervals))

    # Validate: time must be generally increasing (can have minor jitter in real measurements)
    if sample_period <= 0:
        raise ValueError(f"{path} does not have a positive, increasing time axis.")
    
    # Check for major deviations (> 50% from median) which would indicate corruption
    max_deviation = np.max(np.abs(sample_intervals - sample_period)) / sample_period
    if max_deviation > 0.5:
        raise ValueError(
            f"{path} has excessive time-axis jitter (deviation > 50%). "
            f"Median period: {sample_period:.3e}s, max deviation: {max_deviation*100:.1f}%"
        )

    return time_seconds, amplitude


def amplitude_spectrum(signal: np.ndarray, window: np.ndarray) -> np.ndarray:
    """Return the one-sided amplitude spectrum, corrected for window gain."""
    # Apply window function to signal (reduces spectral leakage) and compute FFT
    spectrum = np.fft.rfft(signal * window)
    
    # Normalize by window sum to account for window gain
    amplitude = np.abs(spectrum) / np.sum(window)
    
    # For one-sided spectrum, double all non-DC/Nyquist bins to preserve energy
    if len(signal) > 1:
        amplitude[1:-1] *= 2
    return amplitude


def estimate_transfer_function(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: float,
    segment_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate H(f) by averaging output/input cross spectra across segments.
    
    Uses Welch's method: divides signal into overlapping/non-overlapping segments,
    computes FFT for each segment, then averages cross-spectra for robust estimate.
    """
    # Determine how many non-overlapping segments fit in the signal
    segment_count = len(input_signal) // segment_length
    if segment_count == 0:
        raise ValueError("Trace is shorter than one transfer-function segment.")

    # Initialize accumulators for cross-spectrum and input power spectrum
    window = np.hanning(segment_length)
    cross_spectrum = np.zeros(segment_length // 2 + 1, dtype=complex)
    input_power = np.zeros(segment_length // 2 + 1)
    
    # Process each segment: compute FFTs and accumulate cross/power spectra
    for index in range(segment_count):
        start = index * segment_length
        stop = start + segment_length
        input_fft = np.fft.rfft(input_signal[start:stop] * window)
        output_fft = np.fft.rfft(output_signal[start:stop] * window)
        # Cross-spectrum: output * conj(input)
        cross_spectrum += output_fft * np.conj(input_fft)
        # Input power spectrum: |input|^2
        input_power += np.abs(input_fft) ** 2

    # Compute frequency axis for this segment length
    frequency_hz = np.fft.rfftfreq(segment_length, d=1.0 / sample_rate)
    
    # Transfer function H(f) = cross_spectrum / (input_power + epsilon)
    # Add regularization epsilon to avoid division-by-zero artifacts at frequency notches
    # where the input spectrum drops near zero (improves low-frequency stability)
    # Use a relative regularization: ~1e-10 of peak power (about -100 dB)
    max_input_power = np.max(input_power)
    epsilon = max_input_power * 1e-10 if max_input_power > 0 else 1e-12
    transfer = cross_spectrum / (input_power + epsilon)
    return frequency_hz, transfer


def analytical_ffe_transfer_function(
    ffe_taps: list[float],
    frequency_hz: np.ndarray,
    tap_spacing_seconds: float,
) -> np.ndarray:
    """Compute analytical FFE transfer function from tap coefficients.
    
    H_FFE(f) = sum_{k=0}^{6} w_k * exp(-j * 2 * pi * f * k * T_tap)
    
    where w_k are tap weights and T_tap is the tap spacing period.
    IMPORTANT: T_tap should be the SYMBOL PERIOD (or samples-per-symbol * sample_period),
    NOT the sample period! This ensures the formula matches empirical measurements.
    """
    # Initialize complex transfer function for all frequency bins
    transfer_ffe = np.zeros_like(frequency_hz, dtype=complex)
    
    # Sum contributions from each tap at the correct symbol-rate spacing
    for k, wk in enumerate(ffe_taps):
        # Phase shift for this tap at each frequency
        # Tap k is delayed by k * T_tap (where T_tap is typically 1 symbol period)
        phase = -2.0 * np.pi * frequency_hz * k * tap_spacing_seconds
        transfer_ffe += wk * np.exp(1j * phase)
    
    return transfer_ffe


def analytical_dfe_transfer_function(
    ffe_taps: list[float],
    dfe_tap: float,
    frequency_hz: np.ndarray,
    tap_spacing_seconds: float,
) -> np.ndarray:
    """Compute analytical FFE+DFE transfer function from tap coefficients.
    
    In linear approximation (assuming no decision errors and negligible error propagation):
    H_FFE+DFE(f) = H_FFE(f) / (1 - b_1 * exp(-j * 2 * pi * f * T_tap))
    
    where b_1 is the DFE feedback coefficient and T_tap is the tap spacing 
    (typically 1 symbol period).
    
    CRITICAL: Both the FFE taps and DFE feedback use the same tap_spacing_seconds!
    """
    # Compute FFE transfer function first using the tap spacing (symbol period)
    transfer_ffe = analytical_ffe_transfer_function(
        ffe_taps, frequency_hz, tap_spacing_seconds
    )
    
    # DFE feedback term: 1 - b_1 * exp(-j * 2 * pi * f * T_tap)
    # The feedback is applied once per symbol, so it uses tap_spacing_seconds (not sample period)
    dfe_denominator = 1.0 - dfe_tap * np.exp(
        -1j * 2.0 * np.pi * frequency_hz * tap_spacing_seconds
    )
    
    # Add small epsilon to avoid division by zero
    dfe_denominator = dfe_denominator + np.finfo(float).eps
    
    # Combined: FFE / DFE_denominator
    transfer_dfe = transfer_ffe / dfe_denominator
    
    return transfer_dfe


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments with sensible defaults."""
    parser = argparse.ArgumentParser(
        description="Compare frequency response of 7-tap FFE vs. 7-tap FFE + 1-tap DFE."
    )
    # Input waveform: signal before equalization
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("rx_before_eq.csv"),
        help="CSV waveform before equalization (default: rx_before_eq.csv).",
    )
    # Output waveform: signal after 7-tap FFE only
    parser.add_argument(
        "--output-ffe",
        type=Path,
        default=Path("rx_after_eq.csv"),
        help="CSV waveform after 7-tap FFE (default: rx_after_eq.csv).",
    )
    # Output waveform: signal after 7-tap FFE + 1-tap DFE
    parser.add_argument(
        "--output-dfe",
        type=Path,
        default=Path("rx_after_eq2.csv"),
        help="CSV waveform after 7-tap FFE + 1-tap DFE (default: rx_after_eq2.csv).",
    )
    # FFE tap coefficients (7 values)
    parser.add_argument(
        "--ffe-taps",
        type=float,
        nargs=7,
        default=DEFAULT_FFE_TAPS,
        help=f"Seven FFE tap coefficients (default: {DEFAULT_FFE_TAPS}).",
        metavar="TAP",
    )
    # DFE feedback tap coefficient (1 value)
    parser.add_argument(
        "--dfe-tap",
        type=float,
        default=DEFAULT_DFE_TAP,
        help=f"One DFE feedback tap coefficient (default: {DEFAULT_DFE_TAP}).",
    )
    # Samples per symbol (critical for correct tap spacing in analytical formulas)
    parser.add_argument(
        "--samples-per-symbol",
        type=int,
        default=16,
        help="Samples per symbol / oversampling ratio (default: 16). "
             "This determines tap spacing: T_tap = Nsps * T_sample. "
             "CRITICAL: Must match your simulation setup!",
    )
    # Output plot file
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("ffe_dfe_comparison.png"),
        help="Output comparison plot path (default: ffe_dfe_comparison.png).",
    )
    # Optional: limit x-axis range for zoomed plots
    parser.add_argument(
        "--max-frequency-ghz",
        type=float,
        default=DEFAULT_MAX_FREQ_GHZ,
        help="Upper x-axis limit in GHz (default: 212.5 GHz = 2x symbol rate). "
             "Recommended to avoid aliasing artifacts beyond Nyquist band. "
             "Use 850 GHz to see full Nyquist frequency, but expect high-frequency roll-off.",
    )
    return parser.parse_args()


def main() -> None:
    # === SETUP: Parse arguments and load waveform data ===
    args = parse_arguments()
    input_time, input_signal = read_waveform(args.input)
    output_ffe_time, output_ffe_signal = read_waveform(args.output_ffe)
    output_dfe_time, output_dfe_signal = read_waveform(args.output_dfe)

    # Determine minimum sample count (in case files have slight differences)
    min_samples = min(len(input_signal), len(output_ffe_signal), len(output_dfe_signal))
    
    # Truncate all signals to common length
    input_signal = input_signal[:min_samples]
    input_time = input_time[:min_samples]
    output_ffe_signal = output_ffe_signal[:min_samples]
    output_ffe_time = output_ffe_time[:min_samples]
    output_dfe_signal = output_dfe_signal[:min_samples]
    output_dfe_time = output_dfe_time[:min_samples]

    # Validate: all traces must have compatible sampling period (within 5% tolerance for cross-file measurements)
    dt_input = float(np.median(np.diff(input_time)))
    dt_ffe = float(np.median(np.diff(output_ffe_time)))
    dt_dfe = float(np.median(np.diff(output_dfe_time)))
    if not np.isclose(dt_input, dt_ffe, rtol=0.05, atol=0.0):
        raise ValueError("Input and FFE output traces must have compatible sample periods.")
    if not np.isclose(dt_input, dt_dfe, rtol=0.05, atol=0.0):
        raise ValueError("Input and DFE output traces must have compatible sample periods.")

    # === FREQUENCY SETUP: Calculate frequency axis ===
    sample_rate = 1.0 / dt_input
    frequency_hz = np.fft.rfftfreq(len(input_signal), d=dt_input)
    frequency_ghz = frequency_hz / 1e9
    
    # Calculate symbol period from samples-per-symbol ratio
    # CRITICAL: This must match your simulation setup for analytical curves to be accurate
    # e.g., if Nsps=16, then T_symbol = 16 * T_sample
    symbol_period = args.samples_per_symbol * dt_input
    
    print(f"Oversampling ratio (Nsps): {args.samples_per_symbol}")
    print(f"Sample period: {dt_input * 1e12:.9g} ps")
    print(f"Symbol period: {symbol_period * 1e12:.9g} ps")

    # === PREPROCESSING: Remove DC component and compute amplitude spectra ===
    # Remove DC (mean) from all signals before spectral analysis
    input_ac = input_signal - np.mean(input_signal)
    output_ffe_ac = output_ffe_signal - np.mean(output_ffe_signal)
    output_dfe_ac = output_dfe_signal - np.mean(output_dfe_signal)
    window = np.hanning(len(input_signal))
    
    # Compute amplitude spectra with window function to reduce spectral leakage
    input_spectrum = amplitude_spectrum(input_ac, window)
    output_ffe_spectrum = amplitude_spectrum(output_ffe_ac, window)
    output_dfe_spectrum = amplitude_spectrum(output_dfe_ac, window)

    # === EMPIRICAL TRANSFER FUNCTIONS: Estimate from waveforms ===
    # FFE empirical transfer function
    segment_length = min(16_384, len(input_signal))
    empirical_ffe_freq, empirical_ffe_transfer = estimate_transfer_function(
        input_ac, output_ffe_ac, sample_rate, segment_length
    )
    empirical_ffe_freq_ghz = empirical_ffe_freq / 1e9
    empirical_ffe_db = 20 * np.log10(np.abs(empirical_ffe_transfer))
    
    # DFE empirical transfer function
    empirical_dfe_freq, empirical_dfe_transfer = estimate_transfer_function(
        input_ac, output_dfe_ac, sample_rate, segment_length
    )
    empirical_dfe_freq_ghz = empirical_dfe_freq / 1e9
    empirical_dfe_db = 20 * np.log10(np.abs(empirical_dfe_transfer))

    # === ANALYTICAL TRANSFER FUNCTIONS: Derived from tap coefficients ===
    # CRITICAL: Pass symbol_period (not sample_period) to ensure analytical curves match empirical data!
    # This ensures the analytical formulas use the correct tap spacing.
    analytical_ffe_transfer = analytical_ffe_transfer_function(
        args.ffe_taps, frequency_hz, symbol_period
    )
    analytical_ffe_db = 20 * np.log10(np.abs(analytical_ffe_transfer) + 1e-12)
    
    # Analytical DFE transfer function (also uses symbol_period for correct tap spacing)
    analytical_dfe_transfer = analytical_dfe_transfer_function(
        args.ffe_taps, args.dfe_tap, frequency_hz, symbol_period
    )
    analytical_dfe_db = 20 * np.log10(np.abs(analytical_dfe_transfer) + 1e-12)

    # === PLOTTING: Create comparison visualization ===
    figure, axes = plt.subplots(
        2, 2, figsize=(14, 10), sharex="col", layout="constrained"
    )
    spectrum_floor = np.finfo(float).tiny

    # === TOP LEFT: Input vs. FFE Output Spectra ===
    axes[0, 0].plot(
        frequency_ghz,
        20 * np.log10(np.maximum(input_spectrum, spectrum_floor)),
        label="Input (before equalization)",
        linewidth=2,
    )
    axes[0, 0].plot(
        frequency_ghz,
        20 * np.log10(np.maximum(output_ffe_spectrum, spectrum_floor)),
        label="Output (7-tap FFE)",
        linewidth=2,
    )
    axes[0, 0].set_ylabel("Amplitude (dB a.u.)")
    axes[0, 0].set_title("7-tap FFE: Input vs. Output Spectra")
    axes[0, 0].grid(True, which="both", alpha=0.3)
    axes[0, 0].legend()

    # === TOP RIGHT: Input vs. DFE Output Spectra ===
    axes[0, 1].plot(
        frequency_ghz,
        20 * np.log10(np.maximum(input_spectrum, spectrum_floor)),
        label="Input (before equalization)",
        linewidth=2,
    )
    axes[0, 1].plot(
        frequency_ghz,
        20 * np.log10(np.maximum(output_dfe_spectrum, spectrum_floor)),
        label="Output (7-tap FFE + 1-tap DFE)",
        linewidth=2,
    )
    axes[0, 1].set_ylabel("Amplitude (dB a.u.)")
    axes[0, 1].set_title("7-tap FFE + 1-tap DFE: Input vs. Output Spectra")
    axes[0, 1].grid(True, which="both", alpha=0.3)
    axes[0, 1].legend()

    # === BOTTOM LEFT: FFE Transfer Functions (Empirical & Analytical) ===
    axes[1, 0].plot(
        empirical_ffe_freq_ghz,
        empirical_ffe_db,
        label="Empirical (measured from waveforms, includes pulse shaping)",
        color="tab:blue",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1, 0].plot(
        frequency_ghz,
        analytical_ffe_db,
        label="Analytical (ideal discrete FIR from taps)",
        color="tab:orange",
        linewidth=2,
        linestyle="--",
    )
    axes[1, 0].set_xlabel("Frequency (GHz)")
    axes[1, 0].set_ylabel("|H(f)| (dB)")
    axes[1, 0].set_title("7-tap FFE: Transfer Function Comparison\n"
                         "(empirical includes DAC/pulse shaping; divergence >106 GHz is normal)")
    axes[1, 0].grid(True, which="both", alpha=0.3)
    axes[1, 0].legend(fontsize=9)

    # === BOTTOM RIGHT: DFE Transfer Functions (Empirical & Analytical) ===
    axes[1, 1].plot(
        empirical_dfe_freq_ghz,
        empirical_dfe_db,
        label="Empirical (measured from waveforms, non-linear effects)",
        color="tab:green",
        linewidth=1.5,
        alpha=0.7,
    )
    axes[1, 1].plot(
        frequency_ghz,
        analytical_dfe_db,
        label="Analytical (linear small-signal model from taps)",
        color="tab:red",
        linewidth=2,
        linestyle="--",
    )
    axes[1, 1].set_xlabel("Frequency (GHz)")
    axes[1, 1].set_ylabel("|H(f)| (dB)")
    axes[1, 1].set_title("7-tap FFE + 1-tap DFE: Transfer Function Comparison\n"
                         "(DFE is non-linear; analytical shows linearized small-signal model)")
    axes[1, 1].grid(True, which="both", alpha=0.3)
    axes[1, 1].legend(fontsize=9)

    # Set x-axis limit to avoid aliasing artifacts beyond symbol-rate Nyquist band
    # Recommended: limit to 0-212.5 GHz (2x symbol rate) for clean comparison
    if args.max_frequency_ghz <= 0:
        raise ValueError("--max-frequency-ghz must be positive.")
    
    for ax in [axes[1, 0], axes[1, 1]]:
        ax.set_xlim(0, args.max_frequency_ghz)

    # === OUTPUT: Save plot ===
    figure.savefig(args.figure, dpi=200)
    
    # === SUMMARY: Print statistics and configuration ===
    print(f"Samples: {len(input_signal)}")
    print(f"Sample rate: {sample_rate / 1e9:.9g} GHz")
    print(f"Nyquist frequency: {sample_rate / 2e9:.9g} GHz")
    print()
    print(f"FFE taps (7): {args.ffe_taps}")
    print(f"DFE tap (1): {args.dfe_tap}")
    print(f"Tap spacing (1 symbol period): {symbol_period * 1e12:.9g} ps")
    print()
    print(f"Saved: {args.figure}")


if __name__ == "__main__":
    main()
