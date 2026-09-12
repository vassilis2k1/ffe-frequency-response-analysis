#!/usr/bin/env python3
"""Compare measured and analytical FFE/DFE frequency responses."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FFE_TAPS = [0.05, -0.15, 0.85, -0.20, 0.08, -0.03, 0.01]
DFE_TAP = 0.12
SCRIPT_DIR = Path(__file__).resolve().parent


def read_waveform(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        data = np.loadtxt(path, delimiter=",", comments="#", skiprows=7)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read {path}: {error}") from error
    if data.ndim != 2 or data.shape[1] != 2 or len(data) < 2:
        raise ValueError(f"{path} must contain at least two time,amplitude rows")
    time = data[:, 0] * 1e-9
    intervals = np.diff(time)
    dt = float(np.median(intervals))
    if dt <= 0:
        raise ValueError(f"{path} does not have an increasing time axis")
    jitter = np.max(np.abs(intervals - dt)) / dt
    if jitter > 0.5:
        raise ValueError(f"{path} has excessive time-axis jitter")
    return time, data[:, 1]


def transfer(input_signal: np.ndarray, output_signal: np.ndarray, fs: float, size: int):
    size = min(size, len(input_signal))
    count = len(input_signal) // size
    if not count:
        raise ValueError("Trace is shorter than one transfer-function segment")
    window = np.hanning(size)
    power = np.zeros(size // 2 + 1)
    cross = np.zeros(size // 2 + 1, dtype=complex)
    for index in range(count):
        start = index * size
        stop = start + size
        input_fft = np.fft.rfft(input_signal[start:stop] * window)
        output_fft = np.fft.rfft(output_signal[start:stop] * window)
        power += np.abs(input_fft) ** 2
        cross += output_fft * np.conj(input_fft)
    epsilon = max(np.max(power) * 1e-10, 1e-12)
    return np.fft.rfftfreq(size, 1 / fs), cross / (power + epsilon)


def spectrum(values: np.ndarray) -> np.ndarray:
    window = np.hanning(len(values))
    result = np.abs(np.fft.rfft((values - values.mean()) * window)) / window.sum()
    result[1:-1] *= 2
    return result


def analytical(frequency: np.ndarray, taps: list[float], dfe: float, symbol_period: float):
    tap_index = np.arange(len(taps))
    ffe = np.dot(
        np.exp(-2j * np.pi * np.outer(frequency, tap_index) * symbol_period), taps
    )
    dfe_denominator = 1 - dfe * np.exp(-2j * np.pi * frequency * symbol_period)
    return ffe, ffe / (dfe_denominator + np.finfo(float).eps)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "rx_before_eq.csv")
    parser.add_argument("--output-ffe", type=Path, default=SCRIPT_DIR / "rx_after_eq.csv")
    parser.add_argument("--output-dfe", type=Path, default=SCRIPT_DIR / "rx_after_eq2.csv")
    parser.add_argument("--ffe-taps", type=float, nargs=7, default=FFE_TAPS)
    parser.add_argument("--dfe-tap", type=float, default=DFE_TAP)
    parser.add_argument("--samples-per-symbol", type=int, default=16)
    parser.add_argument("--max-frequency-ghz", type=float, default=212.5)
    parser.add_argument("--figure", type=Path, default=SCRIPT_DIR / "ffe_dfe_comparison.png")
    return parser.parse_args()


def main() -> None:
    args = arguments()
    traces = [read_waveform(path) for path in (args.input, args.output_ffe, args.output_dfe)]
    dt = float(np.median(np.diff(traces[0][0])))
    for time, _ in traces[1:]:
        if not np.isclose(dt, np.median(np.diff(time)), rtol=0.05):
            raise ValueError("All traces must have compatible sample periods")

    count = min(len(amplitude) for _, amplitude in traces)
    signals = [amplitude[:count] - amplitude[:count].mean() for _, amplitude in traces]
    sample_rate = 1 / dt
    symbol_period = args.samples_per_symbol * dt
    frequency = np.fft.rfftfreq(count, dt)
    transfer_frequency, ffe_empirical = transfer(signals[0], signals[1], sample_rate, 16384)
    _, dfe_empirical = transfer(signals[0], signals[2], sample_rate, 16384)
    ffe_analytical, dfe_analytical = analytical(
        frequency, args.ffe_taps, args.dfe_tap, symbol_period
    )

    spectra = [spectrum(signal) for signal in signals]
    db = lambda values: 20 * np.log10(np.abs(values) + 1e-12)
    figure, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    for axis, output, title in zip(
        axes[0], spectra[1:], ("7-tap FFE", "7-tap FFE + 1-tap DFE")
    ):
        axis.plot(frequency / 1e9, db(spectra[0]), label="Input")
        axis.plot(frequency / 1e9, db(output), label="Output")
        axis.set(ylabel="Amplitude (dB a.u.)", title=f"{title}: Input vs Output")

    for axis, measured, ideal, title in zip(
        axes[1],
        (ffe_empirical, dfe_empirical),
        (ffe_analytical, dfe_analytical),
        ("7-tap FFE", "7-tap FFE + 1-tap DFE"),
    ):
        axis.plot(transfer_frequency / 1e9, db(measured), label="Empirical")
        axis.plot(frequency / 1e9, db(ideal), "--", label="Analytical", linewidth=2)
        axis.set(xlabel="Frequency (GHz)", ylabel="|H(f)| (dB)", title=title)

    if args.max_frequency_ghz <= 0:
        raise ValueError("--max-frequency-ghz must be positive")
    for axis in axes.flat:
        axis.set_xlim(0, args.max_frequency_ghz)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=9)
    figure.savefig(args.figure, dpi=200)
    print(f"Samples: {count}")
    print(f"Sample rate: {sample_rate / 1e9:.9g} GHz")
    print(f"Saved: {args.figure}")


if __name__ == "__main__":
    main()