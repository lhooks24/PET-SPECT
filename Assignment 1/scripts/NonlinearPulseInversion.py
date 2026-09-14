"""Pulse-inversion recovery of a nonlinear echo hidden by a linear reflector.

Two acquisitions are made with otherwise identical positive and negative
transmit pulses. Linear echoes reverse sign; an even-order nonlinear target
response does not. Their half-sum therefore cancels linear clutter without
requiring a stored clutter waveform or an isolated measurement of either echo.

This is an educational 1D signal/propagation model. The nonlinear target is
representative of a microbubble-like or other even-order scatterer, not the
linear reverberation C in 1DWaveSimulationLayered.py.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


F0 = 2.0e6
SOUND_SPEED = 1540.0


@dataclass(frozen=True)
class Settings:
    sample_rate: float = 40.0e6
    duration: float = 100.0e-6
    pulse_center: float = 7.0e-6
    reflector_depth: float = 42.0e-3
    nonlinear_depth: float = 42.5e-3
    linear_amplitude: float = 1.0
    nonlinear_amplitude: float = 0.075
    attenuation_db_cm_mhz: float = 0.50
    receive_fractional_bandwidth: float = 1.50


def parse_args():
    p = argparse.ArgumentParser(
        description="Reveal a nonlinear echo hidden beneath a linear reflection")
    p.add_argument("--reflector-depth-mm", type=float, default=42.0)
    p.add_argument("--target-depth-mm", type=float, default=42.5)
    p.add_argument("--nonlinear-fraction", type=float, default=0.075,
                   help="nonlinear echo amplitude relative to the linear echo")
    p.add_argument("--noise-fraction", type=float, default=0.015,
                   help="receiver noise RMS relative to the linear echo peak")
    p.add_argument("--inversion-gain-error", type=float, default=0.0,
                   help="fractional amplitude error in the inverted transmit")
    p.add_argument("--inversion-time-error-ns", type=float, default=0.0,
                   help="timing error in the inverted acquisition")
    p.add_argument("--seed", type=int, default=20260914)
    p.add_argument("--save-figure", type=Path)
    p.add_argument("--save-animation", type=Path)
    p.add_argument("--save-csv", type=Path)
    p.add_argument("--animation-fps", type=int, default=12)
    p.add_argument("--no-show", action="store_true")
    return p.parse_args()


def gaussian_bandpass(frequency, center, fractional_bandwidth):
    half_width = 0.5 * fractional_bandwidth * center
    sigma = half_width / np.sqrt(2.0 * np.log(2.0))
    return np.exp(-0.5 * ((np.abs(frequency) - center) / sigma) ** 2)


def attenuation(frequency, distance, coefficient):
    f_mhz = np.abs(frequency) / 1e6
    distance_cm = distance * 100.0
    amplitude_db = coefficient * f_mhz * distance_cm
    return 10.0 ** (-amplitude_db / 20.0)


def transmit_pulse(time, sign=1.0):
    duration = 3.0 / F0
    sigma = duration / 3.0
    centered = time - Settings.pulse_center
    return sign * np.sin(2 * np.pi * F0 * centered) * np.exp(
        -0.5 * (centered / sigma) ** 2)


def delay_and_filter(signal, frequency, delay, distance, settings,
                     center_frequency):
    receive = gaussian_bandpass(
        frequency, center_frequency, settings.receive_fractional_bandwidth)
    loss = attenuation(frequency, 2.0 * distance,
                       settings.attenuation_db_cm_mhz)
    phase = np.exp(-2j * np.pi * frequency * delay)
    return np.fft.irfft(np.fft.rfft(signal) * receive * loss * phase,
                        n=signal.size)


def simulate(args, target_depth_m=None, noise_pair=None):
    settings = Settings(
        reflector_depth=args.reflector_depth_mm * 1e-3,
        nonlinear_depth=(args.target_depth_mm * 1e-3
                         if target_depth_m is None else target_depth_m),
        nonlinear_amplitude=args.nonlinear_fraction)
    if min(settings.reflector_depth, settings.nonlinear_depth) <= 0:
        raise ValueError("reflector and target depths must be positive")
    if settings.nonlinear_amplitude <= 0 or args.noise_fraction < 0:
        raise ValueError("amplitudes must be positive and noise non-negative")
    if args.inversion_gain_error <= -1.0:
        raise ValueError("--inversion-gain-error must be greater than -1")
    n = int(settings.sample_rate * settings.duration)
    time = np.arange(n) / settings.sample_rate
    frequency = np.fft.rfftfreq(n, 1.0 / settings.sample_rate)
    positive = transmit_pulse(time, +1.0)
    negative = transmit_pulse(time, -1.0)

    reflector_delay = 2.0 * settings.reflector_depth / SOUND_SPEED
    target_delay = 2.0 * settings.nonlinear_depth / SOUND_SPEED
    linear_positive = delay_and_filter(
        positive, frequency, reflector_delay, settings.reflector_depth,
        settings, F0)
    linear_positive /= max(np.max(np.abs(linear_positive)), 1e-15)

    # Imperfect inversion is applied only to the second acquisition. A timing
    # or gain mismatch leaves residual linear clutter after pulse inversion.
    timing_error = args.inversion_time_error_ns * 1e-9
    inverted_phase = np.exp(-2j * np.pi * frequency * timing_error)
    linear_negative = np.fft.irfft(
        np.fft.rfft(-linear_positive) * inverted_phase,
        n=n) * (1.0 + args.inversion_gain_error)

    # Squaring creates a sign-invariant even-order response containing a strong
    # component near 2*f0. Removing its mean suppresses the zero-frequency term.
    nonlinear_source = positive**2
    nonlinear_source -= np.mean(nonlinear_source)
    nonlinear = delay_and_filter(
        nonlinear_source, frequency, target_delay, settings.nonlinear_depth,
        settings, 2.0 * F0)
    nonlinear /= max(np.max(np.abs(nonlinear)), 1e-15)
    nonlinear *= settings.nonlinear_amplitude

    if noise_pair is None:
        rng = np.random.default_rng(args.seed)
        noise_pair = rng.normal(0.0, args.noise_fraction, (2, n))
    measured_positive = linear_positive + nonlinear + noise_pair[0]
    measured_negative = linear_negative + nonlinear + noise_pair[1]

    pulse_inversion = 0.5 * (measured_positive + measured_negative)
    linear_channel = 0.5 * (measured_positive - measured_negative)
    # This half-sum is the recovered nonlinear channel. No waveform template or
    # ground-truth subtraction is used; optional spectral filtering would be a
    # subsequent denoising operation rather than the separation mechanism.
    recovered = pulse_inversion.copy()
    return dict(settings=settings, time=time, positive=positive,
                negative=negative, linear=linear_positive,
                nonlinear=nonlinear, measured_positive=measured_positive,
                measured_negative=measured_negative,
                pulse_inversion=pulse_inversion,
                linear_channel=linear_channel, recovered=recovered,
                noise_pair=noise_pair)


def echo_window(result, margin_us=2.0):
    settings = result["settings"]
    start = settings.pulse_center + 2 * min(
        settings.reflector_depth, settings.nonlinear_depth) / SOUND_SPEED
    stop = settings.pulse_center + 2 * max(
        settings.reflector_depth, settings.nonlinear_depth) / SOUND_SPEED
    margin = margin_us * 1e-6
    return (result["time"] >= start - margin) & (result["time"] <= stop + margin)


def snr_db(signal, estimate):
    error = estimate - signal
    return 20 * np.log10(np.linalg.norm(signal) /
                         max(np.linalg.norm(error), 1e-15))


def plot_result(result):
    time_us = result["time"] * 1e6
    view = echo_window(result)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)

    axes[0].plot(time_us[view], result["measured_positive"][view],
                 color="#0072b2", lw=1, label="Measured after +x(t)")
    axes[0].plot(time_us[view], result["measured_negative"][view],
                 color="#d55e00", lw=1, label="Measured after −x(t)")
    axes[0].plot(time_us[view], result["nonlinear"][view], color="#009e73",
                 lw=2, label="Hidden nonlinear echo (validation truth)")
    axes[0].set(xlabel="Oscilloscope time (µs)", ylabel="Receiver amplitude",
                title="The nonlinear target is buried by the linear reflector")

    axes[1].plot(time_us[view], result["recovered"][view], color="#cc79a7",
                 lw=1.5, label="Recovered half-sum: (y+ + y−)/2")
    axes[1].plot(time_us[view], result["nonlinear"][view], color="black",
                 ls="--", lw=1.3, label="True nonlinear echo (validation only)")
    score = snr_db(result["nonlinear"][view], result["recovered"][view])
    axes[1].set(xlabel="Oscilloscope time (µs)", ylabel="Receiver amplitude",
                title=f"Pulse inversion: linear echo cancels; recovery SNR={score:.1f} dB")

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.suptitle("Recovering a nonlinear echo without a stored clutter template",
                 fontsize=15)
    return fig


def animate_depth_sweep(args, fps, save_path):
    if fps <= 0:
        raise ValueError("--animation-fps must be positive")
    center = args.reflector_depth_mm * 1e-3
    target_depths = np.linspace(center - 3e-3, center + 3e-3, 49)
    base = simulate(args, target_depths[0])
    noise_pair = base["noise_pair"]
    time_us = base["time"] * 1e6
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
    measured, = axes[0].plot([], [], color="#0072b2", lw=1,
                             label="Measured after +x(t)")
    measured_inverted, = axes[0].plot([], [], color="#d55e00", lw=1,
                                      label="Measured after −x(t)")
    hidden, = axes[0].plot([], [], color="#009e73", lw=2,
                           label="Hidden target (validation truth)")
    recovered, = axes[1].plot([], [], color="#cc79a7", lw=2,
                              label="Pulse-inversion recovery")
    truth, = axes[1].plot([], [], color="black", ls="--", lw=1.3,
                          label="True target (validation only)")
    for ax in axes:
        ax.set(xlim=(54, 69), xlabel="Oscilloscope time (µs)",
               ylabel="Receiver amplitude")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    axes[0].set_ylim(-1.15, 1.15)
    axes[1].set_ylim(-0.12, 0.12)

    def update(i):
        result = simulate(args, target_depths[i], noise_pair)
        measured.set_data(time_us, result["measured_positive"])
        measured_inverted.set_data(time_us, result["measured_negative"])
        hidden.set_data(time_us, result["nonlinear"])
        recovered.set_data(time_us, result["recovered"])
        truth.set_data(time_us, result["nonlinear"])
        separation = (target_depths[i] - center) * 1e3
        axes[0].set_title(
            f"Opposite-polarity acquisitions: target depth offset={separation:+.2f} mm")
        axes[1].set_title(
            "Half-sum: opposite linear echoes cancel; nonlinear echo remains")
        return measured, measured_inverted, hidden, recovered, truth

    animation = FuncAnimation(fig, update, frames=len(target_depths),
                              interval=1000 / fps, blit=False)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(save_path, writer="pillow", fps=fps, dpi=110)
    fig._animation = animation
    return fig


def export_csv(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "transmit_positive", "transmit_negative",
                         "measured_positive", "measured_negative",
                         "pulse_inversion_half_sum", "recovered_harmonic",
                         "validation_true_nonlinear"])
        writer.writerows(zip(
            result["time"], result["positive"], result["negative"],
            result["measured_positive"], result["measured_negative"],
            result["pulse_inversion"], result["recovered"],
            result["nonlinear"]))


def main():
    args = parse_args()
    result = simulate(args)
    view = echo_window(result)
    score = snr_db(result["nonlinear"][view], result["recovered"][view])
    print("Nonlinear pulse-inversion experiment")
    print(f"  linear reflector depth: {args.reflector_depth_mm:.3f} mm")
    print(f"  nonlinear target depth: {args.target_depth_mm:.3f} mm")
    print(f"  nonlinear/linear amplitude: {args.nonlinear_fraction:.4f}")
    print(f"  recovery SNR: {score:.2f} dB")
    print("  recovery uses the two measured traces, not a clutter template")
    if args.save_csv:
        export_csv(args.save_csv, result)
    if args.save_figure or not args.no_show:
        figure = plot_result(result)
        if args.save_figure:
            args.save_figure.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save_figure, dpi=180)
        if not args.no_show:
            plt.show()
        else:
            plt.close(figure)
    if args.save_animation:
        figure = animate_depth_sweep(
            args, args.animation_fps, args.save_animation)
        plt.close(figure)


if __name__ == "__main__":
    main()
