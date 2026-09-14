"""1D pulse-echo ultrasound in a water-skin-water-PMMA stack.

This educational model solves the lossless, linear acoustic equations with a
staggered-grid FDTD scheme.  Every impedance discontinuity therefore creates
both a transmitted wave and a signed reflection.  Later arrivals include the
smaller multiple-reflection (reverberation) artifacts that bounce among the
piezo face, skin interfaces, and finite PMMA slab.

The geometry is, from left to right:

    transducer -> water (a) -> dermis (b < a) -> water (c) -> PMMA slab (d)

The simulation is one-dimensional and normal-incidence: "refraction" means
transmission into a new material, not angular bending.  It neglects frequency-
dependent attenuation, shear waves, beam spreading, transducer ringing, and
surface roughness.

Material values and sources
---------------------------
Water (25 deg C): rho=997.047 kg/m^3 and c=1496.699 m/s, from the IAPWS-95
formulation (Wagner & Pruss, J. Phys. Chem. Ref. Data 31, 2002).
Dermis: rho=1151 kg/m^3 and c=1595 m/s, Table VIII of Maneas et al., 2016,
Phys. Med. Biol. 61, 5126-5149, doi:10.1088/0031-9155/61/13/5126.
PMMA: rho=1175 kg/m^3 and c=2750 m/s, Table 1 of Chen et al., 2023,
Photoacoustics 34, 100566, doi:10.1016/j.pacs.2023.100566.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider


DX_M = 25.0e-6
CFL = 0.90
CENTER_FREQUENCY_HZ = 2.0e6
PULSE_CYCLES = 3
DEFAULT_SOURCE_PRESSURE_MPA = 1.0
# Idealized receive chain.  Sensitivity fixes the otherwise arbitrary mapping
# from acoustic pressure to open-circuit voltage; the Gaussian response models
# a clean, phase-linear piezo centered at 2 MHz rather than a specific product.
PIEZO_SENSITIVITY_V_PER_PA = 1.0e-6  # 1 V/MPa
PIEZO_FRACTIONAL_BANDWIDTH = 0.80    # -6 dB amplitude bandwidth / center freq
GUI_DISTANCE_COUNT = 11              # includes both endpoints and the center

ECHO_STYLES = {
    "A: skin front": ("#e69f00", "-"),
    "B: skin back": ("#d55e00", "--"),
    "C: transducer-skin reverberation": ("#cc3311", ":"),
    "D: PMMA front face": ("#7c3aed", "-"),
    "E: PMMA rear face": ("#332288", "--"),
    "F: PMMA reverberation": ("#aa4499", ":"),
    "additional water-c reverberation": ("#0099bb", ":"),
}
LEFT_PADDING_M = 4.0e-3
RIGHT_PADDING_M = 2.0e-3
SPONGE_THICKNESS_M = 1.5e-3

WATER_DENSITY = 997.047
WATER_SOUND_SPEED = 1496.699
SKIN_DENSITY = 1151.0
SKIN_SOUND_SPEED = 1595.0
PMMA_DENSITY = 1175.0
PMMA_SOUND_SPEED = 2750.0


@dataclass(frozen=True)
class Geometry:
    water_a_m: float
    skin_b_m: float
    water_c_m: float
    pmma_d_m: float
    source_x_m: float = LEFT_PADDING_M

    @property
    def skin_start_m(self):
        return self.source_x_m + self.water_a_m

    @property
    def skin_end_m(self):
        return self.skin_start_m + self.skin_b_m

    @property
    def pmma_start_m(self):
        return self.skin_end_m + self.water_c_m

    @property
    def pmma_end_m(self):
        return self.pmma_start_m + self.pmma_d_m

    @property
    def domain_length_m(self):
        return self.pmma_end_m + RIGHT_PADDING_M


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pulse-echo FDTD model of water-skin-water-PMMA layers"
    )
    parser.add_argument("--seed", type=int, default=20260914,
                        help="seed for reproducible random layer thicknesses")
    parser.add_argument("--a-mm", type=float,
                        help="water standoff (static: random 8-14 mm; GUI: 49.3 mm)")
    parser.add_argument("--b-mm", type=float,
                        help="skin thickness (default: random 1.5-4 mm, always < a)")
    parser.add_argument("--c-mm", type=float,
                        help="second water thickness (default: random 4-10 mm)")
    parser.add_argument("--d-mm", type=float,
                        help="PMMA thickness (default: random 3-7 mm)")
    parser.add_argument("--animate", action="store_true",
                        help="animate pressure as the pulse traverses the layers")
    parser.add_argument("--interval", type=int, default=25, metavar="MS",
                        help="animation frame delay in milliseconds (default: 25)")
    parser.add_argument(
        "--source-pressure-mpa", type=float, default=DEFAULT_SOURCE_PRESSURE_MPA,
        metavar="MPA", help="peak source pressure in MPa (default: 1.0)",
    )
    parser.add_argument("--noise", type=float, default=0.0, metavar="FRACTION",
                        help="receiver Gaussian noise relative to RF peak")
    parser.add_argument("--save-csv", type=Path, metavar="PATH")
    parser.add_argument("--save-figure", type=Path, metavar="PATH")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def choose_geometry(args):
    rng = np.random.default_rng(args.seed)
    if args.a_mm is not None:
        a_mm = args.a_mm
    elif args.animate:
        a_mm = 49.3
    else:
        a_mm = rng.uniform(8.0, 14.0)
    b_upper_mm = min(4.0, 0.8 * a_mm)
    b_mm = args.b_mm if args.b_mm is not None else rng.uniform(1.5, b_upper_mm)
    c_mm = args.c_mm if args.c_mm is not None else rng.uniform(4.0, 10.0)
    d_mm = args.d_mm if args.d_mm is not None else rng.uniform(3.0, 7.0)
    if min(a_mm, b_mm, c_mm, d_mm) <= 0:
        raise ValueError("all layer thicknesses must be positive")
    if b_mm >= a_mm:
        raise ValueError("skin thickness b must be smaller than water thickness a")
    return Geometry(a_mm * 1e-3, b_mm * 1e-3, c_mm * 1e-3, d_mm * 1e-3)


def material_arrays(geometry):
    nx = int(np.ceil(geometry.domain_length_m / DX_M))
    x = (np.arange(nx) + 0.5) * DX_M
    material = np.full(nx, "water", dtype="U5")
    material[(x >= geometry.skin_start_m) & (x < geometry.skin_end_m)] = "skin"
    pmma = (x >= geometry.pmma_start_m) & (x < geometry.pmma_end_m)
    material[pmma] = "pmma"
    density = np.select(
        [material == "skin", material == "pmma"],
        [SKIN_DENSITY, PMMA_DENSITY], default=WATER_DENSITY,
    )
    speed = np.select(
        [material == "skin", material == "pmma"],
        [SKIN_SOUND_SPEED, PMMA_SOUND_SPEED], default=WATER_SOUND_SPEED,
    )
    bulk_modulus = density * speed**2
    density_face = 2.0 * density[:-1] * density[1:] / (density[:-1] + density[1:])
    return x, material, density, speed, bulk_modulus, density_face


def source_waveform(time_s, amplitude_pa):
    duration = PULSE_CYCLES / CENTER_FREQUENCY_HZ
    center = 2.5 * duration
    sigma = duration / 3.0
    return (amplitude_pa
            * np.sin(2.0 * np.pi * CENTER_FREQUENCY_HZ * (time_s - center))
            * np.exp(-0.5 * ((time_s - center) / sigma) ** 2))


def build_sponge(nx):
    cells = max(8, round(SPONGE_THICKNESS_M / DX_M))
    damping_p = np.ones(nx)
    ramp = np.linspace(1.0, 0.0, cells, endpoint=False)
    edge = np.exp(-0.12 * ramp**2)
    damping_p[:cells] *= edge
    damping_p[-cells:] *= edge[::-1]
    return damping_p, np.sqrt(damping_p[:-1] * damping_p[1:])


def analytic_envelope(signal):
    spectrum = np.fft.fft(signal)
    weights = np.zeros(signal.size)
    weights[0] = 1.0
    if signal.size % 2 == 0:
        weights[1:signal.size // 2] = 2.0
        weights[signal.size // 2] = 1.0
    else:
        weights[1:(signal.size + 1) // 2] = 2.0
    return np.abs(np.fft.ifft(spectrum * weights))


def ideal_piezo_voltage(pressure_pa, dt):
    """Convert pressure to voltage through an ideal 2 MHz receive response.

    The symmetric real Gaussian frequency response introduces no phase delay.
    Its amplitude is unity at 2 MHz and one half at the stated -6 dB bandwidth
    edges.  The result is open-circuit voltage; cables, matching networks,
    amplifiers, electrical noise, and transmit feedthrough are intentionally
    omitted.
    """
    frequencies = np.fft.fftfreq(pressure_pa.size, dt)
    half_width_hz = 0.5 * PIEZO_FRACTIONAL_BANDWIDTH * CENTER_FREQUENCY_HZ
    sigma_hz = half_width_hz / np.sqrt(2.0 * np.log(2.0))
    response = np.exp(
        -0.5 * ((np.abs(frequencies) - CENTER_FREQUENCY_HZ) / sigma_hz) ** 2
    )
    filtered_pressure = np.fft.ifft(np.fft.fft(pressure_pa) * response).real
    return PIEZO_SENSITIVITY_V_PER_PA * filtered_pressure


def simulation_duration_s(geometry):
    one_way_s = (geometry.water_a_m / WATER_SOUND_SPEED
                 + geometry.skin_b_m / SKIN_SOUND_SPEED
                 + geometry.water_c_m / WATER_SOUND_SPEED
                 + geometry.pmma_d_m / PMMA_SOUND_SPEED)
    return 2.5 * PULSE_CYCLES / CENTER_FREQUENCY_HZ + 4.2 * one_way_s


def simulate(geometry, noise_fraction=0.0, seed=0, capture_animation=False,
             source_pressure_mpa=DEFAULT_SOURCE_PRESSURE_MPA,
             duration_s=None):
    x, material, density, speed, bulk_modulus, density_face = material_arrays(geometry)
    dt = CFL * DX_M / np.max(speed)
    # Covers the principal back-plane echo and several later reverberations.
    if duration_s is None:
        duration_s = simulation_duration_s(geometry)
    nt = int(np.ceil(duration_s / dt))
    time_s = np.arange(nt) * dt
    if source_pressure_mpa <= 0:
        raise ValueError("--source-pressure-mpa must be positive")
    source = source_waveform(time_s, source_pressure_mpa * 1e6)
    damping_p, damping_v = build_sponge(x.size)
    pressure = np.zeros(x.size)
    velocity = np.zeros(x.size - 1)
    source_i = int(np.clip(geometry.source_x_m / DX_M, 0, x.size - 1))
    receiver_i = source_i + 2  # close to the piezo, but not the source cell
    trace = np.empty(nt)
    frame_stride = max(1, int(np.ceil(nt / 360)))
    pressure_frames = []
    frame_times = []

    for step in range(nt):
        velocity -= (dt / (density_face * DX_M)) * np.diff(pressure)
        # The ideal piezo face is a rigid acoustic termination (normal particle
        # velocity is zero).  This launches into the water on the right and
        # reflects returning echoes with pressure coefficient +1, allowing the
        # reflected echo to make another trip toward the layered sample.
        velocity[source_i - 1] = 0.0
        padded_velocity = np.pad(velocity, (1, 1), mode="edge")
        pressure -= (bulk_modulus * dt / DX_M) * np.diff(padded_velocity)
        pressure *= damping_p
        velocity *= damping_v
        pressure[source_i] += source[step]
        trace[step] = pressure[receiver_i]
        if capture_animation and step % frame_stride == 0:
            pressure_frames.append(pressure.copy())
            frame_times.append(time_s[step])

    if noise_fraction < 0:
        raise ValueError("--noise must be non-negative")
    if noise_fraction:
        rng = np.random.default_rng(seed + 1)
        trace += rng.normal(0.0, noise_fraction * np.max(np.abs(trace)), nt)
    voltage = ideal_piezo_voltage(trace, dt)
    return {"x": x, "material": material, "density": density, "speed": speed,
            "time": time_s, "source": source, "trace": trace,
            "voltage": voltage, "dt": dt,
            "pressure_frames": np.asarray(pressure_frames),
            "frame_times": np.asarray(frame_times)}


def interface_coefficients():
    z_water = WATER_DENSITY * WATER_SOUND_SPEED
    z_skin = SKIN_DENSITY * SKIN_SOUND_SPEED
    z_pmma = PMMA_DENSITY * PMMA_SOUND_SPEED
    reflection = lambda z1, z2: (z2 - z1) / (z2 + z1)
    return {
        "water_to_skin": reflection(z_water, z_skin),
        "skin_to_water": reflection(z_skin, z_water),
        "water_to_pmma": reflection(z_water, z_pmma),
        "pmma_to_water": reflection(z_pmma, z_water),
    }


def predicted_arrivals(geometry):
    burst_center = 2.5 * PULSE_CYCLES / CENTER_FREQUENCY_HZ
    to_skin = geometry.water_a_m / WATER_SOUND_SPEED
    through_skin = geometry.skin_b_m / SKIN_SOUND_SPEED
    through_c = geometry.water_c_m / WATER_SOUND_SPEED
    return {
        "A: skin front": burst_center + 2.0 * to_skin,
        "B: skin back": burst_center + 2.0 * (to_skin + through_skin),
        # C is a second round trip across water layer a: the A echo returns to
        # the rigid piezo, reflects toward the skin front, and reflects again.
        "C: transducer-skin reverberation": burst_center + 4.0 * to_skin,
        "D: PMMA front face": burst_center + 2.0 * (
            to_skin + through_skin + through_c),
        "E: PMMA rear face": burst_center + 2.0 * (
            to_skin + through_skin + through_c
            + geometry.pmma_d_m / PMMA_SOUND_SPEED),
        # F: after reaching the PMMA front, the pulse completes two round trips
        # through the finite plate before transmitting back toward the receiver.
        "F: PMMA reverberation": burst_center + 2.0 * (
            to_skin + through_skin + through_c
            + 2.0 * geometry.pmma_d_m / PMMA_SOUND_SPEED),
        "additional water-c reverberation": burst_center + 2.0 * (
            to_skin + through_skin + 2.0 * through_c),
    }


def highlight_echoes(ax, time_s, waveform, geometry):
    """Color and label the measured RF pulse nearest each A-F prediction."""
    envelope = analytic_envelope(waveform)
    pulse_duration_s = PULSE_CYCLES / CENTER_FREQUENCY_HZ
    time_us = time_s * 1e6
    handles = []
    labeled_arrivals = [
        (label, arrival_s)
        for label, arrival_s in predicted_arrivals(geometry).items()
        if label[0] in "ABCDEF"
    ]
    for arrival_index, (label, predicted_s) in enumerate(labeled_arrivals):
        neighbor_gaps = []
        if arrival_index:
            neighbor_gaps.append(
                predicted_s - labeled_arrivals[arrival_index - 1][1]
            )
        if arrival_index + 1 < len(labeled_arrivals):
            neighbor_gaps.append(
                labeled_arrivals[arrival_index + 1][1] - predicted_s
            )
        nearest_gap_s = min(neighbor_gaps) if neighbor_gaps else np.inf
        # Never let a stronger adjacent echo enter this echo's peak-search
        # gate. This is important for thin layers, whose RF bursts can overlap.
        search_half_width_s = min(1.25 * pulse_duration_s,
                                  0.40 * nearest_gap_s)
        display_half_width_s = min(0.65 * pulse_duration_s,
                                   0.45 * nearest_gap_s)
        search = np.flatnonzero(
            (time_s >= predicted_s - search_half_width_s)
            & (time_s <= predicted_s + search_half_width_s)
        )
        if not search.size:
            continue
        center_i = search[np.argmax(envelope[search])]
        center_s = time_s[center_i]
        segment = ((time_s >= center_s - display_half_width_s)
                   & (time_s <= center_s + display_half_width_s))
        color, _ = ECHO_STYLES[label]
        ax.plot(time_us[segment], waveform[segment], color=color, lw=2.2,
                zorder=4)
        rf_segment_indices = np.flatnonzero(segment)
        peak_i = rf_segment_indices[
            np.argmax(np.abs(waveform[rf_segment_indices]))
        ]
        ax.scatter(time_us[peak_i], waveform[peak_i], color=color, s=22,
                   edgecolor="white", linewidth=0.5, zorder=5)
        ax.annotate(label[0], (time_us[peak_i], waveform[peak_i]),
                    xytext=(0, 8 if waveform[peak_i] >= 0 else -13),
                    textcoords="offset points", ha="center", color=color,
                    weight="bold", fontsize=9)
        handles.append(Line2D([0], [0], color=color, lw=2.5,
                              label=label))
    return handles


def plot_results(geometry, result):
    x_mm = result["x"] * 1e3
    time_us = result["time"] * 1e6
    voltage_v = result["voltage"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True)

    ax = axes[0]
    layer_spans = (
        (0.0, geometry.skin_start_m, "#d9effb", "Water a"),
        (geometry.skin_start_m, geometry.skin_end_m, "#f7d8a7", "Skin b"),
        (geometry.skin_end_m, geometry.pmma_start_m, "#d9effb", "Water c"),
        (geometry.pmma_start_m, geometry.pmma_end_m, "#d8cdef", "PMMA d"),
        (geometry.pmma_end_m, geometry.domain_length_m, "#d9effb", "Water"),
    )
    for start_m, end_m, color, label in layer_spans:
        ax.axvspan(start_m * 1e3, end_m * 1e3, color=color, alpha=0.55,
                   label=label)
    density_line, = ax.plot(
        x_mm, result["density"], color="tab:blue", lw=2.2,
        label="Density (kg/m³)"
    )
    speed_ax = ax.twinx()
    speed_line, = speed_ax.plot(
        x_mm, result["speed"], color="tab:orange", lw=2.2, linestyle="--",
        label="Sound speed (m/s)"
    )
    for x_value, label in ((geometry.skin_start_m, "skin front"),
                           (geometry.skin_end_m, "skin back"),
                           (geometry.pmma_start_m, "PMMA front"),
                           (geometry.pmma_end_m, "PMMA back")):
        ax.axvline(x_value * 1e3, color="0.25", linestyle="--", alpha=0.7)
        ax.text(x_value * 1e3, ax.get_ylim()[1], label, rotation=90, va="top", ha="right")
    ax.set(xlabel="Position (mm)", ylabel="Density (kg/m³)", title="Layered geometry")
    speed_ax.set_ylabel("Sound speed (m/s)")
    # Independent, deliberately asymmetric ranges prevent the two PMMA plateaus
    # (which are both the maxima of their respective data) from overlapping.
    density_margin = 0.20 * np.ptp(result["density"])
    speed_margin = 0.55 * np.ptp(result["speed"])
    ax.set_ylim(result["density"].min() - density_margin,
                result["density"].max() + density_margin)
    speed_ax.set_ylim(result["speed"].min() - 0.10 * speed_margin,
                      result["speed"].max() + speed_margin)
    material_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                      edgecolor="none", alpha=0.55, label=label)
                        for _, _, color, label in layer_spans[:4]]
    ax.legend(handles=[density_line, speed_line, *material_handles],
              loc="upper left", ncol=3, fontsize=8)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ax.plot(time_us, voltage_v, color="0.62", lw=0.9,
            label="Unclassified waveform")
    echo_handles = highlight_echoes(ax, result["time"], voltage_v, geometry)
    ax.set(xlabel="Time (µs)", ylabel="Piezo voltage (V)",
           title="Ideal 2 MHz piezoelectric receive waveform")
    ax.grid(alpha=0.25)
    ax.legend(handles=echo_handles, ncol=3, fontsize=8, loc="upper right")
    fig.suptitle("1D water–skin–water–PMMA ultrasound simulation", fontsize=15)
    return fig


def animate_wave(geometry, result, interval_ms, precomputed):
    """Animate pressure while showing a static, instantly selectable A-scan."""
    if interval_ms <= 0:
        raise ValueError("--interval must be positive")
    if not result["pressure_frames"].size:
        raise ValueError("animation frames were not captured")
    state = {"geometry": geometry, "result": result, "frame": 0,
             "scope_end_us": result["time"][-1] * 1e6}
    fig = plt.figure(figsize=(15, 7))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.7, 1.0),
                           left=0.06, right=0.98, top=0.92, bottom=0.19,
                           wspace=0.25)
    ax = fig.add_subplot(grid[0, 0])
    scope_ax = fig.add_subplot(grid[0, 1])
    slider_ax = fig.add_axes((0.13, 0.075, 0.72, 0.035))
    status = fig.text(0.13, 0.025, "Select a precomputed transducer distance",
                      fontsize=9, color="0.3")
    allowed_a_mm = np.array(sorted(precomputed))
    a_slider = Slider(
        slider_ax, "Water a (mm)", allowed_a_mm[0], allowed_a_mm[-1],
        valinit=geometry.water_a_m * 1e3, valstep=allowed_a_mm,
    )
    artists = {}

    def draw_panels():
        current_geometry = state["geometry"]
        current = state["result"]
        frames = current["pressure_frames"]
        x_mm = current["x"] * 1e3
        limit = 1.1 * max(np.max(np.abs(frames)), 1e-12)
        ax.clear()
        ax.axvspan(current_geometry.source_x_m * 1e3,
                   current_geometry.skin_start_m * 1e3,
                   color="#9ecae1", alpha=0.22, label="Water a")
        ax.axvspan(current_geometry.skin_start_m * 1e3,
                   current_geometry.skin_end_m * 1e3,
                   color="#f4c27a", alpha=0.32, label="Skin b")
        ax.axvspan(current_geometry.skin_end_m * 1e3,
                   current_geometry.pmma_start_m * 1e3,
                   color="#9ecae1", alpha=0.22, label="Water c")
        ax.axvspan(current_geometry.pmma_start_m * 1e3,
                   current_geometry.pmma_end_m * 1e3,
                   color="#b5a7d8", alpha=0.35, label="PMMA d")
        ax.axvline(current_geometry.source_x_m * 1e3, color="black",
                   linestyle=":", label="Rigid piezo face")
        receiver_x_mm = (current_geometry.source_x_m + 2.0 * DX_M) * 1e3
        ax.axvline(receiver_x_mm, color="tab:red", linestyle="--",
                   label="Receiver")
        artists["wave"], = ax.plot(x_mm, frames[0], color="tab:blue", lw=1.5,
                                    label="Acoustic pressure")
        ax.set(xlim=(current_geometry.source_x_m * 1e3 - 0.5, x_mm[-1]),
               ylim=(-limit, limit), xlabel="Position (mm)",
               ylabel="Pressure (Pa)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=3, fontsize=8)

        full_time_us = current["time"] * 1e6
        voltage_v = current["voltage"]
        scope_limit = 1.08 * max(np.max(np.abs(voltage_v)), 1e-12)
        scope_ax.clear()
        artists["scope"], = scope_ax.plot(
            full_time_us, voltage_v, color="0.62", lw=1.0,
            label="Complete measured voltage",
        )
        echo_handles = highlight_echoes(
            scope_ax, current["time"], voltage_v, current_geometry
        )
        scope_ax.set(xlim=(0.0, state["scope_end_us"]),
                     ylim=(-scope_limit, scope_limit), xlabel="Time (µs)",
                     ylabel="Piezo voltage (V)",
                     title="Live ideal 2 MHz piezo oscilloscope")
        scope_ax.grid(alpha=0.25)
        scope_ax.legend(handles=echo_handles, loc="upper right", ncol=2,
                        fontsize=8)

    draw_panels()

    def update(_frame_number):
        current = state["result"]
        frames = current["pressure_frames"]
        frame_index = state["frame"] % len(frames)
        artists["wave"].set_ydata(frames[frame_index])
        ax.set_title(
            f"Wave propagation through water–skin–water–PMMA: "
            f"t = {current['frame_times'][frame_index] * 1e6:.2f} µs"
        )
        state["frame"] = frame_index + 1
        return (artists["wave"],)

    def select_distance(new_a_mm):
        selected_a_mm = float(allowed_a_mm[np.argmin(abs(allowed_a_mm - new_a_mm))])
        new_geometry, new_result = precomputed[selected_a_mm]
        state["geometry"] = new_geometry
        state["result"] = new_result
        state["frame"] = 0
        draw_panels()
        c_time = predicted_arrivals(new_geometry)[
            "C: transducer-skin reverberation"
        ] * 1e6
        status.set_text(
            f"Precomputed FDTD: a = {selected_a_mm:.2f} mm; "
            f"C center = {c_time:.2f} µs"
        )
        fig.canvas.draw_idle()

    a_slider.on_changed(select_distance)
    animation = FuncAnimation(fig, update, interval=interval_ms,
                              cache_frame_data=False, blit=False)
    fig._animation = animation
    fig._a_slider = a_slider
    return fig


def export_csv(path, geometry, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    envelope = analytic_envelope(result["voltage"])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([f"a_m={geometry.water_a_m}", f"b_m={geometry.skin_b_m}",
                         f"c_m={geometry.water_c_m}", f"d_m={geometry.pmma_d_m}"])
        writer.writerow(["time_s", "source_pressure_pa", "receiver_pressure_pa",
                         "piezo_voltage_v", "piezo_voltage_envelope_v"])
        writer.writerows(zip(result["time"], result["source"], result["trace"],
                             result["voltage"], envelope))


def print_summary(geometry):
    coeffs = interface_coefficients()
    print("Layered 1D ultrasound simulation")
    print(f"  water a: {geometry.water_a_m * 1e3:.3f} mm")
    print(f"  skin  b: {geometry.skin_b_m * 1e3:.3f} mm")
    print(f"  water c: {geometry.water_c_m * 1e3:.3f} mm")
    print(f"  PMMA  d: {geometry.pmma_d_m * 1e3:.3f} mm")
    print("  normal-incidence pressure reflection coefficients:")
    for name, value in coeffs.items():
        print(f"    {name.replace('_', ' '):20s} R = {value:+.5f}, intensity = {value**2:.4%}")
    print("  predicted echo centers:")
    for name, arrival_s in predicted_arrivals(geometry).items():
        print(f"    {name:34s} {arrival_s * 1e6:9.3f} us")


def main():
    args = parse_args()
    geometry = choose_geometry(args)
    fixed_animation_duration_s = None
    precomputed = None
    if args.animate:
        center_a_mm = geometry.water_a_m * 1e3
        minimum_a_mm = max(center_a_mm - 25.0,
                           1.02 * geometry.skin_b_m * 1e3, 1.0)
        maximum_a_mm = center_a_mm + 25.0
        allowed_a_mm = np.linspace(minimum_a_mm, maximum_a_mm,
                                   GUI_DISTANCE_COUNT)
        starting_a_mm = float(allowed_a_mm[np.argmin(abs(allowed_a_mm - center_a_mm))])
        geometry = Geometry(starting_a_mm * 1e-3, geometry.skin_b_m,
                            geometry.water_c_m, geometry.pmma_d_m,
                            geometry.source_x_m)
        longest_geometry = Geometry(maximum_a_mm * 1e-3, geometry.skin_b_m,
                                    geometry.water_c_m, geometry.pmma_d_m,
                                    geometry.source_x_m)
        fixed_animation_duration_s = simulation_duration_s(longest_geometry)
        precomputed = {}
        print(f"Precomputing {GUI_DISTANCE_COUNT} GUI distances...")
        for index, a_mm in enumerate(allowed_a_mm, start=1):
            candidate = Geometry(float(a_mm) * 1e-3, geometry.skin_b_m,
                                 geometry.water_c_m, geometry.pmma_d_m,
                                 geometry.source_x_m)
            print(f"  [{index:2d}/{GUI_DISTANCE_COUNT}] a = {a_mm:.2f} mm")
            precomputed[float(a_mm)] = (
                candidate,
                simulate(candidate, args.noise, args.seed, True,
                         args.source_pressure_mpa, fixed_animation_duration_s),
            )
        geometry, result = precomputed[starting_a_mm]
    else:
        result = simulate(geometry, args.noise, args.seed, False,
                          args.source_pressure_mpa)
    print_summary(geometry)
    if args.save_csv:
        export_csv(args.save_csv, geometry, result)
    if args.save_figure or (not args.no_show and not args.animate):
        figure = plot_results(geometry, result)
        if args.save_figure:
            args.save_figure.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(args.save_figure, dpi=180)
        if not args.no_show:
            plt.show()
        else:
            plt.close(figure)
    if args.animate and not args.no_show:
        animate_wave(geometry, result, args.interval, precomputed)
        plt.show()


if __name__ == "__main__":
    main()
