# PET-SPECT / Imaging Methods

This repository is a growing collection of code, simulations, documentation,
and results developed while studying medical and scientific imaging methods.
Although the repository is named **PET-SPECT** for its broader intended scope,
its current contents focus on one-dimensional ultrasound propagation and
measurement.

## Current contents

### Assignment 1 — 1D ultrasound simulations

`Assignment 1` contains three baseline Python simulations that build
progressively from a basic wave-equation demonstration to a density-aware,
layered pulse-echo model, plus a nonlinear imaging extension:

- **`1DWaveSimulation.py`** — introductory 1D FDTD wave propagation across a
  single change in sound speed, with static snapshots and animation.
- **`1DWaveSimulationDensity.py`** — staggered-grid pressure/particle-velocity
  acoustics with water and dermis, density and impedance discontinuities,
  pulse-echo and through-transmission measurements, CSV output, and estimated
  material properties.
- **`1DWaveSimulationLayered.py`** — the finalized intermediate model. It uses
  a water–dermis–water–PMMA geometry, a finite PMMA plate, a rigid idealized
  2 MHz piezo face, multiple reflections, labeled A–F echoes, an ideal receive
  voltage, static figures, CSV export, and an interactive GUI with precomputed
  transducer standoff distances.
- **`NonlinearPulseInversion.py`** — a two-acquisition pulse-inversion study. It
  demonstrates how a weak even-order nonlinear target can be recovered while
  its echo overlaps a much stronger linear reflector, without subtracting a
  known reflector template.

The assignment also includes:

- Reproducible Python dependencies managed by Astral `uv`.
- Canonical figures and waveform CSV files under `Assignment 1/outputs`.
- A development transcript under `Assignment 1/outputs/documents`.
- A detailed technical README explaining the acoustic equations, staggered-grid
  FDTD method, material parameters, echo paths, assumptions, scholarly sources,
  and exact run commands.

See [Assignment 1/README.md](Assignment%201/README.md) for the complete usage and
physics documentation.

## Repository layout

```text
PET-SPECT/
|-- README.md
|-- Assignment 1.zip
`-- Assignment 1/
    |-- README.md
    |-- scripts/
    |   |-- 1DWaveSimulation.py
    |   |-- 1DWaveSimulationDensity.py
    |   |-- 1DWaveSimulationLayered.py
    |   |-- NonlinearPulseInversion.py
    |   |-- pyproject.toml
    |   `-- uv.lock
    `-- outputs/
        |-- data/
        |-- documents/
        `-- figures/
```

`Assignment 1.zip` is an archived snapshot. The unpacked `Assignment 1`
directory is the working, documented version.

## Quick start

Install [Astral uv](https://docs.astral.sh/uv/), then run:

```powershell
cd ".\Assignment 1\scripts"
uv sync --locked
uv run python .\1DWaveSimulationLayered.py
```

Launch the interactive layered simulation with:

```powershell
uv run python .\1DWaveSimulationLayered.py --animate
```

Run the nonlinear pulse-inversion extension with:

```powershell
uv run python .\NonlinearPulseInversion.py
```

## Scope and status

The current ultrasound programs are educational, idealized one-dimensional
models—not validated clinical or instrument-design software. The
pulse-inversion extension represents a generic strong linear reflector rather
than full elastic propagation through bone. Future PET, SPECT, and other
imaging work can be added as separate top-level assignments or project
directories while retaining this structure and its reproducible environments.
