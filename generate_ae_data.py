"""
generate_ae_data.py — Synthetic AE Arrival-Time Data Generator (Geometric Wave Model)
=======================================================================================
Reproduce Section 2.1 of "Wave-Equation Constrained PINN for Acoustic Emission
Source Localization in Bolted Connections: A Cyber-Physical Digital Twin
Framework with ifcJSON Middleware" — generates synthetic AE arrival times using
the analytic geometric model t_i = dist(source, S_i) / c + ε.

Standalone script: no dependencies on openseespy, belico-stack, config/params.yaml,
or any factory infrastructure. All physical parameters are hardcoded below with
annotated bibliographic sources so the reader can verify every number independently.

Physics model (geometric, Section 2.1)
---------------------------------------
- Steel plate    : 300 mm × 300 mm square [m]
  Source: plate geometry from experimental setup, Secs. 2.1 & 3.
- Bolt position  : (0.15, 0.15) m — geometric centre of the plate
- Sensors (6)    : perimeter positions [m]
    S1 = (0.00, 0.00)  — corner bottom-left
    S2 = (0.15, 0.00)  — edge bottom-mid
    S3 = (0.30, 0.00)  — corner bottom-right
    S4 = (0.30, 0.30)  — corner top-right
    S5 = (0.15, 0.30)  — edge top-mid
    S6 = (0.00, 0.30)  — corner top-left
- Wave speed     : c = 5000 m/s — Lamb S0 mode in 6 mm steel at ~50 kHz.
  Source: Rose (2014) "Ultrasonic Guided Waves in Solid Media",
          Cambridge University Press, p. 107, Table 5.1.
- Arrival time   : t_i = ||source - S_i|| / c + ε_i
                   ε_i ~ N(0, σ²),  σ = σ_base × noise_multiplier
  Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing",
          Springer, §3.4, p. 87.
- σ_base         : 1 μs  — timing uncertainty for a 1 MHz digitizer with
                   40 dB preamplification.
  Source: Grosse & Ohtsu (2008) §3.4.

Source position sampling (polar uniform-area, avoids centre bias)
-----------------------------------------------------------------
  r = sqrt(U) * r_max,  U ~ Uniform[0, 1)
  θ ~ Uniform[0, 2π)
  x = bolt_x + r cos θ,  y = bolt_y + r sin θ
  (clipped to [0.02, 0.28] m to stay away from plate boundary)

Uniform plate sampling uses Uniform[0.02, 0.28]² independently for x and y.

4 Scenarios (100 samples each, default seed=42)
------------------------------------------------
  intact     (  0% torque loss): 5%  near bolt, r < 0.08 m, σ = σ_base × 1.0
  loose_25   ( 25% torque loss): 70% near bolt, r < 0.08 m, σ = σ_base × 1.5
  loose_50   ( 50% torque loss): 85% near bolt, r < 0.05 m, σ = σ_base × 2.0
  full_loose (100% torque loss): 95% near bolt, r < 0.03 m, σ = σ_base × 3.0

Output
------
  data/processed/ae_synthetic_arrivals.csv  (relative to script directory)
  Columns : scenario, torque_loss_pct, source_x, source_y,
            t1, t2, t3, t4, t5, t6   (times in microseconds [μs])
  Rows    : 400 total (100 per scenario, configurable via --n-per-scenario)

Usage
-----
  python generate_ae_data.py
  python generate_ae_data.py --n-per-scenario 200 --seed 7
  python generate_ae_data.py --output-dir /tmp/ae_out
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants — all values annotated with bibliographic sources
# ---------------------------------------------------------------------------

# Lamb S0 wave speed in 6 mm steel plate at ~50 kHz
# Source: Rose (2014) "Ultrasonic Guided Waves in Solid Media",
#         Cambridge University Press, p. 107, Table 5.1.
WAVE_SPEED_MS = 5000.0   # [m/s]

# Plate geometry [m]
PLATE_MIN = 0.00
PLATE_MAX = 0.30   # 300 mm × 300 mm

# Boundary margin — sources kept ≥ 20 mm from plate edge [m]
BOUNDARY_MARGIN = 0.02

# Bolt position — geometric centre of the plate [m]
BOLT_X = 0.15
BOLT_Y = 0.15

# 6 piezoelectric sensors on plate perimeter [m]
# Layout matches experimental rig described in Sec. 2.1.
SENSORS = np.array([
    [0.00, 0.00],   # S1 — corner bottom-left
    [0.15, 0.00],   # S2 — edge bottom-mid
    [0.30, 0.00],   # S3 — corner bottom-right
    [0.30, 0.30],   # S4 — corner top-right
    [0.15, 0.30],   # S5 — edge top-mid
    [0.00, 0.30],   # S6 — corner top-left
], dtype=np.float64)

# Baseline timing noise standard deviation
# Corresponds to 1 MHz ADC digitizer with 40 dB preamplification.
# Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing", §3.4, p. 87.
SIGMA_BASE_S = 1.0e-6    # [s] = 1 μs

# Scenario definitions:
#   (scenario_name, torque_loss_pct, bolt_fraction, bolt_radius_m, noise_multiplier)
#
#   bolt_fraction    — probability that a source event is placed near the bolt
#   bolt_radius_m    — maximum radius of the near-bolt clustering zone [m]
#   noise_multiplier — scales SIGMA_BASE_S; larger = more timing scatter
#                      (models microcrack noise under loose bolt conditions)
SCENARIOS = [
    ("intact",       0,   0.05, 0.08, 1.0),
    ("loose_25",    25,   0.70, 0.08, 1.5),
    ("loose_50",    50,   0.85, 0.05, 2.0),
    ("full_loose", 100,   0.95, 0.03, 3.0),
]

# Default CLI parameters
N_PER_SCENARIO_DEFAULT = 100
RANDOM_SEED_DEFAULT     = 42


# ---------------------------------------------------------------------------
# Geometric arrival time model
# ---------------------------------------------------------------------------

def compute_arrivals_us(source: np.ndarray) -> np.ndarray:
    """Compute noiseless arrival times at all 6 sensors [μs].

    Uses the analytic formula from Sec. 2.1:
        t_i = ||source - S_i||_2 / c

    Parameters
    ----------
    source : (2,) array — (x, y) source position [m]

    Returns
    -------
    (6,) array of arrival times in microseconds [μs]
    """
    # Euclidean distances from source to each sensor [m]
    distances = np.linalg.norm(SENSORS - source, axis=1)   # shape (6,)
    # Convert to time [s] then to [μs]
    return (distances / WAVE_SPEED_MS) * 1.0e6


# ---------------------------------------------------------------------------
# Source position samplers
# ---------------------------------------------------------------------------

def _sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 2) source positions uniformly distributed inside plate with margin.

    Independent Uniform[BOUNDARY_MARGIN, PLATE_MAX - BOUNDARY_MARGIN] for
    each coordinate axis.
    """
    lo = PLATE_MIN + BOUNDARY_MARGIN
    hi = PLATE_MAX - BOUNDARY_MARGIN
    return rng.uniform(lo, hi, size=(n, 2))


def _sample_near_bolt(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """(n, 2) source positions uniformly distributed (area-preserving) within
    `radius` metres of the bolt centre.

    Polar sampling formula:
        r = sqrt(U) * radius,   U ~ Uniform[0, 1)
        θ ~ Uniform[0, 2π)
    This ensures uniform area density — naive r ~ U would pile up at centre.

    Coordinates are clipped to [BOUNDARY_MARGIN, PLATE_MAX - BOUNDARY_MARGIN]
    to remain within the usable plate area.
    """
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r     = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    x = np.clip(BOLT_X + r * np.cos(theta),
                PLATE_MIN + BOUNDARY_MARGIN,
                PLATE_MAX - BOUNDARY_MARGIN)
    y = np.clip(BOLT_Y + r * np.sin(theta),
                PLATE_MIN + BOUNDARY_MARGIN,
                PLATE_MAX - BOUNDARY_MARGIN)
    return np.column_stack([x, y])   # shape (n, 2)


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def generate_scenario(
    rng: np.random.Generator,
    n: int,
    scenario_name: str,
    torque_loss_pct: int,
    bolt_fraction: float,
    bolt_radius: float,
    noise_multiplier: float,
) -> pd.DataFrame:
    """Generate `n` geometric-model AE events for one torque-loss scenario.

    For each event:
      1. Sample a source position (near-bolt or uniform plate, by probability).
      2. Compute t_i = dist(source, S_i) / c  [analytic, no FEM].
      3. Add Gaussian noise: t_i += N(0, (σ_base * noise_multiplier)^2).

    Parameters
    ----------
    rng              : Seeded NumPy random generator (caller provides).
    n                : Number of AE events to generate.
    scenario_name    : Label written to 'scenario' column.
    torque_loss_pct  : Integer percentage torque loss (0 / 25 / 50 / 100).
    bolt_fraction    : Probability that an event is placed near the bolt.
    bolt_radius      : Near-bolt clustering radius [m].
    noise_multiplier : Scale applied to SIGMA_BASE_S.

    Returns
    -------
    pd.DataFrame with columns:
        scenario, torque_loss_pct, source_x, source_y,
        t1, t2, t3, t4, t5, t6  (times in microseconds)
    """
    sigma_us = (SIGMA_BASE_S * noise_multiplier) * 1.0e6   # convert to [μs]

    # Vectorised source positions for all n events
    # Determine which events are near-bolt
    is_near_bolt = rng.uniform(size=n) < bolt_fraction   # (n,) bool mask
    n_near  = int(is_near_bolt.sum())
    n_unif  = n - n_near

    sources = np.empty((n, 2), dtype=np.float64)
    if n_near > 0:
        sources[is_near_bolt]  = _sample_near_bolt(rng, n_near, bolt_radius)
    if n_unif > 0:
        sources[~is_near_bolt] = _sample_uniform(rng, n_unif)

    # Compute all arrival times vectorised: (n, 6) array [μs]
    # dist[i, j] = ||sources[i] - SENSORS[j]||
    # Broadcasting: sources[:,None,:] → (n,1,2), SENSORS[None,:,:] → (1,6,2)
    dists   = np.linalg.norm(sources[:, None, :] - SENSORS[None, :, :], axis=2)  # (n,6)
    arrivals = (dists / WAVE_SPEED_MS) * 1.0e6   # [μs]

    # Add Gaussian timing noise
    # Source: Grosse & Ohtsu (2008) §3.4
    noise    = rng.normal(0.0, sigma_us, size=(n, 6))   # [μs]
    arrivals = arrivals + noise                          # [μs]

    df = pd.DataFrame({
        "scenario":       scenario_name,
        "torque_loss_pct": torque_loss_pct,
        "source_x":       sources[:, 0],
        "source_y":       sources[:, 1],
        "t1": arrivals[:, 0],
        "t2": arrivals[:, 1],
        "t3": arrivals[:, 2],
        "t4": arrivals[:, 3],
        "t5": arrivals[:, 4],
        "t6": arrivals[:, 5],
    })
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic AE arrival-time dataset for bolted-joint SHM "
            "using the geometric wave model t = dist/c + noise (standalone)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-per-scenario",
        type=int,
        default=N_PER_SCENARIO_DEFAULT,
        metavar="INT",
        help="AE events per torque-loss scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED_DEFAULT,
        metavar="INT",
        help="NumPy random seed for reproducibility.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output directory (default: data/processed/ relative to this script).",
    )
    return parser


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    t0     = time.perf_counter()
    parser = _build_parser()
    args   = parser.parse_args()

    # Resolve output directory relative to this script (not CWD)
    script_dir = Path(__file__).resolve().parent
    if args.output_dir is not None:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute()
            else script_dir / args.output_dir
        )
    else:
        output_dir = script_dir / "data" / "processed"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[generate_ae_data] ERROR — cannot create output dir: {exc}",
              file=sys.stderr)
        return 1

    output_path = output_dir / "ae_synthetic_arrivals.csv"

    rng    = np.random.default_rng(args.seed)
    frames = []

    print(
        f"\n[generate_ae_data] Geometric model: t = dist/c + N(0,σ²)\n"
        f"  Wave speed  : {WAVE_SPEED_MS:.0f} m/s  "
        f"(Lamb S0, Rose 2014)\n"
        f"  σ_base      : {SIGMA_BASE_S * 1e6:.1f} μs  "
        f"(1 MHz ADC, Grosse & Ohtsu 2008 §3.4)\n"
        f"  Plate       : {int(PLATE_MAX*1000)}×{int(PLATE_MAX*1000)} mm  "
        f"Bolt: ({BOLT_X},{BOLT_Y}) m  "
        f"Sensors: {len(SENSORS)}\n"
        f"  Target      : {args.n_per_scenario} × {len(SCENARIOS)} = "
        f"{args.n_per_scenario * len(SCENARIOS)} samples\n"
    )

    for scenario_name, torque_loss_pct, bolt_fraction, bolt_radius, noise_mult \
            in SCENARIOS:

        print(
            f"  Scenario [{scenario_name:10s}]  "
            f"torque_loss={torque_loss_pct:3d}%  "
            f"bolt_fraction={bolt_fraction:.0%}  "
            f"r_max={bolt_radius*1000:.0f}mm  "
            f"σ={SIGMA_BASE_S * noise_mult * 1e6:.1f}μs"
        )

        df = generate_scenario(
            rng=rng,
            n=args.n_per_scenario,
            scenario_name=scenario_name,
            torque_loss_pct=torque_loss_pct,
            bolt_fraction=bolt_fraction,
            bolt_radius=bolt_radius,
            noise_multiplier=noise_mult,
        )
        frames.append(df)
        print(f"  → {len(df)} samples generated\n")

    result = pd.concat(frames, ignore_index=True)

    # Canonical column order
    result = result[[
        "scenario", "torque_loss_pct", "source_x", "source_y",
        "t1", "t2", "t3", "t4", "t5", "t6",
    ]]

    try:
        result.to_csv(output_path, index=False, float_format="%.6f")
    except OSError as exc:
        print(f"[generate_ae_data] ERROR — cannot write CSV: {exc}",
              file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - t0
    print(
        f"[generate_ae_data] {len(result)} total samples → {output_path}\n"
        f"[generate_ae_data] Runtime: {elapsed:.2f}s\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
