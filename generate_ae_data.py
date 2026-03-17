"""
generate_ae_data.py — Synthetic AE Arrival-Time Data Generator (standalone)
=============================================================================
Reproduce Section 2.1 of "Wave-Equation Constrained PINN for Acoustic Emission
Source Localization in Bolted Connections: A Cyber-Physical Digital Twin
Framework with ifcJSON Middleware" — generates synthetic AE arrival times.

Standalone script: no dependencies on belico-stack, config/params.yaml, or any
factory infrastructure. All physical parameters are hardcoded below with
annotated sources so the reader can verify every number independently.

Physics model
-------------
- Steel plate  : 300 mm × 300 mm  (0.0 – 0.3 m in x and y)
- Bolt position: (0.15, 0.15) m   (plate centre)
- Wave speed   : 5000 m/s  — Lamb S0 mode in a 6 mm steel plate at ~50 kHz.
                 Source: Rose (2014) "Ultrasonic Guided Waves in Solid Media",
                 Cambridge University Press, §5.3; typical range 4500–5400 m/s.
- Timing noise : σ = 1 µs  — 1 MHz ADC resolution, Johnson noise < 0.1 µs for
                 PAC 40 dB preamplifiers.
                 Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing", §3.4.
- 6 sensors on perimeter:
    S1=(0.00, 0.00)  S2=(0.15, 0.00)  S3=(0.30, 0.00)
    S4=(0.30, 0.30)  S5=(0.15, 0.30)  S6=(0.00, 0.30)
- Arrival time model: t_i = dist(source, S_i) / wave_speed + N(0, sigma_noise)

Four torque-loss scenarios (100 samples each by default)
---------------------------------------------------------
intact     ( 0% loss): uniform sources across plate — 5% clustered near bolt
loose_25  (25% loss): 70% within r < 0.08 m of bolt, noise ×1.5
loose_50  (50% loss): 85% within r < 0.05 m of bolt, noise ×2.0
full_loose(100% loss): 95% within r < 0.03 m of bolt, noise ×3.0

Output
------
data/processed/ae_synthetic_arrivals.csv  (relative to this script's directory)
Columns: source_x, source_y, t1, t2, t3, t4, t5, t6, scenario, torque_loss_pct
Rows   : 400 total (100 per scenario, configurable via --n-per-scenario)

Usage
-----
  python generate_ae_data.py
  python generate_ae_data.py --n-per-scenario 200 --seed 7
  python generate_ae_data.py --wave-speed 4800 --noise-sigma 2e-6
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants — all values annotated with sources
# ---------------------------------------------------------------------------

# Lamb S0 wave speed in a 6 mm steel plate at ~50 kHz centre frequency.
# Source: Rose (2014) §5.3; Giurgiutiu (2014) Table 2-1.
# Typical range: 4500–5400 m/s depending on thickness and frequency.
WAVE_SPEED_MS = 5000.0   # m/s

# Timing noise standard deviation — 1 µs.
# Rationale: 1 MHz ADC → 1 µs resolution; Johnson noise < 0.1 µs for typical
# preamplifiers (PAC 2/4/6 series, 40 dB gain).
# Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing", §3.4.
NOISE_SIGMA_S = 1.0e-6   # s

# Plate geometry — fixed for this study
PLATE_SIZE_M = 0.3       # 300 mm × 300 mm square plate (m)
PLATE_MIN    = 0.0       # m
PLATE_MAX    = 0.3       # m

# Bolt centre position
BOLT_X = 0.15            # m  (plate centre)
BOLT_Y = 0.15            # m  (plate centre)

# 6 piezoelectric sensors arranged on the plate perimeter
# Layout: 3 along the bottom edge + 3 along the top edge
# All coordinates in metres
SENSORS = [
    (0.00, 0.00),   # S1 — corner bottom-left
    (0.15, 0.00),   # S2 — edge bottom-mid
    (0.30, 0.00),   # S3 — corner bottom-right
    (0.30, 0.30),   # S4 — corner top-right
    (0.15, 0.30),   # S5 — edge top-mid
    (0.00, 0.30),   # S6 — corner top-left
]

# Scenario definitions (ordered from healthy to fully loose)
# Tuple: (scenario_name, torque_loss_pct, bolt_fraction, bolt_radius_m, noise_multiplier)
#   bolt_fraction    — fraction of events placed near the bolt centre
#   bolt_radius_m    — clustering radius around bolt (m)
#   noise_multiplier — scales NOISE_SIGMA_S (damaged bolts create more AE scatter)
SCENARIOS = [
    ("intact",      0,   0.05, 0.08, 1.0),   # ~uniform; 5% near bolt
    ("loose_25",   25,   0.70, 0.08, 1.5),
    ("loose_50",   50,   0.85, 0.05, 2.0),
    ("full_loose", 100,  0.95, 0.03, 3.0),
]

# Default samples per scenario (400 total)
N_PER_SCENARIO = 100

# Random seed for full reproducibility
RANDOM_SEED = 42

# Margin from plate boundary to avoid edge singularities (m)
_BOUNDARY_MARGIN = 0.02  # 20 mm


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    """Return (n, 2) array of source positions uniformly distributed on the
    plate with a 20 mm margin from each boundary (avoids edge singularities)."""
    lo = PLATE_MIN + _BOUNDARY_MARGIN
    hi = PLATE_MAX - _BOUNDARY_MARGIN
    return rng.uniform(lo, hi, size=(n, 2))


def _sample_near_bolt(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """Return (n, 2) array of source positions within `radius` of the bolt.

    Uses polar sampling (uniform in r², uniform in θ) to avoid centre bias.
    Coordinates are clipped to [0.02, 0.28] to stay within the plate margin.
    """
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    # Uniform-area sampling: r = sqrt(U) * radius
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    x = np.clip(BOLT_X + r * np.cos(angles), 0.02, 0.28)
    y = np.clip(BOLT_Y + r * np.sin(angles), 0.02, 0.28)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Scenario generator
# ---------------------------------------------------------------------------

def _generate_scenario(
    rng: np.random.Generator,
    n: int,
    bolt_fraction: float,
    bolt_radius: float,
    noise_multiplier: float,
    wave_speed: float,
    base_noise_sigma: float,
    scenario_name: str,
    torque_loss_pct: int,
) -> pd.DataFrame:
    """Generate `n` synthetic AE events for a single torque-loss scenario.

    Parameters
    ----------
    rng              : Seeded NumPy random generator (caller provides for reproducibility)
    n                : Number of AE events to generate
    bolt_fraction    : Fraction of events clustered near the bolt (0–1)
    bolt_radius      : Clustering radius around bolt centre (m)
    noise_multiplier : Scale factor applied to base_noise_sigma
    wave_speed       : Lamb S0 wave speed (m/s)
    base_noise_sigma : Base timing noise std deviation (s)
    scenario_name    : Label written to the 'scenario' column
    torque_loss_pct  : Integer written to the 'torque_loss_pct' column

    Returns
    -------
    pd.DataFrame with columns:
        source_x, source_y, t1, t2, t3, t4, t5, t6, scenario, torque_loss_pct
    """
    n_bolt    = int(round(n * bolt_fraction))
    n_uniform = n - n_bolt

    parts = []
    if n_bolt > 0:
        parts.append(_sample_near_bolt(rng, n_bolt, bolt_radius))
    if n_uniform > 0:
        parts.append(_sample_uniform(rng, n_uniform))

    sources = np.vstack(parts)   # (n, 2)

    # Shuffle bolt-cluster and uniform events together
    sources = sources[rng.permutation(n)]

    sigma = base_noise_sigma * noise_multiplier

    # Compute arrival times: t_i = dist(source, S_i) / wave_speed + N(0, sigma)
    arrivals = np.zeros((n, len(SENSORS)))
    for j, (sx, sy) in enumerate(SENSORS):
        dist = np.sqrt((sources[:, 0] - sx) ** 2 + (sources[:, 1] - sy) ** 2)
        noise = rng.normal(0.0, sigma, size=n)
        arrivals[:, j] = dist / wave_speed + noise

    return pd.DataFrame({
        "source_x":        sources[:, 0],
        "source_y":        sources[:, 1],
        "t1":              arrivals[:, 0],
        "t2":              arrivals[:, 1],
        "t3":              arrivals[:, 2],
        "t4":              arrivals[:, 3],
        "t5":              arrivals[:, 4],
        "t6":              arrivals[:, 5],
        "scenario":        scenario_name,
        "torque_loss_pct": torque_loss_pct,
    })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic AE arrival-time dataset for bolted-joint SHM "
            "(standalone reproducibility script)."
        )
    )
    parser.add_argument(
        "--n-per-scenario",
        type=int,
        default=N_PER_SCENARIO,
        metavar="INT",
        help=f"AE events per torque-loss scenario (default: {N_PER_SCENARIO})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        metavar="INT",
        help=f"NumPy random seed for reproducibility (default: {RANDOM_SEED})",
    )
    parser.add_argument(
        "--wave-speed",
        type=float,
        default=WAVE_SPEED_MS,
        metavar="FLOAT",
        help=f"Lamb S0 wave speed in m/s (default: {WAVE_SPEED_MS})",
    )
    parser.add_argument(
        "--noise-sigma",
        type=float,
        default=NOISE_SIGMA_S,
        metavar="FLOAT",
        help=f"Base timing noise std dev in seconds (default: {NOISE_SIGMA_S})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output directory (default: data/processed/ relative to this script)",
    )
    return parser


def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    parser = _build_parser()
    args = parser.parse_args()

    # Resolve output directory relative to this script (not CWD),
    # so the script works regardless of where it is invoked from.
    script_dir = Path(__file__).resolve().parent
    if args.output_dir is not None:
        output_dir = args.output_dir if args.output_dir.is_absolute() \
                     else script_dir / args.output_dir
    else:
        output_dir = script_dir / "data" / "processed"

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"[generate_ae_data] ERROR — cannot create output dir: {exc}",
              file=sys.stderr)
        return 1

    output_path = output_dir / "ae_synthetic_arrivals.csv"

    # Generate data for each scenario
    rng    = np.random.default_rng(args.seed)
    frames = []

    for scenario_name, torque_loss_pct, bolt_fraction, bolt_radius, noise_mult \
            in SCENARIOS:
        df = _generate_scenario(
            rng=rng,
            n=args.n_per_scenario,
            bolt_fraction=bolt_fraction,
            bolt_radius=bolt_radius,
            noise_multiplier=noise_mult,
            wave_speed=args.wave_speed,
            base_noise_sigma=args.noise_sigma,
            scenario_name=scenario_name,
            torque_loss_pct=torque_loss_pct,
        )
        frames.append(df)
        print(f"  Generated {len(df)} samples  [scenario={scenario_name}]")

    result = pd.concat(frames, ignore_index=True)

    try:
        result.to_csv(output_path, index=False, float_format="%.9e")
    except OSError as exc:
        print(f"[generate_ae_data] ERROR — cannot write CSV: {exc}",
              file=sys.stderr)
        return 1

    print(f"\nGenerated {len(result)} samples total → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
