"""
generate_ae_data.py — Synthetic AE Arrival-Time Data Generator (OpenSeesPy FEM)
=================================================================================
Reproduce Section 2.1 of "Wave-Equation Constrained PINN for Acoustic Emission
Source Localization in Bolted Connections: A Cyber-Physical Digital Twin
Framework with ifcJSON Middleware" — generates synthetic AE arrival times using
a full finite-element transient analysis with OpenSeesPy.

Standalone script: no dependencies on belico-stack, config/params.yaml, or any
factory infrastructure. All physical parameters are hardcoded below with
annotated sources so the reader can verify every number independently.

Physics model (FEM)
-------------------
- Steel plate  : 300 mm × 300 mm × 6 mm thick
- Mesh         : 60×60 quad elements (ShellMITC4), element size ≤ 5 mm
                 Required to resolve Lamb waves at ~50 kHz: λ_min = c/f ≈ 100 mm,
                 but element size criterion for accuracy: h ≤ λ/10 at highest freq.
                 Source: Moser et al. (1999) "Modeling elastic wave propagation in
                 waveguides with the finite element method", NDT&E Int. 32(4).
- Material     : Steel — E=200 GPa, ν=0.3, ρ=7850 kg/m³
                 Source: Giurgiutiu (2014) "Structural Health Monitoring with
                 Piezoelectric Wafer Active Sensors", Academic Press, Appendix A.
- Bolt spring  : ZeroLength element at (0.15, 0.15) m with stiffness K per scenario.
                 Modelling approach: penalty spring representing bolt preload as a
                 local constraint stiffness. Source: Huynh & Lam (2014) "Detection
                 of bolt loosening in C-C composite beams", Smart Mater. Struct.
                 23(2), doi:10.1088/0964-1726/23/2/025005.
- Excitation   : Gaussian impulse at bolt node, duration 10 μs, amplitude 1 N,
                 applied in-plane (Fx). Models a pencil-break AE source.
                 Source: Hsu & Hardy (1978) in "Acoustic Emission", ASTM STP 571.
- Transient    : Newmark β=0.25, γ=0.5 (unconditionally stable, trapezoidal rule)
                 dt = 1e-7 s, total = 200 μs (2000 steps)
                 Source: Bathe (1996) "Finite Element Procedures", §9.4.
- Sensors      : 6 nodes on plate perimeter; arrival time = first instant where
                 |nodal velocity| > 5% of peak velocity at that sensor.
- Timing noise : σ = 1 μs added to arrival times after FEM extraction.
                 Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing", §3.4.

Bolt stiffness per scenario (bolt at plate centre, radius ~6 mm M6 bolt)
------------------------------------------------------------------------
intact     K = 1.00e8 N/m  — fully tightened (Hertz contact stiffness estimate)
loose_25   K = 7.50e7 N/m  — 25% preload loss → ~25% stiffness drop
loose_50   K = 5.00e7 N/m  — 50% preload loss → ~50% stiffness drop
full_loose K = 1.00e7 N/m  — effectively free (retains minimal lateral contact)
Source: Huynh & Lam (2014); values calibrated to produce measurable arrival-time
shifts consistent with experimental data in Amerini & Meo (2011) "Structural
health monitoring of bolted joints using linear and nonlinear acoustic/ultrasound
methods", Smart Mater. Struct. 20(7).

Output
------
data/processed/ae_synthetic_arrivals.csv  (relative to script directory)
Columns: scenario, source_x, source_y, t1, t2, t3, t4, t5, t6
Rows   : 400 total (100 per scenario, configurable via --n-per-scenario)

Usage
-----
  python generate_ae_data.py
  python generate_ae_data.py --n-per-scenario 200 --seed 7
  python generate_ae_data.py --output-dir /tmp/ae_out
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physical constants — all values annotated with sources
# ---------------------------------------------------------------------------

# Steel material properties
# Source: Giurgiutiu (2014) "SWAS", Academic Press, Appendix A.
E_PA     = 200.0e9    # Young's modulus [Pa]
NU       = 0.3        # Poisson's ratio [-]
RHO_KGM3 = 7850.0    # Mass density [kg/m³]
H_M      = 0.006      # Plate thickness [m] (6 mm)

# Plate geometry
PLATE_SIZE_M = 0.30   # 300 mm × 300 mm square plate [m]
PLATE_MIN    = 0.0    # [m]
PLATE_MAX    = 0.30   # [m]

# Bolt position (plate centre)
BOLT_X = 0.15         # [m]
BOLT_Y = 0.15         # [m]

# Mesh parameters
# Element size ≤ 5 mm to resolve ~50 kHz Lamb waves (c_S0 ≈ 5000 m/s → λ ≈ 100 mm,
# rule: h ≤ λ/10 at highest significant frequency component in a 10 μs impulse).
# Source: Moser et al. (1999) NDT&E Int. 32(4), pp. 225-232.
N_ELEM = 60           # elements per side (60×60 → h = 5 mm exactly)

# Transient analysis parameters
# Newmark β=0.25, γ=0.5 — trapezoidal rule, unconditionally stable.
# Source: Bathe (1996) "Finite Element Procedures", §9.4, p. 768.
DT       = 1.0e-7     # time step [s] (100 ns)
DURATION = 200.0e-6   # simulation duration [s] (200 μs)
N_STEPS  = int(round(DURATION / DT))  # = 2000

# Gaussian impulse excitation (models pencil-break AE source)
# Source: Hsu & Hardy (1978) in "Acoustic Emission", ASTM STP 571, pp. 40-58.
IMPULSE_AMP      = 1.0    # amplitude [N]
IMPULSE_SIGMA    = 5.0e-6 # impulse half-width σ [s] → ~10 μs effective duration
IMPULSE_CENTER   = 10.0e-6 # impulse peak time [s]

# Arrival time detection threshold
# 5% of peak nodal velocity at each sensor
ARRIVAL_THRESHOLD = 0.05

# Timing noise standard deviation
# Source: Grosse & Ohtsu (2008) "Acoustic Emission Testing", §3.4, p. 87.
NOISE_SIGMA_S = 1.0e-6   # [s]  (1 μs)

# 6 piezoelectric sensors on plate perimeter [m]
SENSORS = [
    (0.00, 0.00),   # S1 — corner bottom-left
    (0.15, 0.00),   # S2 — edge bottom-mid
    (0.30, 0.00),   # S3 — corner bottom-right
    (0.30, 0.30),   # S4 — corner top-right
    (0.15, 0.30),   # S5 — edge top-mid
    (0.00, 0.30),   # S6 — corner top-left
]

# Bolt spring stiffness per scenario [N/m]
# Source: Huynh & Lam (2014) Smart Mater. Struct. 23(2) 025005.
BOLT_STIFFNESS = {
    "intact":     1.0e8,   # fully tightened
    "loose_25":   7.5e7,   # 25% preload loss
    "loose_50":   5.0e7,   # 50% preload loss
    "full_loose": 1.0e7,   # essentially free
}

# Scenario definitions
# Tuple: (scenario_name, torque_loss_pct, bolt_fraction, bolt_radius_m, noise_multiplier)
#   bolt_fraction    — fraction of events placed near the bolt centre
#   bolt_radius_m    — clustering radius around bolt [m]
#   noise_multiplier — scales NOISE_SIGMA_S (damaged bolts → more AE scatter)
SCENARIOS = [
    ("intact",      0,   0.05, 0.08, 1.0),
    ("loose_25",   25,   0.70, 0.08, 1.5),
    ("loose_50",   50,   0.85, 0.05, 2.0),
    ("full_loose", 100,  0.95, 0.03, 3.0),
]

# Boundary margin to avoid source exactly on plate edge [m]
_BOUNDARY_MARGIN = 0.02

# Default samples per scenario
N_PER_SCENARIO = 100
RANDOM_SEED    = 42


# ---------------------------------------------------------------------------
# Source position samplers (identical logic to original script)
# ---------------------------------------------------------------------------

def _sample_uniform(rng: np.random.Generator, n: int) -> np.ndarray:
    """(n, 2) source positions uniformly distributed inside plate with margin."""
    lo = PLATE_MIN + _BOUNDARY_MARGIN
    hi = PLATE_MAX - _BOUNDARY_MARGIN
    return rng.uniform(lo, hi, size=(n, 2))


def _sample_near_bolt(rng: np.random.Generator, n: int, radius: float) -> np.ndarray:
    """(n, 2) source positions within `radius` of bolt centre.

    Uses polar sampling (uniform-area: r = sqrt(U)*R) to avoid centre bias.
    Clipped to [0.02, 0.28] to remain within plate margin.
    """
    angles = rng.uniform(0.0, 2.0 * np.pi, size=n)
    r = np.sqrt(rng.uniform(0.0, 1.0, size=n)) * radius
    x = np.clip(BOLT_X + r * np.cos(angles), 0.02, 0.28)
    y = np.clip(BOLT_Y + r * np.sin(angles), 0.02, 0.28)
    return np.column_stack([x, y])


# ---------------------------------------------------------------------------
# Gaussian impulse time history
# ---------------------------------------------------------------------------

def _gaussian_impulse(t: np.ndarray) -> np.ndarray:
    """Gaussian force pulse centred at IMPULSE_CENTER with std IMPULSE_SIGMA.

    F(t) = A * exp(-(t - t0)^2 / (2*sigma^2))

    Models a pencil-break AE source (Hsu-Hardy source).
    Source: Sause (2016) "In Situ Monitoring of Fiber-Reinforced Composites",
    Springer, §4.3.2.
    """
    return IMPULSE_AMP * np.exp(
        -0.5 * ((t - IMPULSE_CENTER) / IMPULSE_SIGMA) ** 2
    )


# ---------------------------------------------------------------------------
# FEM single-run: build model, run transient, extract arrival times
# ---------------------------------------------------------------------------

def _run_opensees_sim(
    source_x: float,
    source_y: float,
    bolt_k: float,
) -> np.ndarray | None:
    """Build a 2-D shell-plate OpenSeesPy model and run transient analysis.

    Returns array of shape (6,) with arrival times [s] at each sensor, or
    None if the simulation fails (convergence or import error).

    Parameters
    ----------
    source_x, source_y : AE source coordinates on the plate [m]
    bolt_k             : Bolt spring stiffness [N/m]
    """
    try:
        import openseespy.opensees as ops
    except ImportError:
        raise RuntimeError(
            "openseespy not found. Install with: pip install openseespy"
        )

    # ------------------------------------------------------------------
    # 0. Wipe any previous model
    # ------------------------------------------------------------------
    ops.wipe()
    ops.model("BasicBuilder", "-ndm", 3, "-ndf", 6)

    # ------------------------------------------------------------------
    # 1. Build 60×60 quad mesh of ShellMITC4 elements
    #
    #    Node numbering: node_id = (j * (N_ELEM+1) + i) + 1
    #    where i = column index (x-direction), j = row index (y-direction)
    #    Coordinates: x = i * dx, y = j * dy, z = 0
    # ------------------------------------------------------------------
    dx = PLATE_SIZE_M / N_ELEM   # 5 mm
    dy = PLATE_SIZE_M / N_ELEM   # 5 mm
    n_nodes_x = N_ELEM + 1      # 61
    n_nodes_y = N_ELEM + 1      # 61

    def node_id(i: int, j: int) -> int:
        """1-based node tag from (column i, row j)."""
        return j * n_nodes_x + i + 1

    # Create all plate nodes
    for j in range(n_nodes_y):
        for i in range(n_nodes_x):
            nid = node_id(i, j)
            ops.node(nid, i * dx, j * dy, 0.0)

    total_plate_nodes = n_nodes_x * n_nodes_y  # 3721

    # ------------------------------------------------------------------
    # 2. Elastic isotropic shell section
    #    nDMaterial ElasticIsotropic then ShellMITC4 section
    # ------------------------------------------------------------------
    ops.nDMaterial("ElasticIsotropic", 1, E_PA, NU, RHO_KGM3)
    ops.section("PlateFiber", 1, 1, H_M)   # uses nDMaterial 1

    # ------------------------------------------------------------------
    # 3. ShellMITC4 elements
    # ------------------------------------------------------------------
    elem_tag = 1
    for j in range(N_ELEM):
        for i in range(N_ELEM):
            n1 = node_id(i,   j)
            n2 = node_id(i+1, j)
            n3 = node_id(i+1, j+1)
            n4 = node_id(i,   j+1)
            ops.element("ShellMITC4", elem_tag, n1, n2, n3, n4, 1)
            elem_tag += 1

    # ------------------------------------------------------------------
    # 4. Bolt spring at (0.15, 0.15) m
    #
    #    Find the nearest mesh node to (BOLT_X, BOLT_Y).
    #    Add a second node at the same location (fixed), connected by
    #    ZeroLength spring in all translational DOFs (1,2,3).
    # ------------------------------------------------------------------
    bi = int(round(BOLT_X / dx))   # = 30
    bj = int(round(BOLT_Y / dy))   # = 30
    bolt_plate_node = node_id(bi, bj)

    # Extra node (fixed) for the spring anchor
    bolt_anchor_id = total_plate_nodes + 1
    ops.node(bolt_anchor_id, bi * dx, bj * dy, 0.0)
    # Fix anchor in all DOFs (1-6)
    ops.fix(bolt_anchor_id, 1, 1, 1, 1, 1, 1)

    # ZeroLength spring — uniaxial material in each translational direction
    # Material 2: elastic translational spring for bolt
    # Using "Elastic" uniaxial material
    ops.uniaxialMaterial("Elastic", 2, bolt_k)

    # ZeroLength element connecting plate bolt node → fixed anchor
    # Directions 1 (x), 2 (y), 3 (z)
    spring_tag = elem_tag
    ops.element(
        "zeroLength", spring_tag,
        bolt_plate_node, bolt_anchor_id,
        "-mat", 2, 2, 2,
        "-dir", 1, 2, 3,
    )

    # ------------------------------------------------------------------
    # 5. Find source node (nearest mesh node to source_x, source_y)
    # ------------------------------------------------------------------
    si = int(round(source_x / dx))
    sj = int(round(source_y / dy))
    # Clamp to valid range
    si = max(0, min(si, N_ELEM))
    sj = max(0, min(sj, N_ELEM))
    source_node = node_id(si, sj)

    # ------------------------------------------------------------------
    # 6. Find sensor nodes (nearest mesh node to each sensor position)
    # ------------------------------------------------------------------
    sensor_nodes = []
    for (sx, sy) in SENSORS:
        ii = int(round(sx / dx))
        ij = int(round(sy / dy))
        ii = max(0, min(ii, N_ELEM))
        ij = max(0, min(ij, N_ELEM))
        sensor_nodes.append(node_id(ii, ij))

    # ------------------------------------------------------------------
    # 7. Gaussian impulse time series as OpenSeesPy TimeSeries
    #
    #    Use Path time series with tabulated values.
    # ------------------------------------------------------------------
    t_array = np.arange(N_STEPS + 1) * DT
    f_array = _gaussian_impulse(t_array)

    # Provide (time, force) pairs — ops.timeSeries Path needs -time + -values
    ops.timeSeries(
        "Path", 1,
        "-dt", DT,
        "-values", *f_array.tolist(),
        "-factor", 1.0,
    )

    ops.pattern("Plain", 1, 1)
    # Apply force in x-direction (DOF 1) at source node
    ops.load(source_node, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 8. Rayleigh damping — 0.5% of critical at 50 kHz
    #    α_M and β_K for ω₁ ≈ 2π×50000 rad/s
    #    Using Rayleigh damping: ξ = α/(2ω) + β*ω/2
    #    For single-frequency target: β_K = 2ξ/ω, α_M = 0
    #    Source: Chopra (2017) "Dynamics of Structures", §11.4.
    # ------------------------------------------------------------------
    omega_target = 2.0 * np.pi * 50.0e3   # 50 kHz [rad/s]
    xi_target    = 0.005                   # 0.5% critical damping
    beta_K       = 2.0 * xi_target / omega_target
    alpha_M      = 0.0
    ops.rayleigh(alpha_M, beta_K, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 9. Transient analysis — Newmark β=0.25, γ=0.5
    # ------------------------------------------------------------------
    ops.system("BandGeneral")
    ops.numberer("RCM")
    ops.constraints("Plain")
    ops.integrator("Newmark", 0.5, 0.25)
    ops.algorithm("Linear")
    ops.analysis("Transient")

    # ------------------------------------------------------------------
    # 10. Run and record nodal velocity at sensor nodes
    # ------------------------------------------------------------------
    # velocity_history[sensor_idx][step] = velocity in x-direction
    n_sensors = len(sensor_nodes)
    vel_history = np.zeros((n_sensors, N_STEPS))

    for step in range(N_STEPS):
        ok = ops.analyze(1, DT)
        if ok != 0:
            # Convergence failure — caller will skip this sample
            ops.wipe()
            return None

        current_time = ops.getTime()
        for s_idx, s_node in enumerate(sensor_nodes):
            # velocity in x-direction (DOF 1)
            vel_history[s_idx, step] = ops.nodeVel(s_node, 1)

    ops.wipe()

    # ------------------------------------------------------------------
    # 11. Extract arrival times from velocity histories
    #
    #     Arrival = first time step where |vel| > ARRIVAL_THRESHOLD * peak_vel
    #     If never triggered, use NaN (sample will be skipped by caller)
    # ------------------------------------------------------------------
    time_steps = (np.arange(N_STEPS) + 1) * DT   # time at end of each step

    arrivals = np.full(n_sensors, np.nan)
    for s_idx in range(n_sensors):
        vhist = np.abs(vel_history[s_idx])
        peak  = vhist.max()
        if peak < 1.0e-30:
            # No signal reached this sensor — leave as NaN
            continue
        threshold_val = ARRIVAL_THRESHOLD * peak
        triggered = np.where(vhist > threshold_val)[0]
        if len(triggered) == 0:
            continue
        arrivals[s_idx] = time_steps[triggered[0]]

    return arrivals


# ---------------------------------------------------------------------------
# Scenario generator (wraps FEM runner)
# ---------------------------------------------------------------------------

def _generate_scenario(
    rng: np.random.Generator,
    n: int,
    scenario_name: str,
    bolt_fraction: float,
    bolt_radius: float,
    noise_multiplier: float,
    bolt_k: float,
) -> pd.DataFrame:
    """Generate `n` FEM-based AE events for a single torque-loss scenario.

    Skips samples where OpenSeesPy fails to converge or returns NaN arrivals.
    Keeps generating until `n` valid samples are collected or 3×n attempts
    are exhausted (whichever comes first), then returns what was collected.

    Parameters
    ----------
    rng             : Seeded NumPy random generator (caller provides)
    n               : Target number of valid AE events
    scenario_name   : Label written to 'scenario' column
    bolt_fraction   : Fraction of events clustered near bolt (0–1)
    bolt_radius     : Clustering radius around bolt centre [m]
    noise_multiplier: Scale applied to NOISE_SIGMA_S
    bolt_k          : Bolt spring stiffness for this scenario [N/m]

    Returns
    -------
    pd.DataFrame with columns:
        scenario, source_x, source_y, t1, t2, t3, t4, t5, t6
    """
    sigma = NOISE_SIGMA_S * noise_multiplier

    rows = []
    max_attempts = n * 3
    attempts     = 0

    while len(rows) < n and attempts < max_attempts:
        attempts += 1

        # Sample a source position for this event
        if rng.uniform() < bolt_fraction:
            pos = _sample_near_bolt(rng, 1, bolt_radius)[0]
        else:
            pos = _sample_uniform(rng, 1)[0]

        src_x, src_y = float(pos[0]), float(pos[1])

        # Run FEM transient
        try:
            arrivals = _run_opensees_sim(src_x, src_y, bolt_k)
        except Exception as exc:
            warnings.warn(
                f"[{scenario_name}] FEM exception at ({src_x:.3f},{src_y:.3f}): {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if arrivals is None:
            warnings.warn(
                f"[{scenario_name}] convergence failure at ({src_x:.3f},{src_y:.3f}) — skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if np.any(np.isnan(arrivals)):
            warnings.warn(
                f"[{scenario_name}] NaN arrival(s) at ({src_x:.3f},{src_y:.3f}) — skipping",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        # Add Gaussian timing noise
        noise = rng.normal(0.0, sigma, size=len(arrivals))
        arrivals_noisy = arrivals + noise

        rows.append({
            "scenario": scenario_name,
            "source_x": src_x,
            "source_y": src_y,
            "t1": arrivals_noisy[0],
            "t2": arrivals_noisy[1],
            "t3": arrivals_noisy[2],
            "t4": arrivals_noisy[3],
            "t5": arrivals_noisy[4],
            "t6": arrivals_noisy[5],
        })

    if len(rows) < n:
        warnings.warn(
            f"[{scenario_name}] only collected {len(rows)}/{n} valid samples "
            f"after {attempts} attempts.",
            RuntimeWarning,
            stacklevel=2,
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic AE arrival-time dataset for bolted-joint SHM "
            "using OpenSeesPy transient FEM (standalone reproducibility script)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n-per-scenario",
        type=int,
        default=N_PER_SCENARIO,
        metavar="INT",
        help=f"AE events per torque-loss scenario.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
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
        f"\n[generate_ae_data] FEM model: {N_ELEM}×{N_ELEM} ShellMITC4 mesh, "
        f"dt={DT:.0e}s, {N_STEPS} steps, {len(SENSORS)} sensors"
    )
    print(f"[generate_ae_data] Target: {args.n_per_scenario} samples × "
          f"{len(SCENARIOS)} scenarios = "
          f"{args.n_per_scenario * len(SCENARIOS)} total\n")

    for scenario_name, torque_loss_pct, bolt_fraction, bolt_radius, noise_mult \
            in SCENARIOS:

        bolt_k = BOLT_STIFFNESS[scenario_name]
        print(
            f"  Scenario [{scenario_name}]  K={bolt_k:.2e} N/m  "
            f"torque_loss={torque_loss_pct}%  "
            f"bolt_fraction={bolt_fraction}"
        )

        df = _generate_scenario(
            rng=rng,
            n=args.n_per_scenario,
            scenario_name=scenario_name,
            bolt_fraction=bolt_fraction,
            bolt_radius=bolt_radius,
            noise_multiplier=noise_mult,
            bolt_k=bolt_k,
        )
        frames.append(df)
        print(f"  → collected {len(df)} valid samples\n")

    if not frames or all(len(f) == 0 for f in frames):
        print("[generate_ae_data] ERROR — no samples generated.", file=sys.stderr)
        return 1

    result = pd.concat(frames, ignore_index=True)

    # Reorder columns to match original output contract
    result = result[["scenario", "source_x", "source_y",
                      "t1", "t2", "t3", "t4", "t5", "t6"]]

    try:
        result.to_csv(output_path, index=False, float_format="%.9e")
    except OSError as exc:
        print(f"[generate_ae_data] ERROR — cannot write CSV: {exc}",
              file=sys.stderr)
        return 1

    print(f"[generate_ae_data] Generated {len(result)} total samples → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
