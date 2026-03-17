"""
train_pinn.py — Physics-Informed Neural Network for AE Source Localization (standalone)
=========================================================================================
Reproduce Table 1 of "Wave-Equation Constrained PINN for Acoustic Emission Source
Localization in Bolted Connections: A Cyber-Physical Digital Twin Framework with
ifcJSON Middleware".

Standalone script: no dependencies on belico-stack, config/params.yaml, or any
factory infrastructure. All hyperparameters and physical constants are hardcoded
below with comments explaining each choice.

Architecture
------------
MLP  6 → [64 × 64 × 64 × 64] → 2
  Input  : [t1..t6] — normalized arrival times (6 sensors)
  Output : [x, y]  — source coordinates normalized to [0, 1] (Sigmoid)
  Hidden : 4 × 64 neurons, Tanh activations, Xavier-uniform initialization

Hybrid loss (Wu et al. 2023 normalized weighting)
--------------------------------------------------
  L_total = (L_data / L_data_0) + λ * (L_physics / L_phys_0)

  L_data    = MSE(pred_xy, true_xy)           — data fidelity
  L_physics = MSE(t̂_i_norm, t_i_norm)        — wave-equation residual
              where t̂_i = ||pred_source - S_i||_2 / wave_speed  (seconds)
              converted to microseconds and standardized with the same
              z-score normalization used for network inputs.

  Normalization by initial losses (L_data_0, L_phys_0) makes the relative
  weight λ=0.1 scale-invariant: physics contributes 10% relative to data
  regardless of their absolute magnitudes (Wu et al., NeurIPS 2023).

Physical constants
------------------
  plate_size  = 0.3 m        — 300 mm × 300 mm steel plate
  wave_speed  = 5000 m/s     — Lamb S0 mode (Rose 2014, §5.3)
  6 sensors at perimeter (same layout as generate_ae_data.py)

Usage
-----
  # Step 1 — generate data
  python generate_ae_data.py

  # Step 2 — train PINN and print Table 1
  python train_pinn.py

  # With custom hyperparameters
  python train_pinn.py --epochs 1000 --lambda-phys 0.05

Outputs
-------
  data/processed/pinn_localization_results.csv  — per-sample predictions + error
  data/processed/training_history.csv           — per-epoch loss breakdown
  models/pinn_localization.pt                   — PyTorch model state dict
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths — all relative to this script's directory (standalone)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR   = SCRIPT_DIR / "data" / "processed"
MODELS_DIR = SCRIPT_DIR / "models"

# ---------------------------------------------------------------------------
# Physical constants (annotated)
# ---------------------------------------------------------------------------

# Steel plate dimensions — 300 mm × 300 mm
PLATE_SIZE_M = 0.30    # m

# Lamb S0 wave speed in steel plate (~6 mm thick, ~50 kHz)
# Source: Rose (2014) "Ultrasonic Guided Waves in Solid Media" §5.3
WAVE_SPEED_MS = 5000.0   # m/s

# Conversion factor: seconds → microseconds (used for input normalization)
US_SCALE = 1e6

# 6 piezoelectric sensor positions on perimeter (metres)
# Same layout as generate_ae_data.py — must match exactly for physics to work
SENSOR_POSITIONS = [
    [0.00, 0.00],   # S1
    [0.15, 0.00],   # S2
    [0.30, 0.00],   # S3
    [0.30, 0.30],   # S4
    [0.15, 0.30],   # S5
    [0.00, 0.30],   # S6
]

# ---------------------------------------------------------------------------
# Hyperparameters (all hardcoded — change here to reproduce ablation studies)
# ---------------------------------------------------------------------------

EPOCHS      = 500        # training epochs
HIDDEN_SIZE = 64         # neurons per hidden layer  (hidden=[64,64,64,64])
N_LAYERS    = 4          # number of hidden layers
LAMBDA_PHYS = 0.1        # physics loss weight λ (relative, normalized)
BATCH_SIZE  = 32         # mini-batch size
LR          = 1e-3       # Adam learning rate
SEED        = 42         # random seed (data split + weight init)
TRAIN_FRAC  = 0.80       # 80/20 stratified split by scenario

# Save a model checkpoint every N epochs if validation MAE improves
CHECKPOINT_INTERVAL = 100

# ---------------------------------------------------------------------------
# PyTorch import guard
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        print(
            "[ERROR] PyTorch not installed.\n"
            "  pip install torch\n"
            "  or: pip install torch --index-url https://download.pytorch.org/whl/cpu",
            file=sys.stderr,
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_arrivals(data_dir: Path):
    """Load ae_synthetic_arrivals.csv from data_dir.

    Returns
    -------
    t_raw    : ndarray (N, 6)  — arrival times in seconds
    xy_true  : ndarray (N, 2)  — source coordinates in metres
    scenarios: list[str]       — scenario label per row
    torques  : list[float]     — torque_loss_pct per row
    """
    csv_path = data_dir / "ae_synthetic_arrivals.csv"
    if not csv_path.exists():
        print(
            f"[ERROR] Data file not found: {csv_path}\n"
            f"  Run first: python generate_ae_data.py",
            file=sys.stderr,
        )
        sys.exit(1)

    t_rows, xy_rows, scenarios, torques = [], [], [], []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            t_rows.append([float(row[f"t{i}"]) for i in range(1, 7)])
            xy_rows.append([float(row["source_x"]), float(row["source_y"])])
            scenarios.append(row["scenario"])
            torques.append(float(row["torque_loss_pct"]))

    t_raw   = np.array(t_rows,  dtype=np.float32)
    xy_true = np.array(xy_rows, dtype=np.float32)
    print(f"[INFO] Loaded {len(t_rows)} samples from {csv_path.name}")
    return t_raw, xy_true, scenarios, torques


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def build_normalizers(t_raw_train: np.ndarray):
    """Compute normalization statistics on training data only.

    Arrival times:  seconds → microseconds (×1e6), then z-score (μ, σ)
    Coordinates  :  [0, PLATE_SIZE_M] → [0, 1] via divide by PLATE_SIZE_M

    Returns callables and stats dict for later use in physics residual.
    """
    t_us   = t_raw_train * US_SCALE
    t_mean = t_us.mean(axis=0)                # (6,)
    t_std  = t_us.std(axis=0) + 1e-8          # (6,) — avoid division by zero

    def normalize_t(t_s: np.ndarray) -> np.ndarray:
        return (t_s * US_SCALE - t_mean) / t_std

    def normalize_xy(xy_m: np.ndarray) -> np.ndarray:
        return xy_m / PLATE_SIZE_M            # → [0, 1]

    def denormalize_xy(xy_norm: np.ndarray) -> np.ndarray:
        return xy_norm * PLATE_SIZE_M         # → metres

    return normalize_t, normalize_xy, denormalize_xy, {"t_mean": t_mean, "t_std": t_std}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(hidden: int, n_layers: int):
    """Build MLP: 6 → [hidden × n_layers] → 2, Tanh + Sigmoid output.

    Sigmoid on the output layer constrains predictions to [0, 1]
    (normalized plate coordinates), providing a hard physical bound.
    Xavier-uniform initialization is standard for Tanh networks.
    """
    torch, nn, _ = _require_torch()

    layers = [nn.Linear(6, hidden), nn.Tanh()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden, hidden), nn.Tanh()]
    layers += [nn.Linear(hidden, 2), nn.Sigmoid()]

    model = nn.Sequential(*layers)

    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    torch, nn, F = _require_torch()

    # Reproducibility
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    # Load data
    t_raw, xy_true, scenarios, torques = load_arrivals(Path(args.data_dir))

    # Stratified 80/20 split by scenario
    unique_sc = sorted(set(scenarios))
    train_idx, test_idx = [], []
    for sc in unique_sc:
        idx = [i for i, s in enumerate(scenarios) if s == sc]
        rng.shuffle(idx)
        split = max(1, int(math.ceil(len(idx) * TRAIN_FRAC)))
        train_idx.extend(idx[:split])
        test_idx.extend(idx[split:])
    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)
    print(f"[INFO] Split: train={len(train_idx)}, test={len(test_idx)} "
          f"(stratified 80/20 by scenario)")

    # Normalizers fitted on training data only
    normalize_t, normalize_xy, denormalize_xy, stats = build_normalizers(
        t_raw[train_idx]
    )

    X_all = normalize_t(t_raw).astype(np.float32)
    Y_all = normalize_xy(xy_true).astype(np.float32)

    X_train = torch.tensor(X_all[train_idx])
    Y_train = torch.tensor(Y_all[train_idx])
    X_test  = torch.tensor(X_all[test_idx])

    # Model, optimizer
    model = build_model(args.hidden, args.layers)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse   = nn.MSELoss()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Model: MLP 6→[{args.hidden}×{args.layers}]→2  params={n_params}")
    print(f"[INFO] Hyperparams: epochs={args.epochs}, lr={args.lr}, "
          f"batch={args.batch_size}, λ_phys={args.lambda_phys}")
    print(f"[INFO] Physics: t̂_i = ||pred - S_i||/c, c={WAVE_SPEED_MS:.0f} m/s")

    # Sensor tensor (reused every batch)
    sensor_t = torch.tensor(SENSOR_POSITIONS, dtype=torch.float32)   # (6, 2)

    # Normalization stats as tensors (for physics residual in normalized space)
    t_mean_t = torch.tensor(stats["t_mean"], dtype=torch.float32)    # (6,)
    t_std_t  = torch.tensor(stats["t_std"],  dtype=torch.float32)    # (6,)

    # Create output dirs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # --- Initial-loss calibration (Wu et al. 2023) ---
    # Compute L_data_0 and L_phys_0 on first batch so that relative weights
    # are scale-invariant: loss = (L_data / L_data_0) + λ * (L_phys / L_phys_0)
    # This guarantees λ=0.1 means "physics contributes 10% relative to data"
    # regardless of the absolute magnitude of each term at initialization.
    n_train    = len(X_train)
    batch_size = args.batch_size

    model.eval()
    with torch.no_grad():
        Xb0 = X_train[:batch_size]
        Yb0 = Y_train[:batch_size]
        pred0      = model(Xb0)
        L_data_0   = F.mse_loss(pred0, Yb0).item() + 1e-8

        # Wave-equation residual on first batch
        pred_m0    = pred0 * PLATE_SIZE_M                              # metres
        diffs0     = pred_m0.unsqueeze(1) - sensor_t.unsqueeze(0)     # (B, 6, 2)
        dists0     = torch.sqrt((diffs0 ** 2).sum(2).clamp(min=1e-8)) # (B, 6)
        t_hat_us0  = (dists0 / WAVE_SPEED_MS) * US_SCALE              # μs
        t_hat_n0   = (t_hat_us0 - t_mean_t) / t_std_t                 # normalized
        L_phys_0   = F.mse_loss(t_hat_n0, Xb0).item() + 1e-8
    model.train()

    print(f"[INFO] Loss calibration: L_data_0={L_data_0:.6f}  "
          f"L_phys_0={L_phys_0:.6f}  "
          f"→ physics relative weight = {args.lambda_phys*100:.0f}%")

    # Training loop
    history_rows = []   # (epoch, loss_total, loss_data, loss_phys, val_mae_mm)
    best_val_mae = float("inf")
    best_state   = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_train)
        ep_total = ep_data = ep_phys = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            b_idx = perm[start: start + batch_size]
            Xb, Yb = X_train[b_idx], Y_train[b_idx]

            # Data loss
            pred_b    = model(Xb)
            loss_data = mse(pred_b, Yb)

            # Physics loss: wave-equation residual
            # Step 1 — denormalize prediction to metres
            pred_m = pred_b * PLATE_SIZE_M                              # (B, 2)
            # Step 2 — theoretical arrival times for each sensor
            #   t̂_i = ||pred_source - S_i|| / wave_speed
            diffs  = pred_m.unsqueeze(1) - sensor_t.unsqueeze(0)       # (B, 6, 2)
            dists  = torch.sqrt((diffs ** 2).sum(2).clamp(min=1e-8))   # (B, 6) m
            # Step 3 — convert to µs, then z-score normalize
            #   so physics residual lives in the same numerical range as L_data
            t_hat_us   = (dists / WAVE_SPEED_MS) * US_SCALE             # µs
            t_hat_norm = (t_hat_us - t_mean_t) / t_std_t                # normalized
            # Step 4 — penalize against normalized measured arrival times (Xb)
            loss_phys  = F.mse_loss(t_hat_norm, Xb)

            # Normalized total loss (scale-invariant relative weighting)
            loss_total = (loss_data / L_data_0) + args.lambda_phys * (loss_phys / L_phys_0)

            optim.zero_grad()
            loss_total.backward()
            optim.step()

            ep_total += loss_total.item()
            ep_data  += loss_data.item()
            ep_phys  += loss_phys.item()
            n_batches += 1

        avg_total = ep_total / n_batches
        avg_data  = ep_data  / n_batches
        avg_phys  = ep_phys  / n_batches

        # Validation MAE (mm)
        model.eval()
        with torch.no_grad():
            pred_val   = model(X_test).numpy()
            pred_val_m = denormalize_xy(pred_val)
            true_val_m = xy_true[test_idx]
            errs_mm    = np.sqrt(((pred_val_m - true_val_m) ** 2).sum(axis=1)) * 1000.0
            val_mae_mm = float(errs_mm.mean())

        history_rows.append((epoch, avg_total, avg_data, avg_phys, val_mae_mm))

        # Checkpoint on improvement
        if epoch % CHECKPOINT_INTERVAL == 0:
            if val_mae_mm < best_val_mae:
                best_val_mae = val_mae_mm
                best_state   = {k: v.clone() for k, v in model.state_dict().items()}
                ckpt = MODELS_DIR / "pinn_localization.pt"
                torch.save(best_state, ckpt)
                print(f"  [CKPT] Epoch {epoch:4d} — val_mae={val_mae_mm:.2f} mm → {ckpt}")

        # Progress every 10% of epochs
        if epoch % max(1, args.epochs // 10) == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}/{args.epochs}  "
                f"loss={avg_total:.6f}  data={avg_data:.6f}  "
                f"phys={avg_phys:.6f}  val_mae={val_mae_mm:.2f} mm"
            )

    # Save best or final model
    model_path = MODELS_DIR / "pinn_localization.pt"
    if best_state is not None:
        torch.save(best_state, model_path)
        model.load_state_dict(best_state)
        print(f"\n[INFO] Best checkpoint restored (val_mae={best_val_mae:.2f} mm)")
    else:
        torch.save(model.state_dict(), model_path)
    print(f"     Saved model: {model_path}")

    return (model, history_rows, X_test, test_idx,
            xy_true, scenarios, torques, denormalize_xy)


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def _save_history(history_rows, out_dir: Path) -> Path:
    """Write training_history.csv."""
    out_path = out_dir / "training_history.csv"
    lines = ["epoch,loss_total,loss_data,loss_physics,val_mae_mm"]
    for epoch, lt, ld, lp, vm in history_rows:
        lines.append(f"{epoch},{lt:.8f},{ld:.8f},{lp:.8f},{vm:.4f}")
    out_path.write_text("\n".join(lines) + "\n")
    print(f"     Written: {out_path}")
    return out_path


def _save_results(model, X_test_t, test_idx, xy_true,
                  scenarios, torques, denormalize_xy, out_dir: Path):
    """Write pinn_localization_results.csv with per-sample predictions."""
    torch, _, _ = _require_torch()

    model.eval()
    with torch.no_grad():
        pred_norm = model(X_test_t).numpy()
    pred_m = denormalize_xy(pred_norm)
    true_m = xy_true[test_idx]
    errs_mm = np.sqrt(((pred_m - true_m) ** 2).sum(axis=1)) * 1000.0

    out_path = out_dir / "pinn_localization_results.csv"
    lines = ["source_x,source_y,pred_x,pred_y,error_mm,scenario,torque_loss_pct"]
    for i, gi in enumerate(test_idx):
        lines.append(
            f"{true_m[i,0]:.6f},{true_m[i,1]:.6f},"
            f"{pred_m[i,0]:.6f},{pred_m[i,1]:.6f},"
            f"{errs_mm[i]:.4f},"
            f"{scenarios[gi]},"
            f"{torques[gi]:.1f}"
        )
    out_path.write_text("\n".join(lines) + "\n")
    print(f"     Written: {out_path}")
    return out_path, errs_mm


def _mae_by_scenario(errs_mm, test_idx, scenarios) -> dict:
    """Compute mean absolute error (mm) per scenario on the test set."""
    sc_errors = {}
    for i, gi in enumerate(test_idx):
        sc = scenarios[gi]
        sc_errors.setdefault(sc, []).append(errs_mm[i])
    return {sc: float(np.mean(v)) for sc, v in sorted(sc_errors.items())}


def _print_table1(mae_by_sc: dict, global_mae: float, lambda_phys: float):
    """Print Table 1 reproduction in the format specified by the paper."""
    sc_display = {
        "intact":     "intact",
        "loose_25":   "loose_25",
        "loose_50":   "loose_50",
        "full_loose":  "full_loose",
    }

    print()
    print("=" * 45)
    print("  TABLE 1 REPRODUCTION")
    print(f"  Scenario        MAE (mm)  [λ={lambda_phys}]")
    print("-" * 45)
    for sc_key, sc_label in sc_display.items():
        if sc_key in mae_by_sc:
            print(f"  {sc_label:<16}{mae_by_sc[sc_key]:.2f}")
    print("-" * 45)
    print(f"  {'global':<16}{global_mae:.2f}")
    print("=" * 45)
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train wave-equation constrained PINN for AE source localization "
            "(standalone reproducibility script — reproduces Table 1)."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--lambda-phys", "--lambda",
        type=float,
        default=LAMBDA_PHYS,
        dest="lambda_phys",
        help=f"Physics loss weight λ (default: {LAMBDA_PHYS})",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Adam learning rate (default: {LR})",
    )
    parser.add_argument(
        "--hidden",
        type=int,
        default=HIDDEN_SIZE,
        help=f"Hidden layer width (default: {HIDDEN_SIZE})",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=N_LAYERS,
        help=f"Number of hidden layers (default: {N_LAYERS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random seed (default: {SEED})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        dest="batch_size",
        help=f"Mini-batch size (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        dest="data_dir",
        help=f"Data directory (default: {DATA_DIR})",
    )
    args = parser.parse_args()

    print()
    print("=" * 60)
    print(" PINN Source Localization — 300×300 mm Bolted Steel Plate")
    print("=" * 60)
    print()

    (model, history_rows, X_test_t, test_idx,
     xy_true, scenarios, torques, denormalize_xy) = train(args)

    print("\n[OUTPUTS]")
    _save_history(history_rows, Path(args.data_dir))
    results_path, errs_mm = _save_results(
        model, X_test_t, test_idx, xy_true,
        scenarios, torques, denormalize_xy,
        Path(args.data_dir),
    )

    mae_by_sc    = _mae_by_scenario(errs_mm, test_idx, scenarios)
    global_mae   = float(errs_mm.mean())

    _print_table1(mae_by_sc, global_mae, args.lambda_phys)

    print(f"[OK] Training complete — {args.epochs} epochs, λ={args.lambda_phys}")
    print(f"     Model saved: {MODELS_DIR / 'pinn_localization.pt'}")
    print(f"     Results  : {results_path}")

    return mae_by_sc, global_mae


if __name__ == "__main__":
    main()
