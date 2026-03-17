"""
export_ifc.py — PINN Localization Results → ifcJSON Export (standalone)
========================================================================
Reproduce the ifcJSON export described in Section 2.4 of "Wave-Equation
Constrained PINN for Acoustic Emission Source Localization in Bolted
Connections: A Cyber-Physical Digital Twin Framework with ifcJSON Middleware".

Standalone script: no dependencies on belico-stack, config/params.yaml, or any
factory infrastructure. No ifcopenshell required — the script writes valid
ifcJSON directly using Python's built-in json module.

Input
-----
data/processed/pinn_localization_results.csv  (relative to this script)
  Columns: source_x, source_y, pred_x, pred_y, error_mm, scenario, torque_loss_pct

  If the file does not exist (e.g. Table 1 has not been reproduced yet),
  the script falls back to hardcoded Table 1 data so the ifcJSON schema
  can be inspected without running train_pinn.py first.

Output
------
data/processed/ifc_export_sample.json  (relative to this script)

Schema: IfcStructuralPointAction (IFC4 subset)
Each entry encodes one AE source localisation event with:
  - Predicted source coordinates (ae_source_x_m, ae_source_y_m)
  - Localisation error from PINN output (localization_error_mm)
  - Damage state metadata (scenario, torque_loss_pct, ground_truth coords)

Usage
-----
  # After running train_pinn.py (uses real results):
  python export_ifc.py

  # Without running train_pinn.py (uses hardcoded Table 1 fallback):
  python export_ifc.py

  # Custom paths:
  python export_ifc.py --input path/to/results.csv --output path/to/out.json
"""

import argparse
import csv
import json
import sys
import uuid
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — relative to this script (standalone)
# ---------------------------------------------------------------------------
SCRIPT_DIR     = Path(__file__).resolve().parent
DEFAULT_INPUT  = SCRIPT_DIR / "data" / "processed" / "pinn_localization_results.csv"
DEFAULT_OUTPUT = SCRIPT_DIR / "data" / "processed" / "ifc_export_sample.json"

# ---------------------------------------------------------------------------
# Known scenarios (used for summary ordering)
# ---------------------------------------------------------------------------
SCENARIOS = ["intact", "loose_25", "loose_50", "full_loose"]

# ---------------------------------------------------------------------------
# Hardcoded Table 1 fallback data
# These values are the representative MAE results from Table 1 of the paper.
# Used when pinn_localization_results.csv has not been generated yet so that
# the ifcJSON schema can be inspected independently of the training pipeline.
# Each tuple: (source_x, source_y, pred_x, pred_y, error_mm, scenario, torque_loss_pct)
# ---------------------------------------------------------------------------
_TABLE1_FALLBACK = [
    # intact — ~uniform plate coverage
    (0.120, 0.080, 0.123, 0.082, 3.61, "intact",     0.0),
    (0.200, 0.250, 0.198, 0.247, 3.61, "intact",     0.0),
    (0.070, 0.190, 0.073, 0.192, 3.61, "intact",     0.0),
    # loose_25 — events cluster near bolt centre
    (0.148, 0.152, 0.151, 0.148, 4.83, "loose_25",  25.0),
    (0.160, 0.145, 0.163, 0.142, 4.83, "loose_25",  25.0),
    (0.138, 0.158, 0.141, 0.155, 4.83, "loose_25",  25.0),
    # loose_50 — tighter cluster, higher noise
    (0.152, 0.148, 0.156, 0.144, 6.12, "loose_50",  50.0),
    (0.145, 0.155, 0.150, 0.150, 6.12, "loose_50",  50.0),
    (0.158, 0.142, 0.163, 0.137, 6.12, "loose_50",  50.0),
    # full_loose — near-bolt, highest noise
    (0.151, 0.149, 0.158, 0.143, 8.44, "full_loose", 100.0),
    (0.147, 0.153, 0.155, 0.146, 8.44, "full_loose", 100.0),
    (0.153, 0.147, 0.161, 0.140, 8.44, "full_loose", 100.0),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_csv(input_path: Path) -> list[dict]:
    """Load pinn_localization_results.csv and return list of row dicts."""
    rows = []
    with open(input_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _load_from_fallback() -> list[dict]:
    """Return hardcoded Table 1 data as list of row dicts (same schema as CSV)."""
    rows = []
    for (sx, sy, px, py, err, sc, tq) in _TABLE1_FALLBACK:
        rows.append({
            "source_x":        str(sx),
            "source_y":        str(sy),
            "pred_x":          str(px),
            "pred_y":          str(py),
            "error_mm":        str(err),
            "scenario":        sc,
            "torque_loss_pct": str(tq),
        })
    return rows


def load_results(input_path: Path) -> tuple[list[dict], bool]:
    """Load localization results. Falls back to Table 1 data if file absent.

    Returns
    -------
    rows        : list of row dicts
    used_fallback : True if Table 1 hardcoded data was used
    """
    if input_path.exists():
        rows = _load_from_csv(input_path)
        print(f"[export_ifc] Loaded {len(rows)} rows from {input_path}")
        return rows, False
    else:
        print(
            f"[export_ifc] {input_path} not found — using hardcoded Table 1 fallback.",
            file=sys.stderr,
        )
        print(
            "             Run train_pinn.py to generate real results.",
            file=sys.stderr,
        )
        rows = _load_from_fallback()
        print(f"[export_ifc] Using {len(rows)} hardcoded Table 1 samples.")
        return rows, True


# ---------------------------------------------------------------------------
# ifcJSON construction
# ---------------------------------------------------------------------------

def _build_ifc_entry(i: int, row: dict) -> dict:
    """Map one result row to an IfcStructuralPointAction entry.

    Schema follows IFC4 IfcStructuralPointAction with custom AE extensions:
      - appliedLoad.ae_source_{x,y}_m     : PINN-predicted source location
      - appliedLoad.localization_error_mm : Euclidean error vs ground truth
      - damage_state                      : scenario metadata
    """
    scenario    = row["scenario"]
    torque_loss = float(row["torque_loss_pct"])
    pred_x      = float(row["pred_x"])
    pred_y      = float(row["pred_y"])
    source_x    = float(row["source_x"])
    source_y    = float(row["source_y"])
    error_mm    = float(row["error_mm"])

    return {
        "type":        "IfcStructuralPointAction",
        "globalId":    str(uuid.uuid4()),
        "name":        f"AE_Source_{i:04d}",
        "description": f"scenario={scenario}, torque_loss={torque_loss:.1f}%",
        "appliedLoad": {
            "type":                    "IfcStructuralLoad",
            "name":                    "AELoad",
            "ae_source_x_m":           round(pred_x,   6),
            "ae_source_y_m":           round(pred_y,   6),
            "localization_error_mm":   round(error_mm, 4),
        },
        "damage_state": {
            "scenario":          scenario,
            "torque_loss_pct":   torque_loss,
            "ground_truth_x_m":  round(source_x, 6),
            "ground_truth_y_m":  round(source_y, 6),
        },
    }


def _compute_summary(rows: list[dict]) -> dict:
    """Compute global MAE and per-scenario MAE from results rows."""
    errors     = [float(r["error_mm"]) for r in rows]
    global_mae = sum(errors) / len(errors) if errors else 0.0

    mae_by_scenario: dict = {}
    for sc in SCENARIOS:
        sc_errs = [float(r["error_mm"]) for r in rows if r["scenario"] == sc]
        mae_by_scenario[sc] = round(sum(sc_errs) / len(sc_errs), 4) \
                              if sc_errs else None

    # Include any extra scenario present in data but not in the predefined list
    for sc in sorted({r["scenario"] for r in rows}):
        if sc not in mae_by_scenario:
            sc_errs = [float(r["error_mm"]) for r in rows if r["scenario"] == sc]
            mae_by_scenario[sc] = round(sum(sc_errs) / len(sc_errs), 4)

    return {
        "total_events":               len(rows),
        "scenarios":                  sorted({r["scenario"] for r in rows}),
        "mean_localization_error_mm": round(global_mae, 4),
        "mae_by_scenario":            mae_by_scenario,
    }


def export(input_path: Path, output_path: Path) -> tuple[dict, bool]:
    """Build and write the ifcJSON document.

    Returns
    -------
    summary       : summary statistics dict
    used_fallback : True if hardcoded Table 1 data was used
    """
    rows, used_fallback = load_results(input_path)

    data_entries = [_build_ifc_entry(i, row) for i, row in enumerate(rows)]
    summary      = _compute_summary(rows)

    ifc_doc = {
        "type":             "ifcJSON",
        "version":          "0.0.1",
        "schemaIdentifier": "IFC4",
        "description": (
            "AE source localisation results — Wave-Equation Constrained PINN "
            "for Bolted Joint SHM (reproducibility export)"
        ),
        "source": "hardcoded_table1_fallback" if used_fallback else str(input_path),
        "data":    data_entries,
        "summary": summary,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(ifc_doc, fh, indent=2)

    return summary, used_fallback


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Entry point. Returns 0 on success, 1 on error."""
    parser = argparse.ArgumentParser(
        description=(
            "Export PINN AE localisation results to ifcJSON "
            "(standalone reproducibility script — Section 2.4)."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input CSV path (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    try:
        summary, used_fallback = export(args.input, args.output)
    except OSError as exc:
        print(f"[export_ifc] ERROR: {exc}", file=sys.stderr)
        return 1

    source_label = "(Table 1 fallback)" if used_fallback else "(real results)"
    print(f"Exported {summary['total_events']} events {source_label} → {args.output}")
    print(f"  mean_localization_error_mm : {summary['mean_localization_error_mm']}")
    print(f"  mae_by_scenario            : {summary['mae_by_scenario']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
