# Wave-Equation Constrained PINN — Reproducibility Package

Companion code for the paper:

> **"Wave-Equation Constrained PINN for Acoustic Emission Source Localization in Bolted
> Connections: A Cyber-Physical Digital Twin Framework with ifcJSON Middleware"**
> SPIE Smart Structures and NDE 2027 — *under review*

This repository reproduces **Table 1** (MAE by torque-loss scenario) and the ifcJSON export
(Section 2.4) using three self-contained scripts and no external dependencies beyond PyTorch.

---

## Quickstart

```bash
git clone https://github.com/Mikisbell/pinn-bolted-reproducibility.git
cd pinn-bolted-reproducibility
pip install -r requirements.txt
python train_pinn.py          # reproduces Table 1 — ~5 min on CPU
```

Expected output:

```
=== TABLE 1 REPRODUCTION ===
Scenario        MAE (mm)  [λ=0.1]
intact          5.73
loose_25        7.46
loose_50        9.30
full_loose      12.44
global          8.31
============================
```

Exact values may vary ±0.1 mm depending on random seed and hardware; reference global MAE = 8.31 mm.

---

## Repository Structure

```
pinn-bolted-reproducibility/
├── generate_ae_data.py          # Step 1 — synthetic AE arrivals (Section 2.1)
├── train_pinn.py                # Step 2 — PINN training → reproduces Table 1 (Section 2.2–2.3)
├── export_ifc.py                # Step 3 — ifcJSON export (Section 2.4)
├── requirements.txt
├── data/
│   └── processed/
│       └── ae_synthetic_arrivals.csv   # pre-generated (400 samples, 4 scenarios)
└── models/
    └── pinn_localization.pt            # pre-trained weights (λ=0.1, 500 epochs)
```

---

## Step-by-Step Reproduction

### Step 1 — Generate synthetic data (optional — pre-generated data included)

```bash
python generate_ae_data.py
# Writes: data/processed/ae_synthetic_arrivals.csv
# 400 rows × 8 cols: t1..t6 [µs], source_x [m], source_y [m]
# 4 scenarios × 100 samples: intact / loose_25 / loose_50 / full_loose
```

Physical parameters (Section 2.1):
- Plate: 300 × 300 mm steel
- Bolt position: (0.15, 0.15) m
- 6 sensors at plate perimeter corners and edge midpoints
- Wave speed (Lamb S0 mode): 5000 m/s
- Noise σ: 1 µs (intact) → 3 µs (full_loose, 100% torque loss)

### Step 2 — Train the PINN (reproduces Table 1)

```bash
python train_pinn.py
# Reads:  data/processed/ae_synthetic_arrivals.csv
# Writes: models/pinn_localization.pt
#         data/processed/pinn_localization_results.csv
# Prints: Table 1 reproduction
```

Architecture (Section 2.2):
- MLP: 6 inputs → [64 × 4] hidden → 2 outputs (x, y normalized)
- Activation: Tanh (hidden) + Sigmoid (output)
- Loss: L = L_data + λ · L_physics
  - L_physics = MSE(t̂_i, t_measured), where t̂_i = dist(source_predicted, sensor_i) / c
  - Loss normalization by initial values (Wu et al. 2023) prevents physics term from dominating

To reproduce the ablation (Table 2):

```bash
python train_pinn.py --lambda-phys 0.0   # baseline MLP, no physics
python train_pinn.py --lambda-phys 0.1   # wave-equation constrained (default)
```

### Step 3 — Export to ifcJSON (Section 2.4)

```bash
python export_ifc.py
# Reads:  data/processed/pinn_localization_results.csv
# Writes: data/processed/ifc_export_sample.json
# Schema: IfcStructuralPointAction (ORNL ifcJSON, Barbosa et al. 2023)
```

---

## Key Results (Table 1)

| Scenario | Torque loss | λ=0 (baseline) | λ=0.1 (wave-eq PINN) |
|----------|-------------|----------------|----------------------|
| intact | 0% | 4.94 mm | 5.73 mm |
| loose_25 | 25% | 6.83 mm | 7.46 mm |
| loose_50 | 50% | 8.73 mm | 9.30 mm |
| **full_loose** | **100%** | 12.83 mm | **12.44 mm ✓** |
| global | — | 8.33 mm | 8.31 mm |

The physics constraint selectively reduces error in the highest-noise scenario
(full_loose: −0.39 mm, −3.0%) while maintaining comparable global MAE (8.31 vs 8.33 mm),
demonstrating noise-adaptive PDE regularization.

---

## Dependencies

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
```

Python 3.9+ recommended. No GPU required (CPU training: ~5 min for 500 epochs).

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{autor2027pinn,
  title     = {Wave-Equation Constrained PINN for Acoustic Emission Source Localization
               in Bolted Connections: A Cyber-Physical Digital Twin Framework
               with ifcJSON Middleware},
  author    = {[Author names — to be updated upon acceptance]},
  booktitle = {Proc. SPIE Smart Structures and NDE 2027},
  year      = {2027},
  note      = {DOI to be assigned upon publication}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
