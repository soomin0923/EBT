# EBT: EEG Branch Transformer

A Branch Transformer architecture for cross-subject EEG classification,  
targeting Fatigue and Mental Workload detection across three benchmark datasets.

Designed as a challenger to **EEG-Deformer** (IEEE J-BHI 2024).

---

## Results (LOSO Cross-Subject Classification)

### vs. EEG-Deformer

| Dataset | Metric | EEG-Deformer | EBT (ours) | Δ |
|---|---|---|---|---|
| Driving EEG | ACC | 77.22 | 73.66 | -3.56 |
| Driving EEG | F1-macro | 74.13 | 73.53 | -0.60 |
| **EEGMAT** | **ACC** | **70.07** | **77.68** | **+7.61** |
| **EEGMAT** | **F1-macro** | **67.03** | **74.92** | **+7.89** |
| **STEW** | **ACC** | **75.40** | **77.72** | **+2.40** |
| **STEW** | **F1-macro** | **74.10** | **77.20** | **+3.73** |

EBT outperforms EEG-Deformer on 2 out of 3 datasets (EEGMAT, STEW).  
EEGMAT ACC improvement is statistically significant (Wilcoxon p=0.04).

---

## Ablation Study

| Branch Config | Driving ACC | EEGMAT ACC | STEW ACC |
|---|---|---|---|
| Full branch | 73.66 | **77.68** | 77.72 |
| No time branch | 65.28 | 71.65 | 76.27 |
| No time-frequency branch | 69.40 | 68.44 | 72.51 |
| No spatial branch | 74.64 | 74.81 | **78.53** |

---

## Architecture

EBT processes EEG signals through three parallel branches:
- **Time branch** — temporal dynamics
- **Time-frequency branch** — spectral features
- **Spatial branch** — cross-channel relationships

---

## Tech Stack
Python, PyTorch, NumPy

## File Structure
- `model_components.py` — main EBT architecture
- `model_components_ablation.py` — ablation variants
- `dataset.py` — EEG data loading & preprocessing
- `augmentations.py` — data augmentation
- `losses.py` — custom loss functions
- `train_utils_ablation.py` — training utilities
