# Patient-Independent ECG Arrhythmia Classification

**Diagnosing and Fixing Data Leakage in Deep Learning–Based Arrhythmia Detection**

A controlled reproduction study on the MIT-BIH Arrhythmia Database, using patient-wise splitting, class-weighted loss, and SHAP explainability.

---

## Overview

Recent ECG arrhythmia papers (2024–2026) report 91–99% accuracy but rarely confirm patient-independent test sets. Random/beat-wise splitting can let a patient's beats leak across train and test, letting the model partially memorize patients instead of learning arrhythmias. This project trains one identical CNN under two protocols — a random/beat-wise split (matching common practice) and the standard MIT-BIH DS1/DS2 patient-wise split — to measure how much reported accuracy is real.

---

## Dataset

- **Source:** [MIT-BIH Arrhythmia Database](https://physionet.org/content/mitdb/), 360 Hz
- **DS1 (train):** 101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230
- **DS2 (test):** 100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234
- **Classes:** Normal, BBB (L/R), APB, PVC — Paced excluded (its records fall outside DS1/DS2)

---

## Methodology

1. Bandpass filter (0.5–40 Hz) → R-peak-centered beat extraction (250 samples) → per-beat z-score normalization
2. Fixed 1D CNN: two Conv1D+MaxPooling1D blocks (32, 64 filters), Flatten, Dense(64), Dropout(0.3), softmax
3. **Baseline:** random/beat-wise 80/20 split
4. **Corrected:** patient-wise DS1/DS2 split (zero patient overlap, verified), class-weighted loss
5. Evaluate both: accuracy, macro-F1, per-class recall, confusion matrix
6. SHAP applied post-hoc on the corrected model's misclassified minority-class beats

Architecture, preprocessing, and hyperparameters are identical across both runs — only the split and loss weighting change.

---

## Results

| Metric | Baseline (beat-wise) | Corrected (patient-wise) |
|---|---|---|
| Accuracy | 99% | 66% |
| Macro-F1 | 0.97 | 0.36 |
| Normal — Recall | 1.00 | 0.80 |
| BBB — Recall | 1.00 | 0.00 |
| APB — Recall | 0.84 | 0.12 |
| PVC — Recall | 0.97 | 0.91 |

Accuracy drops sharply once leakage is removed — confirming most of the baseline's reported performance reflected patient memorization, not arrhythmia recognition. The collapse is uneven: BBB and APB recall fall to near zero while PVC holds up, suggesting those minority classes were concentrated in a few patients under the original split.

---

## Setup

```bash
pip install wfdb numpy scipy scikit-learn tensorflow shap matplotlib
```
```python
import wfdb
wfdb.dl_database('mitdb', dl_dir='data')
```
Run `preprocess.py` to generate `X.npy`, `y.npy`, `patient_ids.npy`, then proceed to training.

---

## Scope

**In scope:** one fixed CNN, beat-wise vs. patient-wise comparison, class-weighted loss, SHAP explainability.
**Out of scope:** architectural innovation, hyperparameter search, comparison against attention/transformer models — the CNN is deliberately kept simple so the evaluation protocol, not the model, is the isolated variable.

---

## Conclusion

A simple CNN reproduces the inflated ~98–99% accuracy common in recent ECG arrhythmia literature under a beat-wise split. The standard MIT-BIH patient-wise split reveals substantially lower, more honest performance, concentrated in minority-class degradation. Class-weighted loss partially recovers this, and SHAP identifies which waveform regions still drive the corrected model's errors. This argues for patient-wise evaluation as standard practice in future ECG deep learning research.
