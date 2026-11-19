# Consolidated v2 Verification Report (HEFMI-ICH, n=37)

## 1. Ground-Truth Noise Ceiling & Experiment 1 (Two-Regime EEG -> HbO Reconstruction)
- **GT Split-Half Reliability Ceiling**: Raw `0-1s BC` = `+0.1720` (`[+0.0256, +0.3090]`), `CAR` = `+0.2054`, `8-vs-22 cal/te` = `+0.0364`.

### Regime A: Zero-Shot 37-Fold LOSO (0 calibration trials)
| Model | All-90 Signed r | Top-25% Signed r | Top-25% |r| | Evoked RMSE (uM) | Peak Latency Err (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Ridge-FIR (Zero-Shot) | +0.0009 | +0.0109 | 0.3315 ± 0.1361 | 0.642 | 1.50 |
| BlackBox-TCN (Zero-Shot) | +0.0111 | +0.0025 | 0.3585 ± 0.0690 | 0.650 | 2.79 |
| Canonical-HRF (Zero-Shot) | +0.0334 | +0.0409 | 0.6426 ± 0.0942 | 0.642 | 1.52 |
| EB-NVC-Net w/o Graph (Zero-Shot) | +0.0396 | +0.0392 | 0.5000 ± 0.1006 | 0.647 | 2.01 |
| EB-NVC-Net (Zero-Shot) | +0.0400 | +0.0399 | 0.5138 ± 0.1019 | 0.646 | 2.00 |

### Regime B: Matched 8-Trial Local Calibration (cal=0..7 -> Held-Out te=8..29; Top-25% Selected Strictly on cal)
| Model (Matched 8-Trial Budget) | Held-Out Top-25% Signed r | All-90 Signed r | Held-Out Top-25% |r| | Task-Evoked Signed r | Single-Trial Signed r | Held-Out RMSE (uM) | Lat Err (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Ridge-FIR (8-Trial Cal) | +0.0031 ± 0.1649 | +0.0351 | 0.3487 ± 0.1343 | +0.0202 | +0.0011 | 0.619 | 1.82 |
| BlackBox-TCN (8-Trial Cal) | -0.0193 ± 0.1237 | -0.0072 | 0.3717 ± 0.0828 | +0.0230 | +0.0012 | 0.617 | 2.57 |
| Canonical-HRF (K=1, 8-Trial Cal) | +0.0103 ± 0.2054 | +0.0199 | 0.6262 ± 0.0840 | -0.0196 | -0.0023 | 0.696 | 1.61 |
| Local Multi-Basis Ridge (K=5, 8-Trial Cal) | +0.0200 ± 0.2275 | +0.0344 | 0.6685 ± 0.0823 | -0.0064 | -0.0007 | 0.873 | 1.61 |
| EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal) | +0.0213 ± 0.2173 | +0.0394 | 0.6383 ± 0.0771 | +0.0027 | +0.0011 | 0.668 | 1.65 |
- **EB-NVC vs Canon (Regime B Held-Out `te=8..29`)**: Signed r diff `+0.0110` ($p=0.7542$); `|r|` diff `+0.0121` ($p=4.4613e-01$); RMSE diff `-0.0286` ($p=0.0173$).

## 2. Experiment 2: Nested LOSO Screening & BH-FDR Exploratory Clinical Associations
- **Strict Outer-Fold Closed-Form NVC (Zero Stacking)**: AUC = `0.824` (95% CI `[0.665, 0.956]`), Nested Acc = `81.1%` (Sens `70.0%`, Spec `94.1%`, BalAcc `82.1%`), Paired ΔAUC vs EEG-Only boot $p = 0.0890$.
- **Nested 37-fold LOSO NVC Biomarkers (Blind EB-NVC-Net)**: AUC = `0.685` (95% CI `[0.503, 0.850]`), Nested Acc = `64.9%` (Sens `55.0%`, Spec `76.5%`, BalAcc `65.7%`).
- **Nested LOSO EEG-Only**: AUC = `0.579` (95% CI `[0.388, 0.765]`), Paired ΔAUC boot $p = 0.3580$.

## 3. Experiment 3: Strictly In-Fold 6-Step MI Decoding & Placebo Controls (HEFMI-ICH)
| Step / Method | 5-Fold In-Fold CV (All n=37) | 95% Bootstrap CI | Block Split (0..14 -> 15..29) |
| :--- | :---: | :---: | :---: |
| 1_EEG_Raw | 54.50% ± 11.96% | [50.72%, 58.20%] | 52.25% |
| 2_InFold_Shrinkage_EA | 54.95% ± 13.07% | [50.81%, 59.19%] | 56.22% |
| 3_Canon_HRF_K1 | 56.04% ± 13.31% | [51.71%, 60.18%] | 55.14% |
| 4_EB_MultiBasis_K5 | 56.94% ± 13.57% | [52.70%, 61.17%] | 56.76% |
| 5_Add_FastSlow_Cov | 57.66% ± 13.24% | [53.51%, 62.16%] | 56.94% |
| 6_Full_FastSlow_EntropyGated | 57.75% ± 13.31% | [53.51%, 62.16%] | 57.48% |
| Oracle_Real_EEG_HbO | 61.71% ± 13.35% | [57.48%, 66.04%] | 55.50% |
- **200-Run Label-Permutation Test**: Null Full = `45.10% ± 1.87%`, $p_{perm} = 0.0050$.

## 4. Experiment 4: External Pure-EEG Cross-Stroke Validation (Liu2024, n=47) & Stratified Analyses
| Step / Cohort | 10-Fold In-Fold CV | 95% Bootstrap CI | Block Split (0..19 -> 20..39) |
| :--- | :---: | :---: | :---: |
| Liu2024: 1_EEG_Raw_CSP | 56.01% ± 10.71% | [53.03%, 59.10%] | 51.38% |
| Liu2024: 2_InFold_Shrinkage_EA | 55.37% ± 11.19% | [52.23%, 58.78%] | 52.13% |
| Liu2024: 3_Add_Canon_HRF_K1 | 54.04% ± 10.51% | [50.96%, 57.29%] | 51.06% |
| Liu2024: 4_Add_MultiBasis_HRF_K5 | 55.85% ± 13.32% | [52.23%, 59.79%] | 53.30% |
| Liu2024: 5_Add_Joint_FastSlow_sDGF | 56.86% ± 12.73% | [53.35%, 60.59%] | 53.30% |
| Liu2024: 6_Full_FastSlow_EntropyGated | 57.18% ± 12.41% | [53.72%, 60.80%] | 52.87% |
- **Liu2024 Paired Stats**: Full vs EEG-Raw = `+1.17 pp` (p=`0.6521`, `21` improved, `5` unchanged, `21` decreased); Full vs Canon = `+3.14 pp` (p=`0.0567`).
- **200-Run Permutation Test (Liu2024)**: Null Full = `47.16% ± 1.51%`, $p_{perm} = 0.0050$.
- **Pooled Stroke Patients Only (N=67)**: EEG-Raw `55.51%`, Canon `55.02%`, FastSlow-NVC `57.28%` (gain `+1.77 pp`, 95% strat CI `[-1.23, +4.88]`, Wilcoxon p=`0.3365`; vs Canon p=`0.0711`).
- **Stratified All Participants (N=84)**: EEG-Raw `55.35%`, Canon `54.92%`, FastSlow-NVC `57.43%` (gain `+2.08 pp`, 95% strat CI `[-0.55, +4.72]`, Wilcoxon p=`0.1513`; Method x Cohort interaction p=`0.3624`).
