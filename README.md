# Empirical-Bayes Neurovascular Dynamics & Fast-Slow Envelope Manifold Analysis (N = 84)

Reference implementation and reproducible evaluation pipeline for:
**"Empirical-Bayes Neurovascular Coupling Inversion and Fast-Slow Envelope Manifold Analysis Across Hemorrhagic EEG-fNIRS and Pure-EEG Ischemic Stroke Cohorts"**

## Repository Structure

```text
.
├── data_hefmi.py                  # HEFMI-ICH (n=37) data loader & in-fold EEG/u preprocessing
├── models/
│   ├── __init__.py
│   └── physio_nvc_ssm.py          # Mode-exact Gamma HRF bank, homologous graph modulator & EB-NVC-Net
├── train_eval_route_a.py          # Stage 1: 37-fold blind LOSO PyTorch training -> predictions & biomarkers
├── finalize_route_a.py            # Stage 2: HEFMI-ICH audit, nested LOSO, BH-FDR, 200-perm & Fig. 2-4
├── eval_cross_stroke_84.py        # Stage 3: Liu2024 (n=47) + N=84 stratified cohorts & Fig. 5
├── publication_figures.py         # Publication vector figure generator (manuscript Figs. 2-5; ABME)
└── results/
    ├── route_a_metrics.json       # Complete quantitative results & 95% CIs across all experiments
    ├── fold_manifest.json         # Deterministic CV & block-split indices for reproducibility
    ├── subject_nvc_biomarkers.csv # Subject-level HEFMI-ICH (n=37) NVC & EEG descriptors
    ├── clinical_ich_fdr.csv       # Partial Spearman & BH-FDR clinical associations (ICH n=20 / n=19)
    ├── liu2024_nvc_biomarkers.csv # Subject-level Liu2024 (n=47) pure-EEG envelope latency & decoding
    ├── route_a_report.md          # Automatically generated quantitative markdown summary
    └── figures/
        ├── figgenie_monochrome/   # Study workflow diagram (Fig. 1)
        └── publication/           # Publication vector figures (PDF/SVG/PNG/grayscale)
```

## Public Datasets

Both datasets analyzed in this study are publicly available and de-identified by the original authors:
- **HEFMI-ICH ($n = 37$)**: Shi et al., *Scientific Data* 12:1816 (2025), DOI: [`10.1038/s41597-025-06100-7`](https://doi.org/10.1038/s41597-025-06100-7).
- **Liu2024 ($n = 47$ unique patients)**: Liu et al., *Scientific Data* 11:131 (2024), DOI: [`10.1038/s41597-023-02787-8`](https://doi.org/10.1038/s41597-023-02787-8); duplicate files (`sub-5`, `sub-15`, `sub-49`) excluded *a priori* per NEMAR notice `nm000158`.

## Methodological & Anti-Leakage Guarantees

1. **Mode-Exact Multi-Basis Gamma HRF Bank (`models/physio_nvc_ssm.py`)**:
   - Five causal Gamma bases parameterized by exact peak modes $\tau_k \in \{2.00, 3.25, 4.50, 5.75, 7.00\}\,\mathrm{s}$ and dispersion $\sigma = 1.1\,\mathrm{s}$ ($b_k = \frac{-\tau_k + \sqrt{\tau_k^2 + 4\sigma^2}}{2}$, $a_k = 1 + \frac{\tau_k}{b_k}$), normalized to unit sum.
2. **In-Fold Conditioning & Preprocessing (`data_hefmi.py`, `finalize_route_a.py`, `eval_cross_stroke_84.py`)**:
   - Per-trial $0\text{--}1\,\mathrm{s}$ optical baseline subtraction eliminates inter-trial tail spillover on `HEFMI-ICH`.
   - Bad-channel repair ($\sigma_c > 5.0 \cdot \mathrm{median}_j \sigma_j$) and MAD clipping ($[-8, 8]$) on `HEFMI-ICH`, trace-normalized SPD covariances on `Liu2024`, shrinkage Euclidean Alignment ($\bar{\mathbf{R}}_{\mathrm{reg}} = 0.85\bar{\mathbf{R}}_{\mathrm{tr}} + 0.15\frac{\mathrm{tr}(\bar{\mathbf{R}}_{\mathrm{tr}})}{C}\mathbf{I}$), CSP filters, ANOVA `SelectKBest`, scalers, and shrinkage LDA/logistic classifiers are fitted strictly inside each training or calibration fold $\mathcal{I}_{\mathrm{tr}}$.
3. **Explicit Path Separation & Outer-Fold Screening**:
   - Separates blind cross-subject LOSO (`EB-NVC-Net`), matched 8-trial damped Empirical-Bayes calibration (`EB-NVC`, evaluated on held-out trials $8\text{--}29$ with calibration-only top-25% channel selection), and in-fold optical-free `FastSlow-NVC` decoding ($N = 84$).
   - Clinical screening reports combined NVC features including measured HbO ($\mathrm{AUC} = 0.685$), HbO-only screening ($\mathrm{AUC} = 0.674$), and outer-fold closed-form NVC screening with zero Stage-1 cross-subject stacking ($\mathrm{AUC} = 0.824$, nested accuracy $81.1\%$).

## Reproduction Commands

### 1. Full 3-Stage Pipeline (from cached datasets)
> Note: Running Stage 1 generates `results/route_a_predictions.npz` (~188 MB, excluded from git via `.gitignore`).

```bash
# Stage 1: 37-fold blind LOSO training & OOF predictions
python train_eval_route_a.py

# Stage 2: Consolidated HEFMI-ICH audit, 200-run permutation test & Fig. 2-4
python finalize_route_a.py

# Stage 3: External Liu2024 (n=47) validation, N=84 stratified cohorts & Fig. 5
python eval_cross_stroke_84.py
```

### 2. Fast Figure Regeneration (from saved metrics & CSVs)
```bash
python finalize_route_a.py --figures-only
python eval_cross_stroke_84.py --figures-only
```
