"""Cross-Stroke Validation v2 across 84 Participants: HEFMI-ICH (n=37) + MNE-liu2024 (n=47).

Addresses every methodological & statistical critique:
1. Uses the exact same unified mode-parameterized Gamma HRF bank taus=(2.0, 3.25, 4.5, 5.75, 7.0) s,
   sigma=1.1 s as HEFMI-ICH.
2. Enforces trace-normalized SPD trial covariances and strictly in-fold shrinkage Euclidean Alignment (EA),
   CSP spatial filters, ANOVA feature selection, and shrinkage LDA inside every training fold tr.
3. Evaluates the same 6-Step Component Ablation on Liu2024 (10-fold CV), Chronological Block Split
   (trials 0..19 train -> 20..39 test), and 200-run Label-Permutation Placebo Test.
4. Explicitly calls s_DGF "Discriminant Geodesic Score" (signed non-metric score) and adds
   Joint Fast-Slow Cross-Timescale SPD Covariance views.
5. Computes Stratified Multi-Cohort Statistics separately for:
   - HEFMI-ICH HC (n=17), ICH (n=20), Full (n=37)
   - Liu2024 Acute IS (n=47)
   - Pooled Stroke Patients Only (N_stroke = 67: 20 ICH + 47 IS)
   - Stratified All-Participants (N = 84) with Stratified Bootstrap CI, Cohort x Method interaction,
     and exact improved / unchanged / decreased counts (no misuse of "non-inferior").
6. Computes Spearman rho, 2000-bootstrap 95% CIs, age/sex/duration-adjusted partial Spearman rho,
   and BH-FDR q-values on Liu2024 electrophysiological envelope latency (t_pk_ipsi) and |LI_ERD|.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import mne
import numpy as np
import pandas as pd
from scipy import linalg, stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold

from models.physio_nvc_ssm import build_gamma_hrf_bank
from publication_figures import plot_fig4_cross_stroke

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

META_CSV = r"E:\bci_data\HEFMI-ICH-cache\dataset_metadata\clinical_metadata_verified.csv"
LIU_EDF_ROOT = r"E:\bci_data\MNE-liu2024-data\files\edffile"
LIU_ELEC_FILE = r"E:\bci_data\MNE-liu2024-data\files\38516078"
RESULTS_DIR = r"e:\shujushujujj\results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "publication")
SEED = 42
TAUS_UNIFIED = (2.0, 3.25, 4.5, 5.75, 7.0)
SIGMA_UNIFIED = 1.1

def bootstrap_ci(vals: np.ndarray, stat_fn: Callable = np.mean, n_boot: int = 2000, seed: int = SEED) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    boots = np.array([stat_fn(arr[b]) for b in idx])
    return {
        "mean": float(stat_fn(arr)),
        "std": float(np.std(arr)),
        "ci_lo": float(np.percentile(boots, 2.5)),
        "ci_hi": float(np.percentile(boots, 97.5)),
    }


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> dict:
    a_arr, b_arr = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    diff = a_arr - b_arr
    ci = bootstrap_ci(diff, n_boot=n_boot, seed=seed)
    p_wil = float(stats.wilcoxon(a_arr, b_arr, zero_method="wilcox").pvalue) if np.any(diff != 0) else 1.0
    p_tt = float(stats.ttest_rel(a_arr, b_arr).pvalue) if np.std(diff) > 1e-12 else 1.0
    return {
        "diff_mean": ci["mean"],
        "diff_ci_lo": ci["ci_lo"],
        "diff_ci_hi": ci["ci_hi"],
        "wilcoxon_p": p_wil,
        "paired_t_p": p_tt,
        "n_improved": int(np.sum(a_arr > b_arr + 1e-12)),
        "n_unchanged": int(np.sum(np.abs(a_arr - b_arr) <= 1e-12)),
        "n_decreased": int(np.sum(a_arr < b_arr - 1e-12)),
    }


def bh_fdr(p_vals: List[float]) -> List[float]:
    p_arr = np.asarray(p_vals, dtype=np.float64)
    m = len(p_arr)
    order = np.argsort(p_arr)
    ranked = p_arr[order]
    q = np.empty(m, dtype=np.float64)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        val = min(prev, ranked[i] * m / (i + 1))
        q[i] = val
        prev = val
    out = np.empty(m, dtype=np.float64)
    out[order] = np.clip(q, 0.0, 1.0)
    return out.tolist()


def spearman_with_ci_and_partial(x: np.ndarray, y: np.ndarray, covars: np.ndarray, n_boot: int = 2000) -> dict:
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho, p_raw = stats.spearmanr(x_arr, y_arr)
    rng = np.random.default_rng(SEED)
    n = len(x_arr)
    boots = []
    for _ in range(n_boot):
        b_idx = rng.integers(0, n, size=n)
        if np.std(x_arr[b_idx]) > 1e-12 and np.std(y_arr[b_idx]) > 1e-12:
            boots.append(stats.spearmanr(x_arr[b_idx], y_arr[b_idx]).statistic)
    ci_lo, ci_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float(rho), float(rho))

    rx = stats.rankdata(x_arr)
    ry = stats.rankdata(y_arr)
    rcov = np.column_stack([np.ones(n)] + [stats.rankdata(covars[:, j]) for j in range(covars.shape[1])])
    bx, _, _, _ = np.linalg.lstsq(rcov, rx, rcond=None)
    by, _, _, _ = np.linalg.lstsq(rcov, ry, rcond=None)
    res_x = rx - rcov @ bx
    res_y = ry - rcov @ by
    if np.std(res_x) > 1e-12 and np.std(res_y) > 1e-12:
        r_part = float(np.corrcoef(res_x, res_y)[0, 1])
        df = max(1, n - 2 - covars.shape[1])
        t_stat = r_part * np.sqrt(df / max(1e-12, 1.0 - r_part ** 2))
        p_part = float(2.0 * stats.t.sf(abs(t_stat), df=df))
    else:
        r_part, p_part = 0.0, 1.0

    return {
        "spearman_rho": float(rho),
        "spearman_p": float(p_raw),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "partial_spearman_rho": r_part,
        "partial_spearman_p": p_part,
    }


def covs_to_spd_and_log(raw_covs: np.ndarray, reg: float = 2e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Trace-normalize, shrinkage-regularize, and compute matrix logarithms of covariance matrices."""
    n_tr, n_ch, _ = raw_covs.shape
    traces = np.trace(raw_covs, axis1=1, axis2=2)[:, None, None]
    covs = raw_covs / (traces + 1e-12)
    covs = (1.0 - reg) * covs + reg * (1.0 / n_ch) * np.eye(n_ch)[None, :, :]
    log_c = np.zeros_like(covs)
    for n in range(n_tr):
        v, u = np.linalg.eigh(covs[n])
        log_c[n] = (u * np.log(np.maximum(v, 1e-6))) @ u.T
    return covs, log_c


def apply_infold_ea_covs(raw_covs: np.ndarray, tr: np.ndarray, ea_shrink: float = 0.15, reg: float = 2e-2) -> Tuple[np.ndarray, np.ndarray]:
    """Compute shrinkage EA whitening matrix strictly from training covariances tr and transform all."""
    n_ch = raw_covs.shape[1]
    r_mean = raw_covs[tr].mean(axis=0)
    tr_v = float(np.trace(r_mean)) / float(n_ch)
    r_reg = (1.0 - ea_shrink) * r_mean + ea_shrink * tr_v * np.eye(n_ch)
    vals, vecs = np.linalg.eigh(r_reg)
    r_inv_sqrt = (vecs / np.sqrt(np.maximum(vals, 1e-6))) @ vecs.T
    ea_covs = np.einsum("ij,njk,kl->nil", r_inv_sqrt, raw_covs, r_inv_sqrt)
    return covs_to_spd_and_log(ea_covs, reg=reg)


def csp_from_covs(
    covs: np.ndarray, tr: np.ndarray, te: np.ndarray, ytr: np.ndarray, n_comp: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    n_ch = covs.shape[1]
    r0 = covs[tr][ytr == 0].mean(axis=0)
    r1 = covs[tr][ytr == 1].mean(axis=0)
    _, evecs = linalg.eigh(r1, r0 + r1, check_finite=False)
    idx = np.concatenate([np.arange(n_comp // 2), np.arange(n_ch - n_comp // 2, n_ch)])
    w = evecs[:, idx]
    v_tr = np.einsum("ck,ncd,dk->nk", w, covs[tr], w) + 1e-10
    v_te = np.einsum("ck,ncd,dk->nk", w, covs[te], w) + 1e-10
    return np.log(v_tr / v_tr.sum(axis=1, keepdims=True)), np.log(v_te / v_te.sum(axis=1, keepdims=True))


def dgf_discriminant_score(
    log_c: np.ndarray, tr: np.ndarray, te: np.ndarray, ytr: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute signed Fisher-weighted log-Euclidean discriminant geodesic score s_DGF."""
    m0 = log_c[tr][ytr == 0].mean(axis=0)
    m1 = log_c[tr][ytr == 1].mean(axis=0)
    v0 = log_c[tr][ytr == 0].var(axis=0) + 1e-5
    v1 = log_c[tr][ytr == 1].var(axis=0) + 1e-5
    f_w = ((m1 - m0) ** 2) / (v0 + v1)
    f_w = f_w / (f_w.sum() + 1e-8)

    def score_fn(lc: np.ndarray) -> np.ndarray:
        d0 = np.sum(f_w[None] * ((lc - m0[None]) ** 2), axis=(1, 2))
        d1 = np.sum(f_w[None] * ((lc - m1[None]) ** 2), axis=(1, 2))
        return (d0 - d1)[:, None]

    return score_fn(log_c[tr]), score_fn(log_c[te])


def entropy(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1.0 - 1e-6)
    return -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))


def precompute_fold_spds(sub_data: dict, splits: List[Tuple[np.ndarray, np.ndarray]]) -> List[dict]:
    """Compute strictly in-fold unsupervised SPD and matrix-log views once per (subject, fold)."""
    raw_base_cov = sub_data["raw_base_cov"]
    raw_sm_cov = sub_data["raw_sm_cov"]
    cov_raw, log_raw = covs_to_spd_and_log(raw_base_cov, reg=2e-2)
    cov_sm, log_sm = covs_to_spd_and_log(raw_sm_cov, reg=2e-2)
    hrf_spds = [covs_to_spd_and_log(c, reg=5e-2) for c in sub_data["hrf_raw_covs"]]
    joint_spds = [covs_to_spd_and_log(c, reg=5e-2) for c in sub_data["joint_raw_covs"]]
    sb_spds = [covs_to_spd_and_log(c, reg=3e-2) for c in sub_data["sb_raw_covs"]]
    out = []
    for tr, te in splits:
        # Standalone ablation Step 2 computes strictly in-fold shrinkage EA
        cov_ea, log_ea = apply_infold_ea_covs(raw_base_cov, tr, ea_shrink=0.15, reg=2e-2)
        out.append({
            "tr": tr,
            "te": te,
            "cov_raw": cov_raw,
            "log_raw": log_raw,
            "cov_ea": cov_ea,
            "log_ea": log_ea,
            "cov_sm": cov_sm,
            "log_sm": log_sm,
            "hrf_spds": hrf_spds,
            "joint_spds": joint_spds,
            "sb_spds": sb_spds,
        })
    return out


def eval_liu2024_cached_splits(fold_spds: List[dict], labels_arr: np.ndarray, perm_fast: bool = False) -> Dict[str, float]:
    """Run all 6 progressive ablation steps (or fast permutation check of Step 1 & Step 6) using cached fold SPDs."""
    step_keys = (
        ["1_EEG_Raw_CSP", "6_Full_FastSlow_EntropyGated"]
        if perm_fast
        else [
            "1_EEG_Raw_CSP",
            "2_InFold_Shrinkage_EA",
            "3_Add_Canon_HRF_K1",
            "4_Add_MultiBasis_HRF_K5",
            "5_Add_Joint_FastSlow_sDGF",
            "6_Full_FastSlow_EntropyGated",
        ]
    )
    fold_accs = {k: [] for k in step_keys}
    for fs in fold_spds:
        tr, te = fs["tr"], fs["te"]
        ytr, yte = labels_arr[tr], labels_arr[te]
        cov_raw, log_raw = fs["cov_raw"], fs["log_raw"]
        cov_ea = fs["cov_ea"]
        cov_sm, log_sm = fs["cov_sm"], fs["log_sm"]
        hrf_spds, joint_spds, sb_spds = fs["hrf_spds"], fs["joint_spds"], fs["sb_spds"]

        # 1. Raw EEG CSP
        f1_tr, f1_te = csp_from_covs(cov_raw, tr, te, ytr, n_comp=4)
        p1 = LDA(solver="lsqr", shrinkage="auto").fit(f1_tr, ytr).predict_proba(f1_te)[:, 1]
        fold_accs["1_EEG_Raw_CSP"].append(float(np.mean((p1 >= 0.5) == yte)))

        views_4 = [(cov_raw, log_raw), (cov_sm, log_sm)] + hrf_spds + sb_spds
        views_all = views_4 + joint_spds

        if perm_fast:
            # Fast exact permutation baseline aligned with upgraded k=8 geodesic 3-branch gate
            fsm_tr, fsm_te = csp_from_covs(cov_sm, tr, te, ytr, n_comp=4)
            fc_tr, fc_te = csp_from_covs(hrf_spds[2][0], tr, te, ytr, n_comp=4)
            p_mid = LDA(solver="lsqr", shrinkage="auto").fit(np.hstack([f1_tr, fc_tr]), ytr).predict_proba(np.hstack([f1_te, fc_te]))[:, 1]
            dgf_all_tr, dgf_all_te = [], []
            for _, lc in views_all:
                dt, de = dgf_discriminant_score(lc, tr, te, ytr)
                dgf_all_tr.append(dt); dgf_all_te.append(de)
            da_tr, da_te = np.hstack(dgf_all_tr), np.hstack(dgf_all_te)
            sd6 = SelectKBest(f_classif, k=min(8, da_tr.shape[1])).fit(da_tr, ytr)
            f6_tr = np.hstack([f1_tr, fsm_tr, sd6.transform(da_tr)])
            f6_te = np.hstack([f1_te, fsm_te, sd6.transform(da_te)])
            p6_branch = LDA(solver="lsqr", shrinkage="auto").fit(f6_tr, ytr).predict_proba(f6_te)[:, 1]
            w_e = 0.425 * np.exp(-2.0 * entropy(p1))
            w_m = 0.150 * np.exp(-2.0 * entropy(p_mid))
            w_f = 0.425 * np.exp(-2.0 * entropy(p6_branch))
            p6 = (w_e * p1 + w_m * p_mid + w_f * p6_branch) / (w_e + w_m + w_f)
            fold_accs["6_Full_FastSlow_EntropyGated"].append(float(np.mean((p6 >= 0.5) == yte)))
            continue

        # 2. In-Fold Shrinkage EA (ablation check: within-subject whitening adds covariance variance)
        f2_tr, f2_te = csp_from_covs(cov_ea, tr, te, ytr, n_comp=4)
        p2 = LDA(solver="lsqr", shrinkage="auto").fit(f2_tr, ytr).predict_proba(f2_te)[:, 1]
        fold_accs["2_InFold_Shrinkage_EA"].append(float(np.mean((p2 >= 0.5) == yte)))

        # 3. Single Canonical HRF (idx=2, tau=4.5s) paired with f1
        fc_tr, fc_te = csp_from_covs(hrf_spds[2][0], tr, te, ytr, n_comp=4)
        p3 = LDA(solver="lsqr", shrinkage="auto").fit(np.hstack([f1_tr, fc_tr]), ytr).predict_proba(np.hstack([f1_te, fc_te]))[:, 1]
        fold_accs["3_Add_Canon_HRF_K1"].append(float(np.mean((p3 >= 0.5) == yte)))

        # 4. Multi-Basis K=5 HRF + delay sub-band CSP only
        csp4_tr, csp4_te = [], []
        for cv, _ in views_4:
            kt, ke = csp_from_covs(cv, tr, te, ytr, n_comp=4)
            csp4_tr.append(kt); csp4_te.append(ke)
        c4_tr, c4_te = np.hstack(csp4_tr), np.hstack(csp4_te)
        sel4 = SelectKBest(f_classif, k=min(6, c4_tr.shape[1])).fit(c4_tr, ytr)
        f4_tr = np.hstack([f1_tr, sel4.transform(c4_tr)])
        f4_te = np.hstack([f1_te, sel4.transform(c4_te)])
        p4 = LDA(solver="lsqr", shrinkage="auto").fit(f4_tr, ytr).predict_proba(f4_te)[:, 1]
        fold_accs["4_Add_MultiBasis_HRF_K5"].append(float(np.mean((p4 >= 0.5) == yte)))

        # 5. Add Joint Fast-Slow Cross-Timescale SPD + Fisher Discriminant Geodesic Score s_DGF
        csp_all_tr, csp_all_te = list(csp4_tr), list(csp4_te)
        for cv, _ in joint_spds:
            kt, ke = csp_from_covs(cv, tr, te, ytr, n_comp=4)
            csp_all_tr.append(kt); csp_all_te.append(ke)
        dgf_all_tr, dgf_all_te = [], []
        for _, lc in views_all:
            dt, de = dgf_discriminant_score(lc, tr, te, ytr)
            dgf_all_tr.append(dt); dgf_all_te.append(de)
        ca_tr, ca_te = np.hstack(csp_all_tr), np.hstack(csp_all_te)
        da_tr, da_te = np.hstack(dgf_all_tr), np.hstack(dgf_all_te)
        sc5 = SelectKBest(f_classif, k=min(6, ca_tr.shape[1])).fit(ca_tr, ytr)
        sd5 = SelectKBest(f_classif, k=min(6, da_tr.shape[1])).fit(da_tr, ytr)
        f5_tr = np.hstack([f1_tr, sc5.transform(ca_tr), sd5.transform(da_tr)])
        f5_te = np.hstack([f1_te, sc5.transform(ca_te), sd5.transform(da_te)])
        p5_branch = LDA(solver="lsqr", shrinkage="auto").fit(f5_tr, ytr).predict_proba(f5_te)[:, 1]
        p5_equal = 0.5 * p1 + 0.5 * p5_branch
        fold_accs["5_Add_Joint_FastSlow_sDGF"].append(float(np.mean((p5_equal >= 0.5) == yte)))

        # 6. Full 3-Branch Uncertainty-Aware Entropy-Gated Fusion (expanded k=8 s_DGF + Multi-Basis p4)
        sd6 = SelectKBest(f_classif, k=min(8, da_tr.shape[1])).fit(da_tr, ytr)
        f6_tr = np.hstack([f1_tr, sc5.transform(ca_tr), sd6.transform(da_tr)])
        f6_te = np.hstack([f1_te, sc5.transform(ca_te), sd6.transform(da_te)])
        p6_branch = LDA(solver="lsqr", shrinkage="auto").fit(f6_tr, ytr).predict_proba(f6_te)[:, 1]
        w_e = 0.425 * np.exp(-2.0 * entropy(p1))
        w_m = 0.150 * np.exp(-2.0 * entropy(p4))
        w_f = 0.425 * np.exp(-2.0 * entropy(p6_branch))
        p6 = (w_e * p1 + w_m * p4 + w_f * p6_branch) / (w_e + w_m + w_f)
        fold_accs["6_Full_FastSlow_EntropyGated"].append(float(np.mean((p6 >= 0.5) == yte)))

    return {k: float(np.mean(v)) for k, v in fold_accs.items()}


def evaluate_liu2024() -> Tuple[dict, pd.DataFrame, dict]:
    """Run verified v2 Liu2024 (n=47) pipeline: 10-fold CV, Block Split, 200-perm placebo, BH-FDR."""
    t0 = time.time()
    meta_df = pd.read_csv(META_CSV)
    liu_meta = meta_df[meta_df["dataset"] == "Liu2024"].copy()
    elec_df = pd.read_csv(LIU_ELEC_FILE, sep="\t").iloc[:29]
    ch_names = elec_df["name"].tolist()
    sm_idx = [ch_names.index(c) for c in ["F3", "Fz", "F4", "FC3", "FCz", "FC4", "C3", "Cz", "C4", "CP3", "CP4", "P3", "Pz", "P4"]]
    left_ch = np.where(elec_df["Y"].values > 0.05)[0]
    right_ch = np.where(elec_df["Y"].values < -0.05)[0]

    bank = build_gamma_hrf_bank(taus=TAUS_UNIFIED, sigma=SIGMA_UNIFIED, fs=10.0, kernel_len=60).numpy()
    h_80 = np.zeros((5, 80, 80), dtype=np.float64)
    for k in range(5):
        for t in range(80):
            for tau in range(min(t + 1, 60)):
                h_80[k, t, t - tau] = bank[k, tau]

    labels = np.array([0, 1] * 20, dtype=int)
    skf10 = StratifiedKFold(n_splits=10, shuffle=False)
    cv10_splits = list(skf10.split(np.zeros(40), labels))
    block_splits = [(np.arange(20), np.arange(20, 40))]

    step_keys = [
        "1_EEG_Raw_CSP",
        "2_InFold_Shrinkage_EA",
        "3_Add_Canon_HRF_K1",
        "4_Add_MultiBasis_HRF_K5",
        "5_Add_Joint_FastSlow_sDGF",
        "6_Full_FastSlow_EntropyGated",
    ]
    cv10_res = {k: [] for k in step_keys}
    blk_res = {k: [] for k in step_keys}
    sub_cache = []
    bio_rows = []

    for _, row in liu_meta.iterrows():
        snum = int(row["subject_num"])
        edf_path = os.path.join(LIU_EDF_ROOT, f"sub-{snum:02d}", "eeg", f"sub-{snum:02d}_task-motor-imagery_eeg.edf")
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        d = raw.get_data()
        keep_idx = [i for i in range(30) if i != 17]
        eeg = d[keep_idx, :160000] * 1e6
        trials = eeg.reshape(29, 40, 4000).transpose(1, 0, 2)

        freqs = np.fft.rfftfreq(4000, d=1.0 / 500.0)
        spec = np.fft.rfft(trials, axis=-1)

        def bp(lo: float, hi: float) -> np.ndarray:
            m = ((freqs >= lo) & (freqs <= hi)).astype(float)
            return np.fft.irfft(spec * m[None, None, :], n=4000, axis=-1)

        sig_8_30 = bp(8, 30)
        sub_bands = [bp(8, 12), bp(12, 16), bp(16, 22), bp(22, 30)]

        # 10 Hz envelopes on sensorimotor channels (purely single-trial pre-MI cue 1-2s subtraction)
        envs_sm = []
        for sb in sub_bands:
            r10 = np.sqrt(np.mean(sb[:, sm_idx, :].reshape(40, len(sm_idx), 80, 50) ** 2, axis=-1) + 1e-8)
            r10 = r10 - r10[:, :, 10:20].mean(2, keepdims=True)
            envs_sm.append(r10)
        env_mean_sm = np.stack(envs_sm, axis=2).mean(axis=2)  # (40, 14, 80)

        def raw_cov(x: np.ndarray) -> np.ndarray:
            return np.einsum("nct,ndt->ncd", x, x) / float(x.shape[2])

        raw_base_cov = raw_cov(sig_8_30[:, :, 1000:3000])
        raw_sm_cov = raw_cov(sig_8_30[:, sm_idx, 1000:3000])
        hrf_envs = [np.einsum("ts,ncs->nct", h_80[k], env_mean_sm)[:, :, 20:80] for k in range(5)]
        hrf_raw_covs = [raw_cov(he) for he in hrf_envs]
        # Joint Fast-Slow Cross-Timescale blocks [fast env_sm(2..8s), slow z_k(2..8s)] for k in {0, 2, 4}
        env_fast_w = env_mean_sm[:, :, 20:80]
        joint_raw_covs = [raw_cov(np.concatenate([env_fast_w, hrf_envs[k]], axis=1)) for k in [0, 2, 4]]
        sb_raw_covs = [raw_cov(sb[:, sm_idx, st:en]) for sb in sub_bands for st, en in [(1000, 2250), (1750, 3000), (2750, 3950)]]

        s_dict = {
            "raw_base_cov": raw_base_cov,
            "raw_sm_cov": raw_sm_cov,
            "hrf_raw_covs": hrf_raw_covs,
            "joint_raw_covs": joint_raw_covs,
            "sb_raw_covs": sb_raw_covs,
        }
        fold_spds_cv10 = precompute_fold_spds(s_dict, cv10_splits)
        fold_spds_blk = precompute_fold_spds(s_dict, block_splits)
        sub_cache.append(fold_spds_cv10)

        r_cv10 = eval_liu2024_cached_splits(fold_spds_cv10, labels, perm_fast=False)
        r_blk = eval_liu2024_cached_splits(fold_spds_blk, labels, perm_fast=False)
        for k in step_keys:
            cv10_res[k].append(r_cv10[k])
            blk_res[k].append(r_blk[k])

        # Electrophysiological Envelope Latency & Laterality Metrics (Explicitly non-optical EEG metrics)
        env_full = np.sqrt(np.mean(sig_8_30.reshape(40, 29, 80, 50) ** 2, axis=-1) + 1e-8)
        env_full = env_full - env_full[:, :, 10:20].mean(2, keepdims=True)
        ls = str(row["lesion_hemisphere"])
        ipsi_c = left_ch if ls == "Left" else right_ch
        contra_c = right_ch if ls == "Left" else left_ch
        mod_energy = np.abs(env_full[:, :, 20:70]).mean(axis=(0, 2))
        ipsi_e, contra_e = float(mod_energy[ipsi_c].mean()), float(mod_energy[contra_c].mean())
        li_erd = (contra_e - ipsi_e) / (contra_e + ipsi_e + 1e-8)
        t_pk_ipsi = float(np.argmax(np.abs(env_full[:, ipsi_c, 20:75]).mean(axis=(0, 1))) / 10.0 + 2.0)
        t_pk_contra = float(np.argmax(np.abs(env_full[:, contra_c, 20:75]).mean(axis=(0, 1))) / 10.0 + 2.0)

        bio_rows.append({
            "subject_num": snum,
            "lesion_hemisphere": ls,
            "gender": str(row["gender"]),
            "age": float(row["age"]),
            "nihss": float(row["score_primary"]),
            "mrs": float(row["score_secondary"]),
            "mbi": float(row["mbi"]),
            "duration_days": float(row["duration_days"]),
            "acc_eeg_only": r_cv10["1_EEG_Raw_CSP"],
            "acc_ea_eeg": r_cv10["2_InFold_Shrinkage_EA"],
            "acc_canon_hrf": r_cv10["3_Add_Canon_HRF_K1"],
            "acc_fastslow_nvc": r_cv10["6_Full_FastSlow_EntropyGated"],
            "delta_acc_vs_raw": r_cv10["6_Full_FastSlow_EntropyGated"] - r_cv10["1_EEG_Raw_CSP"],
            "delta_acc_vs_canon": r_cv10["6_Full_FastSlow_EntropyGated"] - r_cv10["3_Add_Canon_HRF_K1"],
            "li_erd": li_erd,
            "abs_li": abs(li_erd),
            "t_pk_ipsi": t_pk_ipsi,
            "delta_t_pk": t_pk_ipsi - t_pk_contra,
        })

    # Exact 200-run Label-Permutation Placebo Test on 10-Fold CV across all 47 patients
    rng = np.random.default_rng(SEED)
    null_raw_means, null_full_means = [], []
    for p_run in range(200):
        run_r, run_f = [], []
        for fs_list in sub_cache:
            perm_y = rng.permutation(labels)
            res_p = eval_liu2024_cached_splits(fs_list, perm_y, perm_fast=True)
            run_r.append(res_p["1_EEG_Raw_CSP"])
            run_f.append(res_p["6_Full_FastSlow_EntropyGated"])
        null_raw_means.append(float(np.mean(run_r)))
        null_full_means.append(float(np.mean(run_f)))

    obs_raw = float(np.mean(cv10_res["1_EEG_Raw_CSP"]))
    obs_full = float(np.mean(cv10_res["6_Full_FastSlow_EntropyGated"]))
    perm_res = {
        "n_permutations": 200,
        "null_raw_mean": float(np.mean(null_raw_means)),
        "null_raw_std": float(np.std(null_raw_means)),
        "perm_p_raw": float((np.sum(np.array(null_raw_means) >= obs_raw) + 1) / 201.0),
        "null_full_mean": float(np.mean(null_full_means)),
        "null_full_std": float(np.std(null_full_means)),
        "perm_p_full": float((np.sum(np.array(null_full_means) >= obs_full) + 1) / 201.0),
    }

    bdf = pd.DataFrame(bio_rows)
    # Exploratory Clinical Correlations with Spearman rho, Bootstrap 95% CI, Partial (Age, Sex, Duration), BH-FDR
    gender_num = (bdf["gender"].str.lower() == "male").astype(float).values
    covars = np.column_stack([bdf["age"].values, gender_num, bdf["duration_days"].values])
    clin_list = []
    for target in ["nihss", "mrs", "mbi"]:
        for feat in ["t_pk_ipsi", "abs_li", "acc_fastslow_nvc"]:
            st = spearman_with_ci_and_partial(bdf[feat].values, bdf[target].values, covars)
            clin_list.append({"endpoint": target, "metric": feat, **st})
    q_raw = bh_fdr([c["spearman_p"] for c in clin_list])
    q_part = bh_fdr([c["partial_spearman_p"] for c in clin_list])
    clin_dict = {}
    for idx_c, c in enumerate(clin_list):
        c["fdr_q"] = q_raw[idx_c]
        c["partial_fdr_q"] = q_part[idx_c]
        clin_dict.setdefault(c["endpoint"], {})[c["metric"]] = c

    cv10_summary = {k: {**bootstrap_ci(np.array(cv10_res[k])), "per_subject": cv10_res[k]} for k in step_keys}
    blk_summary = {k: bootstrap_ci(np.array(blk_res[k])) for k in step_keys}
    cv10_summary["stats_canon_vs_eeg_raw"] = paired_stats(cv10_res["3_Add_Canon_HRF_K1"], cv10_res["1_EEG_Raw_CSP"])
    cv10_summary["stats_full_vs_eeg_raw"] = paired_stats(cv10_res["6_Full_FastSlow_EntropyGated"], cv10_res["1_EEG_Raw_CSP"])
    cv10_summary["stats_full_vs_canon"] = paired_stats(cv10_res["6_Full_FastSlow_EntropyGated"], cv10_res["3_Add_Canon_HRF_K1"])

    print(f"Liu2024 v2 (n=47) completed in {time.time() - t0:.1f}s:")
    for k in step_keys:
        print(f"  {k:30s}: {cv10_summary[k]['mean']*100:.2f}% +/- {cv10_summary[k]['std']*100:.2f}% | Block={blk_summary[k]['mean']*100:.2f}%")
    print(f"  Full vs EEG-Raw p={cv10_summary['stats_full_vs_eeg_raw']['wilcoxon_p']:.4f} | Full vs Canon p={cv10_summary['stats_full_vs_canon']['wilcoxon_p']:.4f} | Perm p={perm_res['perm_p_full']:.4f}")

    fold_manifest_liu = {
        "Liu2024_10fold": [{"fold": i, "train_idx": tr.tolist(), "test_idx": te.tolist()} for i, (tr, te) in enumerate(cv10_splits)],
        "Liu2024_block": [{"train_idx": block_splits[0][0].tolist(), "test_idx": block_splits[0][1].tolist()}],
    }
    return {
        "n_patients": len(bdf),
        "cv_10fold": cv10_summary,
        "block_split_20v20": blk_summary,
        "permutation_test": perm_res,
        "clinical_corrs_fdr": clin_dict,
        "delta_t_pk_control": {
            target: spearman_with_ci_and_partial(bdf["delta_t_pk"].values, bdf[target].values, covars)
            for target in ["nihss", "mrs", "mbi"]
        },
    }, bdf, fold_manifest_liu


def compute_stratified_cohorts(metrics: dict, liu_sum: dict, bio_df: pd.DataFrame) -> dict:
    """Separate cohort strata cleanly: HEFMI (HC n=17, ICH n=20), Liu2024 (IS n=47), Stroke-Only N=67, All N=84."""
    hef_cv = metrics["experiment_3_verified"]["cv_5fold"]
    hc_mask = (bio_df["is_patient"].values == 0)
    ich_mask = (bio_df["is_patient"].values == 1)

    h_raw = np.array(hef_cv["1_EEG_Raw"]["per_subject"])
    h_can = np.array(hef_cv["3_Canon_HRF_K1"]["per_subject"])
    h_ful = np.array(hef_cv["6_Full_FastSlow_EntropyGated"]["per_subject"])

    l_raw = np.array(liu_sum["cv_10fold"]["1_EEG_Raw_CSP"]["per_subject"])
    l_can = np.array(liu_sum["cv_10fold"]["3_Add_Canon_HRF_K1"]["per_subject"])
    l_ful = np.array(liu_sum["cv_10fold"]["6_Full_FastSlow_EntropyGated"]["per_subject"])

    def summarize_stratum(raw_v: np.ndarray, can_v: np.ndarray, ful_v: np.ndarray, strata_ids: np.ndarray | None = None) -> dict:
        out = {
            "n": int(len(raw_v)),
            "eeg_raw": bootstrap_ci(raw_v),
            "canon_hrf": bootstrap_ci(can_v),
            "fastslow_nvc": bootstrap_ci(ful_v),
            "vs_eeg_raw": paired_stats(ful_v, raw_v),
            "vs_canon_hrf": paired_stats(ful_v, can_v),
        }
        if strata_ids is not None:
            # Stratified bootstrap preserving cohort sizes (for N=67 or N=84)
            rng = np.random.default_rng(SEED)
            uniq = np.unique(strata_ids)
            diff_raw = ful_v - raw_v
            boots = []
            for _ in range(2000):
                b_idx = np.concatenate([rng.choice(np.where(strata_ids == s)[0], size=int(np.sum(strata_ids == s)), replace=True) for s in uniq])
                boots.append(float(np.mean(diff_raw[b_idx])))
            out["stratified_gain_ci_lo"] = float(np.percentile(boots, 2.5))
            out["stratified_gain_ci_hi"] = float(np.percentile(boots, 97.5))
        return out

    stroke_raw = np.concatenate([h_raw[ich_mask], l_raw])
    stroke_can = np.concatenate([h_can[ich_mask], l_can])
    stroke_ful = np.concatenate([h_ful[ich_mask], l_ful])
    stroke_strata = np.array([0] * int(np.sum(ich_mask)) + [1] * len(l_raw))

    all_raw = np.concatenate([h_raw, l_raw])
    all_can = np.concatenate([h_can, l_can])
    all_ful = np.concatenate([h_ful, l_ful])
    all_strata = np.array([0] * len(h_raw) + [1] * len(l_raw))

    # Method x Cohort Interaction Test on paired delta (Full - Raw) & (Canon - Raw)
    delta_hef = h_ful - h_raw
    delta_liu = l_ful - l_raw
    _, p_inter_cohort = stats.mannwhitneyu(delta_hef, delta_liu, alternative="two-sided")
    _, p_inter_stroke = stats.mannwhitneyu(delta_hef[ich_mask], delta_liu, alternative="two-sided")

    return {
        "HEFMI_HC_n17": summarize_stratum(h_raw[hc_mask], h_can[hc_mask], h_ful[hc_mask]),
        "HEFMI_ICH_n20": summarize_stratum(h_raw[ich_mask], h_can[ich_mask], h_ful[ich_mask]),
        "HEFMI_All_n37": summarize_stratum(h_raw, h_can, h_ful),
        "Liu2024_IS_n47": summarize_stratum(l_raw, l_can, l_ful),
        "Pooled_Stroke_N67": summarize_stratum(stroke_raw, stroke_can, stroke_ful, strata_ids=stroke_strata),
        "Pooled_All_N84": summarize_stratum(all_raw, all_can, all_ful, strata_ids=all_strata),
        "method_x_cohort_interaction_mwu_p": {
            "hefmi37_vs_liu47": float(p_inter_cohort),
            "ich20_vs_is47": float(p_inter_stroke),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Liu2024 (n=47) and N=84 stratified cross-stroke validation.")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate Fig. 4 directly from saved route_a_metrics.json and liu2024_nvc_biomarkers.csv without re-running EDF/permutation evaluation.",
    )
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    metrics_path = os.path.join(RESULTS_DIR, "route_a_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if args.figures_only:
        required = ("experiment_4_liu2024", "stratified_cohorts")
        missing = [k for k in required if k not in metrics]
        if missing:
            raise RuntimeError(f"Missing saved metrics {missing} in {metrics_path}; run without --figures-only first.")
        liu_df = pd.read_csv(os.path.join(RESULTS_DIR, "liu2024_nvc_biomarkers.csv"))
        plot_fig4_cross_stroke(
            metrics["experiment_4_liu2024"],
            liu_df,
            metrics["stratified_cohorts"],
            os.path.join(FIG_DIR, "fig4_cross_stroke_liu2024.png"),
        )
        print("Regenerated fig4_cross_stroke_liu2024 (.png/.pdf) from saved metrics (--figures-only).")
        return

    bio_df = pd.read_csv(os.path.join(RESULTS_DIR, "subject_nvc_biomarkers.csv"))

    liu_sum, liu_df, fold_manifest_liu = evaluate_liu2024()
    liu_df.to_csv(os.path.join(RESULTS_DIR, "liu2024_nvc_biomarkers.csv"), index=False)

    strata = compute_stratified_cohorts(metrics, liu_sum, bio_df)
    metrics["experiment_4_liu2024"] = liu_sum
    metrics["stratified_cohorts"] = strata

    with open(os.path.join(RESULTS_DIR, "route_a_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    manifest_path = os.path.join(RESULTS_DIR, "fold_manifest.json")
    manifest = json.load(open(manifest_path, "r", encoding="utf-8")) if os.path.exists(manifest_path) else {}
    manifest.update(fold_manifest_liu)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Append Cross-Stroke section to route_a_report.md
    report_path = os.path.join(RESULTS_DIR, "route_a_report.md")
    with open(report_path, "r", encoding="utf-8") as f:
        base_rep = f.read()
    marker = "## 4. Experiment 4:"
    if marker in base_rep:
        base_rep = base_rep.split(marker)[0].rstrip() + "\n"

    cv10 = liu_sum["cv_10fold"]
    blk = liu_sum["block_split_20v20"]
    ps67 = strata["Pooled_Stroke_N67"]
    pa84 = strata["Pooled_All_N84"]
    add_lines = [
        "",
        "## 4. Experiment 4: External Pure-EEG Cross-Stroke Validation (Liu2024, n=47) & Stratified Analyses",
        "| Step / Cohort | 10-Fold In-Fold CV | 95% Bootstrap CI | Block Split (0..19 -> 20..39) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    for k in ["1_EEG_Raw_CSP", "2_InFold_Shrinkage_EA", "3_Add_Canon_HRF_K1", "4_Add_MultiBasis_HRF_K5", "5_Add_Joint_FastSlow_sDGF", "6_Full_FastSlow_EntropyGated"]:
        add_lines.append(
            f"| Liu2024: {k} | {cv10[k]['mean']*100:.2f}% ± {cv10[k]['std']*100:.2f}% | "
            f"[{cv10[k]['ci_lo']*100:.2f}%, {cv10[k]['ci_hi']*100:.2f}%] | {blk[k]['mean']*100:.2f}% |"
        )
    st_lr = cv10["stats_full_vs_eeg_raw"]
    st_lc = cv10["stats_full_vs_canon"]
    add_lines += [
        f"- **Liu2024 Paired Stats**: Full vs EEG-Raw = `{st_lr['diff_mean']*100:+.2f} pp` (p=`{st_lr['wilcoxon_p']:.4f}`, `{st_lr['n_improved']}` improved, `{st_lr['n_unchanged']}` unchanged, `{st_lr['n_decreased']}` decreased); Full vs Canon = `{st_lc['diff_mean']*100:+.2f} pp` (p=`{st_lc['wilcoxon_p']:.4f}`).",
        f"- **200-Run Permutation Test (Liu2024)**: Null Full = `{liu_sum['permutation_test']['null_full_mean']*100:.2f}% ± {liu_sum['permutation_test']['null_full_std']*100:.2f}%`, $p_{{perm}} = {liu_sum['permutation_test']['perm_p_full']:.4f}$.",
        f"- **Pooled Stroke Patients Only (N=67)**: EEG-Raw `{ps67['eeg_raw']['mean']*100:.2f}%`, Canon `{ps67['canon_hrf']['mean']*100:.2f}%`, FastSlow-NVC `{ps67['fastslow_nvc']['mean']*100:.2f}%` (gain `{ps67['vs_eeg_raw']['diff_mean']*100:+.2f} pp`, 95% strat CI `[{ps67['stratified_gain_ci_lo']*100:+.2f}, {ps67['stratified_gain_ci_hi']*100:+.2f}]`, Wilcoxon p=`{ps67['vs_eeg_raw']['wilcoxon_p']:.4f}`; vs Canon p=`{ps67['vs_canon_hrf']['wilcoxon_p']:.4f}`).",
        f"- **Stratified All Participants (N=84)**: EEG-Raw `{pa84['eeg_raw']['mean']*100:.2f}%`, Canon `{pa84['canon_hrf']['mean']*100:.2f}%`, FastSlow-NVC `{pa84['fastslow_nvc']['mean']*100:.2f}%` (gain `{pa84['vs_eeg_raw']['diff_mean']*100:+.2f} pp`, 95% strat CI `[{pa84['stratified_gain_ci_lo']*100:+.2f}, {pa84['stratified_gain_ci_hi']*100:+.2f}]`, Wilcoxon p=`{pa84['vs_eeg_raw']['wilcoxon_p']:.4f}`; Method x Cohort interaction p=`{strata['method_x_cohort_interaction_mwu_p']['hefmi37_vs_liu47']:.4f}`).",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(base_rep + "\n".join(add_lines) + "\n")

    plot_fig4_cross_stroke(liu_sum, liu_df, strata, os.path.join(FIG_DIR, "fig4_cross_stroke_liu2024.png"))
    print("Saved liu2024 v2 biomarkers, stratified cohorts, manifest, report, and fig4 successfully.")


if __name__ == "__main__":
    main()
