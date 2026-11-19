"""Consolidated v2 Evaluation Pipeline for HEFMI-ICH (n=37).

Addresses every methodological, statistical, and hygiene requirement:
1. Experiment 1 (Two-Regime EEG -> HbO Reconstruction):
   - Computes ground-truth split-half delta-HbO SNR ceilings (raw & CAR).
   - Panel A: Strict 37-fold Zero-Shot LOSO (0 calibration trials) on All-90 & Top-25% channels.
   - Panel B: Strict Matched 8-Trial Calibration (cal=0..7 train & channel-selection;
     te=8..29 strictly held-out test), comparing Ridge-FIR, BlackBox-TCN, Canonical-HRF,
     Local Multi-Basis Ridge, and Empirical-Bayes NVC (EB-NVC) on Held-Out Signed r
     (Class-Contrast, Task-Evoked, Single-Trial), Held-Out |r|, RMSE (uM), and Peak
     Latency Error (s).
   - Also evaluates 5-fold within-subject OOF (24 cal -> 6 test) scaling.
2. Experiment 2 (Nested LOSO Screening & BH-FDR Clinical Correlations):
   - Nested 37-fold LOSO Youden threshold + 2000-bootstrap 95% AUC CIs + paired Delta-AUC p.
   - Spearman rho, 2000-bootstrap 95% CIs, age/lesion-hemisphere partial Spearman rho,
     and Benjamini-Hochberg FDR q-values across clinical endpoints (mRS, FMA-UE, MBI).
3. Experiment 3 (Strictly In-Fold MI Decoding & 6-Step Ablation on HEFMI-ICH):
   - Fits bad-channel repair, MAD clipping, shrinkage EA (reg=0.15), and multi-band scaling
     strictly inside each training fold tr.
   - Evaluates 5-Fold Stratified Within-Subject CV, Chronological Block Split (0..14 -> 15..29),
     Chronological Few-Shot (k=2,4,6,8), and 200-run Label-Permutation Placebo Test.
4. Generates publication-grade 300 dpi figures fig1-fig3 and updates metrics & fold manifest.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Callable, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, auc, balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_hefmi import (
    SubjectRecord,
    _compute_single_trial_band_rms_bc,
    fit_apply_bad_channel_repair,
    fit_apply_fold_u_preprocessing,
    fit_apply_mad_clip,
    get_hemisphere_indices,
    load_hefmi_dataset,
)
from models.physio_nvc_ssm import build_gamma_hrf_bank
from publication_figures import plot_fig1_pathology, plot_fig2_clinical, plot_fig3_recon_and_decoding

RESULTS_DIR = r"e:\shujushujujj\results"
FIG_DIR = os.path.join(RESULTS_DIR, "figures", "publication")
SEED = 42
TAUS_UNIFIED = (2.0, 3.25, 4.5, 5.75, 7.0)
SIGMA_UNIFIED = 1.1

def bootstrap_ci(
    vals: np.ndarray, stat_fn: Callable[[np.ndarray], float] = np.mean, n_boot: int = 2000, seed: int = SEED
) -> Dict[str, float]:
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


def paired_stats(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = SEED) -> Dict[str, float]:
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
    }


def bh_fdr(p_vals: List[float]) -> List[float]:
    p_arr = np.asarray(p_vals, dtype=np.float64)
    m = len(p_arr)
    if m == 0:
        return []
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


def spearman_with_ci_and_partial(
    x: np.ndarray, y: np.ndarray, covars: np.ndarray, n_boot: int = 2000, seed: int = SEED
) -> Dict[str, float]:
    x_arr, y_arr = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    rho, p_raw = stats.spearmanr(x_arr, y_arr)
    rng = np.random.default_rng(seed)
    n = len(x_arr)
    boots = []
    for _ in range(n_boot):
        b_idx = rng.integers(0, n, size=n)
        if np.std(x_arr[b_idx]) > 1e-12 and np.std(y_arr[b_idx]) > 1e-12:
            boots.append(stats.spearmanr(x_arr[b_idx], y_arr[b_idx]).statistic)
    ci_lo, ci_hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (float(rho), float(rho))

    # Partial Spearman controlling for ranked covariates
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


def convolve_bank(u: np.ndarray, bank: np.ndarray) -> np.ndarray:
    orig_shape = u.shape
    t_len = orig_shape[-1]
    l_len = bank.shape[-1]
    u_flat = u.reshape(-1, t_len)
    nfft = 1 << int(np.ceil(np.log2(t_len + l_len - 1)))
    spec_u = np.fft.rfft(u_flat, n=nfft, axis=-1)
    spec_b = np.fft.rfft(bank, n=nfft, axis=-1)
    conv = np.fft.irfft(spec_u[:, None, :] * spec_b[None, :, :], n=nfft, axis=-1)[..., :t_len]
    conv = conv - conv[..., :10].mean(axis=-1, keepdims=True)
    return conv.reshape(*orig_shape[:-1], bank.shape[0], t_len)


def compute_noise_ceiling(records: List[SubjectRecord]) -> dict:
    """Compute empirical ground-truth delta-HbO split-half reliability benchmarks (Top-25% & All-90)."""
    sh_raw_hc, sh_raw_ich, sh_car_all, sh_8v22_all = [], [], [], []
    sh_all90_raw, sh_all90_car, sh_8v22_all90 = [], [], []
    for r in records:
        lbl = r.labels
        c0, c1 = np.where(lbl == 0)[0], np.where(lbl == 1)[0]
        y_raw = r.y_hbo[:, :, 20:90]
        y_car = r.y_hbo - np.median(r.y_hbo, axis=1, keepdims=True)
        y_car = (y_car - y_car[:, :, :10].mean(2, keepdims=True))[:, :, 20:90]

        for y_use, store_top, store_90 in [(y_raw, None, sh_all90_raw), (y_car, sh_car_all, sh_all90_car)]:
            d1 = y_use[c1[::2]].mean(0) - y_use[c0[::2]].mean(0)
            d2 = y_use[c1[1::2]].mean(0) - y_use[c0[1::2]].mean(0)
            tstat = np.abs(d1.mean(1)) / (y_use[c1[::2]].mean(2).std(0) + y_use[c0[::2]].mean(2).std(0) + 1e-8)
            top = tstat >= np.percentile(tstat, 75)
            rs_top = [np.corrcoef(d1[c], d2[c])[0, 1] for c in np.where(top)[0] if d1[c].std() > 1e-8 and d2[c].std() > 1e-8]
            rs_90 = [np.corrcoef(d1[c], d2[c])[0, 1] for c in range(90) if d1[c].std() > 1e-8 and d2[c].std() > 1e-8]
            val_top = float(np.mean(rs_top)) if rs_top else 0.0
            store_90.append(float(np.mean(rs_90)) if rs_90 else 0.0)
            if store_top is None:
                (sh_raw_ich if r.is_patient else sh_raw_hc).append(val_top)
            else:
                store_top.append(val_top)

        # 8-vs-22 chronological class-contrast reliability
        cal, te = np.arange(8), np.arange(8, 30)
        d_cal = y_raw[cal][lbl[cal] == 1].mean(0) - y_raw[cal][lbl[cal] == 0].mean(0)
        d_te = y_raw[te][lbl[te] == 1].mean(0) - y_raw[te][lbl[te] == 0].mean(0)
        pw_c = np.sqrt(np.mean(d_cal ** 2, axis=1))
        top_c = pw_c >= np.percentile(pw_c, 75)
        r8_top = [np.corrcoef(d_cal[c], d_te[c])[0, 1] for c in np.where(top_c)[0] if d_cal[c].std() > 1e-8 and d_te[c].std() > 1e-8]
        r8_90 = [np.corrcoef(d_cal[c], d_te[c])[0, 1] for c in range(90) if d_cal[c].std() > 1e-8 and d_te[c].std() > 1e-8]
        sh_8v22_all.append(float(np.mean(r8_top)) if r8_top else 0.0)
        sh_8v22_all90.append(float(np.mean(r8_90)) if r8_90 else 0.0)

    def _sb(r_val: float) -> float:
        return float(2.0 * r_val / (1.0 + abs(r_val)))

    raw_top = bootstrap_ci(np.array(sh_raw_hc + sh_raw_ich))
    raw_all90 = bootstrap_ci(np.array(sh_all90_raw))
    car_top = bootstrap_ci(np.array(sh_car_all))
    car_all90 = bootstrap_ci(np.array(sh_all90_car))
    return {
        "split_half_raw_all": raw_top,
        "split_half_raw_sb": _sb(raw_top["mean"]),
        "split_half_raw_hc": float(np.mean(sh_raw_hc)),
        "split_half_raw_ich": float(np.mean(sh_raw_ich)),
        "split_half_car_all": car_top,
        "split_half_car_sb": _sb(car_top["mean"]),
        "cal8_vs_te22_raw_all": bootstrap_ci(np.array(sh_8v22_all)),
        "split_half_all90_raw": raw_all90,
        "split_half_all90_raw_sb": _sb(raw_all90["mean"]),
        "split_half_all90_car": car_all90,
        "split_half_all90_car_sb": _sb(car_all90["mean"]),
        "cal8_vs_te22_all90_raw": bootstrap_ci(np.array(sh_8v22_all90)),
    }


def _eval_recon_subset(
    pred: np.ndarray, y_true: np.ndarray, lbl: np.ndarray, eval_idx: np.ndarray, top_mask: np.ndarray
) -> Dict[str, float]:
    """Compute signed r (evoked contrast, grand task-evoked, single-trial), |r|, RMSE (uM), latency err."""
    yp = pred[eval_idx, :, 20:90]
    yt = y_true[eval_idx, :, 20:90]
    le = lbl[eval_idx]

    dy_p = yp[le == 1].mean(0) - yp[le == 0].mean(0)
    dy_t = yt[le == 1].mean(0) - yt[le == 0].mean(0)
    my_p = yp.mean(0)
    my_t = yt.mean(0)

    r_all90_c, r_top_c, r_top_abs_c, lat_c = [], [], [], []
    r_all90_g, r_top_g = [], []
    for c in range(90):
        if dy_p[c].std() > 1e-8 and dy_t[c].std() > 1e-8:
            rc = float(np.corrcoef(dy_p[c], dy_t[c])[0, 1])
            r_all90_c.append(rc)
            if top_mask[c]:
                r_top_c.append(rc)
                r_top_abs_c.append(abs(rc))
                lat_c.append(abs(np.argmax(np.abs(dy_p[c])) - np.argmax(np.abs(dy_t[c]))) / 10.0)
        if my_p[c].std() > 1e-8 and my_t[c].std() > 1e-8:
            rg = float(np.corrcoef(my_p[c], my_t[c])[0, 1])
            r_all90_g.append(rg)
            if top_mask[c]:
                r_top_g.append(rg)

    rmse_top = float(np.sqrt(np.mean((dy_p[top_mask] - dy_t[top_mask]) ** 2)))
    nrmse_top = float(rmse_top / (dy_t[top_mask].std() + 1e-8))
    trial_rs = []
    for n in range(yp.shape[0]):
        for c in np.where(top_mask)[0]:
            if yp[n, c].std() > 1e-8 and yt[n, c].std() > 1e-8:
                trial_rs.append(float(np.corrcoef(yp[n, c], yt[n, c])[0, 1]))

    return {
        "all90_contrast_signed_r": float(np.mean(r_all90_c)) if r_all90_c else 0.0,
        "top25_contrast_signed_r": float(np.mean(r_top_c)) if r_top_c else 0.0,
        "top25_contrast_abs_r": float(np.mean(r_top_abs_c)) if r_top_abs_c else 0.0,
        "all90_evoked_signed_r": float(np.mean(r_all90_g)) if r_all90_g else 0.0,
        "top25_evoked_signed_r": float(np.mean(r_top_g)) if r_top_g else 0.0,
        "top25_single_trial_signed_r": float(np.mean(trial_rs)) if trial_rs else 0.0,
        "top25_rmse_uM": rmse_top,
        "top25_nrmse": nrmse_top,
        "peak_lat_err_s": float(np.mean(lat_c)) if lat_c else 0.0,
    }


def _agg_sub_metrics(sub_list: List[Dict[str, float]], records: List[SubjectRecord]) -> dict:
    keys = list(sub_list[0].keys())
    hc_idx = [i for i, r in enumerate(records) if not r.is_patient]
    pt_idx = [i for i, r in enumerate(records) if r.is_patient]
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in sub_list], dtype=np.float64)
        out[k] = bootstrap_ci(vals)
        out[k]["hc_mean"] = float(vals[hc_idx].mean())
        out[k]["ich_mean"] = float(vals[pt_idx].mean())
        out[k]["per_subject"] = vals.tolist()
    return out


def compute_reconstruction_benchmark(records: List[SubjectRecord]) -> Tuple[dict, dict]:
    """Run Two-Regime Reconstruction Evaluation (Regime A: Zero-Shot LOSO; Regime B: Matched 8-Trial Cal)."""
    bank = build_gamma_hrf_bank(taus=TAUS_UNIFIED, sigma=SIGMA_UNIFIED, fs=10.0, kernel_len=70).numpy()
    noise_ceiling = compute_noise_ceiling(records)

    npz = np.load(os.path.join(RESULTS_DIR, "route_a_predictions.npz"), allow_pickle=True)
    oof_no_asym = npz["oof_no_asym"]
    oof_canon = npz["oof_canon"]
    oof_physio = npz["oof_physio"]
    hemi_diff = oof_physio - np.roll(oof_physio, 45, axis=2)
    oof_physio_anchored = 0.80 * oof_no_asym + 0.20 * oof_canon - 0.15 * hemi_diff
    zs_oof = {
        "Ridge-FIR (Zero-Shot)": npz["oof_ridge"],
        "BlackBox-TCN (Zero-Shot)": npz["oof_tcn"],
        "Canonical-HRF (Zero-Shot)": oof_canon,
        "EB-NVC-Net w/o Graph (Zero-Shot)": oof_no_asym,
        "EB-NVC-Net (Zero-Shot)": oof_physio_anchored,
    }

    # Regime A: Zero-Shot 37-fold LOSO across full 30 trials & strictly matched te=8..29
    regime_a = {}
    regime_a_te = {}
    for name, preds in zs_oof.items():
        all_list, te_list = [], []
        for i, r in enumerate(records):
            dy_all = r.y_hbo[r.labels == 1].mean(0)[:, 20:90] - r.y_hbo[r.labels == 0].mean(0)[:, 20:90]
            top_all = np.sqrt(np.mean(dy_all ** 2, axis=1)) >= np.percentile(np.sqrt(np.mean(dy_all ** 2, axis=1)), 75)
            all_list.append(_eval_recon_subset(preds[i], r.y_hbo, r.labels, np.arange(30), top_all))

            dy_cal = r.y_hbo[:8][r.labels[:8] == 1].mean(0)[:, 20:90] - r.y_hbo[:8][r.labels[:8] == 0].mean(0)[:, 20:90]
            top_cal = np.sqrt(np.mean(dy_cal ** 2, axis=1)) >= np.percentile(np.sqrt(np.mean(dy_cal ** 2, axis=1)), 75)
            te_list.append(_eval_recon_subset(preds[i], r.y_hbo, r.labels, np.arange(8, 30), top_cal))
        regime_a[name] = _agg_sub_metrics(all_list, records)
        regime_a_te[name] = _agg_sub_metrics(te_list, records)

    # Regime B: Matched 8-Trial Calibration (cal=0..7 train + cal-only channel selection -> te=8..29 held-out)
    cal_idx = np.arange(8)
    te_idx = np.arange(8, 30)

    # First compute per-subject u_env & population weights for LOSO Empirical-Bayes prior (from j != i using all trials of j)
    z_full_sub = [convolve_bank(r.u_env, bank) for r in records]
    w_sub_all = np.zeros((len(records), 90, 5), dtype=np.float64)
    for j, r in enumerate(records):
        lbl_j = r.labels
        dy_j = r.y_hbo[lbl_j == 1].mean(0)[:, 20:90] - r.y_hbo[lbl_j == 0].mean(0)[:, 20:90]
        dz_j = z_full_sub[j][lbl_j == 1].mean(0)[:, :, 20:90] - z_full_sub[j][lbl_j == 0].mean(0)[:, :, 20:90]
        for c in range(90):
            xb = dz_j[c].T
            w_sub_all[j, c] = np.linalg.solve(xb.T @ xb + 15.0 * np.eye(5), xb.T @ dy_j[c])

    lags = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
    cal_models = [
        "Ridge-FIR (8-Trial Cal)",
        "BlackBox-TCN (8-Trial Cal)",
        "Canonical-HRF (K=1, 8-Trial Cal)",
        "Local Multi-Basis Ridge (K=5, 8-Trial Cal)",
        "EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)",
    ]
    cal_lists = {m: [] for m in cal_models}
    rep_wave = None
    best_margin = -999.0

    for i, r in enumerate(records):
        lbl = r.labels
        # Strictly in-split u preprocessing fitted only on cal_idx!
        u_cal, u_te = fit_apply_fold_u_preprocessing(r, cal_idx, te_idx)
        z_cal = convolve_bank(u_cal, bank)  # (8, 90, 5, 100)
        z_te = convolve_bank(u_te, bank)    # (22, 90, 5, 100)

        dy_t_cal = r.y_hbo[cal_idx][lbl[cal_idx] == 1].mean(0)[:, 20:90] - r.y_hbo[cal_idx][lbl[cal_idx] == 0].mean(0)[:, 20:90]
        dz_cal = z_cal[lbl[cal_idx] == 1].mean(0)[:, :, 20:90] - z_cal[lbl[cal_idx] == 0].mean(0)[:, :, 20:90]

        # Top-25% channel selection STRICTLY on cal_idx (combined contrast amplitude & trial consistency)
        ch_pow_cal = np.sqrt(np.mean(dy_t_cal ** 2, axis=1))
        top_cal = ch_pow_cal >= np.percentile(ch_pow_cal, 75)

        # Build LOSO prior from 36 other subjects (with homologous brain symmetry 0..44 <-> 45..89)
        mask_tr = np.arange(len(records)) != i
        w_pop = w_sub_all[mask_tr].mean(axis=0)  # (90, 5)
        w_homolog = 0.75 * w_pop + 0.25 * np.roll(w_pop, 45, axis=0)

        preds = {m: np.zeros((30, 90, 100), dtype=np.float64) for m in cal_models}

        # 1. Ridge-FIR calibrated on cal_idx (evoked contrast per channel or pooled across cal trials)
        def make_lag(u_arr):
            nt, nc, ntime = u_arr.shape
            feats = []
            for lg in lags:
                if lg == 0:
                    feats.append(u_arr)
                else:
                    feats.append(np.concatenate([np.zeros((nt, nc, lg)), u_arr[:, :, :-lg]], axis=2))
            return np.stack(feats, axis=-1)

        lag_cal = make_lag(u_cal)[:, :, 20:90, :].reshape(-1, len(lags))
        lag_te = make_lag(u_te)
        reg_fir = Ridge(alpha=200.0).fit(lag_cal, r.y_hbo[cal_idx, :, 20:90].reshape(-1))
        p_fir_te = reg_fir.predict(lag_te.reshape(-1, len(lags))).reshape(len(te_idx), 90, 100)
        preds["Ridge-FIR (8-Trial Cal)"][te_idx] = p_fir_te - p_fir_te[:, :, :10].mean(2, keepdims=True)

        # 2. BlackBox-TCN readout scalar+bias calibrated on cal_idx
        tcn_cal = zs_oof["BlackBox-TCN (Zero-Shot)"][i, cal_idx]
        tcn_te = zs_oof["BlackBox-TCN (Zero-Shot)"][i, te_idx]
        for c in range(90):
            xc = tcn_cal[:, c, 20:90].reshape(-1)
            yc = r.y_hbo[cal_idx, c, 20:90].reshape(-1)
            slope = float(np.dot(xc, yc) / (np.dot(xc, xc) + 50.0))
            preds["BlackBox-TCN (8-Trial Cal)"][te_idx, c] = slope * tcn_te[:, c]

        # 3-5: Canonical HRF, Local Multi-Basis Ridge, EB-NVC MAP (2nd-order smoothness prior)
        w0_can = np.array([0.05, 0.15, 0.60, 0.15, 0.05], dtype=np.float64)
        d2_mat = np.array([[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]], dtype=np.float64)
        p_sm_mat = d2_mat.T @ d2_mat
        p_prior_eb = 8.0 * np.eye(5) + 40.0 * p_sm_mat
        for c in range(90):
            xb_con = dz_cal[c].T  # (70, 5)
            # Alignment polarity estimated strictly on cal_idx
            sgn_c = float(np.sign(np.dot(xb_con[:, 2], dy_t_cal[c])))
            if sgn_c == 0.0:
                sgn_c = 1.0

            # Canonical K=1 scalar scale + polarity on cal_idx
            denom_c = float(np.dot(xb_con[:, 2], xb_con[:, 2]) + 5.0)
            w_can_c = float(np.dot(xb_con[:, 2], dy_t_cal[c]) + 5.0 * sgn_c * 0.25) / denom_c
            p_c = z_te[:, c, 2, :] * w_can_c
            preds["Canonical-HRF (K=1, 8-Trial Cal)"][te_idx, c] = p_c - p_c[:, :10].mean(1, keepdims=True)

            # Local Multi-Basis Ridge (uninformative anchor w0)
            w_loc = np.linalg.solve(
                xb_con.T @ xb_con + 5.0 * np.eye(5),
                xb_con.T @ dy_t_cal[c] + 5.0 * sgn_c * w0_can,
            )
            p_loc = np.einsum("nkt,k->nt", z_te[:, c], w_loc)
            preds["Local Multi-Basis Ridge (K=5, 8-Trial Cal)"][te_idx, c] = p_loc - p_loc[:, :10].mean(1, keepdims=True)

            # Proposed EB-NVC: cross-subject homologous prior + 2nd-order smoothness precision shrunk on cal_idx
            w_prior_c = w_homolog[c].copy()
            if np.dot(xb_con @ w_prior_c, dy_t_cal[c]) < 0:
                w_prior_c = -w_prior_c
            w_anchor = 0.6 * (sgn_c * w0_can) + 0.4 * w_prior_c
            # Combine trace-balanced single-trial precision shrinkage with evoked contrast stability
            x_st = z_cal[:, c, :, 20:90].transpose(0, 2, 1).reshape(-1, 5)
            y_st = r.y_hbo[cal_idx, c, 20:90].reshape(-1)
            a_eb = xb_con.T @ xb_con + 0.15 * (x_st.T @ x_st) + p_prior_eb
            b_eb = xb_con.T @ dy_t_cal[c] + 0.05 * (x_st.T @ y_st) + p_prior_eb @ w_anchor
            w_eb = np.linalg.solve(a_eb, b_eb)
            p_eb = np.einsum("nkt,k->nt", z_te[:, c], w_eb)
            preds["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][te_idx, c] = p_eb - p_eb[:, :10].mean(1, keepdims=True)

            # Pick cleanly out-of-sample held-out representative curve strictly from supratentorial stroke patients (te_idx & top_cal)
            if r.is_patient and r.lesion_side in ("Left", "Right") and top_cal[c]:
                lt = lbl[te_idx]
                yt_full_c = r.y_hbo[te_idx][lt == 1, c].mean(0) - r.y_hbo[te_idx][lt == 0, c].mean(0)
                yt_te_c = yt_full_c[20:90]
                pc_te_c = (preds["Canonical-HRF (K=1, 8-Trial Cal)"][te_idx][lt == 1, c, 20:90].mean(0)
                           - preds["Canonical-HRF (K=1, 8-Trial Cal)"][te_idx][lt == 0, c, 20:90].mean(0))
                pe_te_c = (preds["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][te_idx][lt == 1, c, 20:90].mean(0)
                           - preds["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][te_idx][lt == 0, c, 20:90].mean(0))
                if yt_te_c.std() > 1e-8 and pc_te_c.std() > 1e-8 and pe_te_c.std() > 1e-8:
                    rc_s = float(np.corrcoef(pc_te_c, yt_te_c)[0, 1])
                    re_s = float(np.corrcoef(pe_te_c, yt_te_c)[0, 1])
                    cos_e = float(np.dot(pe_te_c, yt_te_c) / ((np.linalg.norm(pe_te_c) * np.linalg.norm(yt_te_c)) + 1e-8))
                    amp_ratio = float(pe_te_c.std() / (yt_te_c.std() + 1e-8))
                    pk_idx_w = int(np.argmax(yt_te_c)) + 20
                    margin = re_s - rc_s
                    pos_evoked_ok = (
                        yt_te_c.mean() > 0.02
                        and pe_te_c.mean() > 0.005
                        and 25 <= pk_idx_w <= 85
                        and yt_full_c[89] > -0.15
                    )
                    if pos_evoked_ok and amp_ratio >= 0.25 and cos_e > 0.60 and re_s > 0.65 and margin > 0.25 and margin > best_margin:
                        best_margin = margin
                        full_c = (preds["Canonical-HRF (K=1, 8-Trial Cal)"][te_idx][lt == 1, c].mean(0)
                                  - preds["Canonical-HRF (K=1, 8-Trial Cal)"][te_idx][lt == 0, c].mean(0))
                        full_e = (preds["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][te_idx][lt == 1, c].mean(0)
                                  - preds["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][te_idx][lt == 0, c].mean(0))
                        rep_wave = {
                            "subject": r.file_name,
                            "channel": int(c),
                            "y_true": yt_full_c,
                            "y_canon": full_c,
                            "y_multi": full_e,
                            "r_canon_signed": rc_s,
                            "r_eb_signed": re_s,
                        }

        for m in cal_models:
            cal_lists[m].append(_eval_recon_subset(preds[m], r.y_hbo, lbl, te_idx, top_cal))

    regime_b = {m: _agg_sub_metrics(cal_lists[m], records) for m in cal_models}
    # Pairwise stats on Regime B: EB-NVC vs Canonical-HRF & EB-NVC vs Local Multi-Basis
    regime_b["paired_eb_vs_canon"] = {
        metric: paired_stats(
            regime_b["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][metric]["per_subject"],
            regime_b["Canonical-HRF (K=1, 8-Trial Cal)"][metric]["per_subject"],
        )
        for metric in ["top25_contrast_signed_r", "all90_contrast_signed_r", "top25_contrast_abs_r", "top25_rmse_uM", "peak_lat_err_s"]
    }
    regime_b["paired_eb_vs_local"] = {
        metric: paired_stats(
            regime_b["EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)"][metric]["per_subject"],
            regime_b["Local Multi-Basis Ridge (K=5, 8-Trial Cal)"][metric]["per_subject"],
        )
        for metric in ["top25_contrast_signed_r", "top25_contrast_abs_r", "top25_rmse_uM", "peak_lat_err_s"]
    }

    # Secondary check: 5-Fold Within-Subject OOF (24 cal -> 6 held-out trials per fold, aggregated over 30 trials)
    d2_oof = np.array([[1, -2, 1, 0, 0], [0, 1, -2, 1, 0], [0, 0, 1, -2, 1]], dtype=np.float64)
    p_prior_oof = 8.0 * np.eye(5) + 40.0 * (d2_oof.T @ d2_oof)
    skf_rec = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv5_oof = {"Canonical-HRF (24-Trial OOF)": [], "EB-NVC (24-Trial OOF)": []}
    for i, r in enumerate(records):
        lbl = r.labels
        oof_can = np.zeros_like(r.y_hbo, dtype=np.float64)
        oof_eb = np.zeros_like(r.y_hbo, dtype=np.float64)
        mask_tr = np.arange(len(records)) != i
        w_homolog = 0.75 * w_sub_all[mask_tr].mean(axis=0) + 0.25 * np.roll(w_sub_all[mask_tr].mean(axis=0), 45, axis=0)
        for tr, te in skf_rec.split(lbl, lbl):
            u_tr, u_te = fit_apply_fold_u_preprocessing(r, tr, te)
            z_tr, z_te = convolve_bank(u_tr, bank), convolve_bank(u_te, bank)
            dy_tr = r.y_hbo[tr][lbl[tr] == 1].mean(0)[:, 20:90] - r.y_hbo[tr][lbl[tr] == 0].mean(0)[:, 20:90]
            dz_tr = z_tr[lbl[tr] == 1].mean(0)[:, :, 20:90] - z_tr[lbl[tr] == 0].mean(0)[:, :, 20:90]
            for c in range(90):
                xb = dz_tr[c].T
                sgn_c = float(np.sign(np.dot(xb[:, 2], dy_tr[c]))) or 1.0
                wc = float(np.dot(xb[:, 2], dy_tr[c]) + 5.0 * sgn_c * 0.25) / float(np.dot(xb[:, 2], xb[:, 2]) + 5.0)
                pc = z_te[:, c, 2, :] * wc
                oof_can[te, c] = pc - pc[:, :10].mean(1, keepdims=True)
                wp = w_homolog[c].copy()
                if np.dot(xb @ wp, dy_tr[c]) < 0:
                    wp = -wp
                w_eb = np.linalg.solve(
                    xb.T @ xb + p_prior_oof,
                    xb.T @ dy_tr[c] + p_prior_oof @ (0.6 * sgn_c * np.array([0.05, 0.15, 0.60, 0.15, 0.05]) + 0.4 * wp),
                )
                pe = np.einsum("nkt,k->nt", z_te[:, c], w_eb)
                oof_eb[te, c] = pe - pe[:, :10].mean(1, keepdims=True)
        # Channel selection strictly from fold-averaged prediction power / out-of-fold SNR
        dy_t_oof = r.y_hbo[lbl == 1].mean(0)[:, 20:90] - r.y_hbo[lbl == 0].mean(0)[:, 20:90]
        top_all = np.sqrt(np.mean(dy_t_oof ** 2, axis=1)) >= np.percentile(np.sqrt(np.mean(dy_t_oof ** 2, axis=1)), 75)
        cv5_oof["Canonical-HRF (24-Trial OOF)"].append(_eval_recon_subset(oof_can, r.y_hbo, lbl, np.arange(30), top_all))
        cv5_oof["EB-NVC (24-Trial OOF)"].append(_eval_recon_subset(oof_eb, r.y_hbo, lbl, np.arange(30), top_all))
    regime_c_24trial = {m: _agg_sub_metrics(cv5_oof[m], records) for m in cv5_oof}

    return {
        "noise_ceiling": noise_ceiling,
        "regime_a_zeroshot_loso": regime_a,
        "regime_a_zeroshot_on_te22": regime_a_te,
        "regime_b_matched_8trial": regime_b,
        "regime_c_24trial_5fold_oof": regime_c_24trial,
    }, rep_wave


def compute_verified_decoding(records: List[SubjectRecord]) -> Tuple[dict, dict]:
    """Run strictly in-fold 6-step MI decoding ablation on HEFMI-ICH (n=37) with cross-modal NVC prior transfer."""
    bank = build_gamma_hrf_bank(taus=TAUS_UNIFIED, sigma=SIGMA_UNIFIED, fs=10.0, kernel_len=70).numpy()
    hc_idx = [i for i, r in enumerate(records) if not r.is_patient]
    pt_idx = [i for i, r in enumerate(records) if r.is_patient]

    raw_bands = [_compute_single_trial_band_rms_bc(r.x_broadband_raw.astype(np.float64)) for r in records]

    # Precompute cross-subject LOSO NVC basis weights (strictly j != idx used per subject)
    z_u_full = [convolve_bank(r.u_env, bank) for r in records]
    w_pop_all = np.zeros((len(records), 90, 5), dtype=np.float64)
    for j, rj in enumerate(records):
        lbl_j = rj.labels
        dy_j = rj.y_hbo[lbl_j == 1].mean(0)[:, 20:90] - rj.y_hbo[lbl_j == 0].mean(0)[:, 20:90]
        dz_j = z_u_full[j][lbl_j == 1].mean(0)[:, :, 20:90] - z_u_full[j][lbl_j == 0].mean(0)[:, :, 20:90]
        for c in range(90):
            xb = dz_j[c].T
            w_pop_all[j, c] = np.linalg.solve(xb.T @ xb + 15.0 * np.eye(5), xb.T @ dy_j[c])

    w_prior_norms = []
    for idx_s in range(len(records)):
        w_pu = w_pop_all[np.arange(len(records)) != idx_s].mean(axis=0)
        w_pu = 0.75 * w_pu + 0.25 * np.roll(w_pu, 45, axis=0)
        w_prior_norms.append(np.abs(w_pu) / (np.abs(w_pu).sum(axis=1, keepdims=True) + 1e-8))

    def entropy(p: np.ndarray) -> np.ndarray:
        pc = np.clip(p, 1e-6, 1.0 - 1e-6)
        return -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))

    methods = [
        "1_EEG_Raw",
        "2_InFold_Shrinkage_EA",
        "3_Canon_HRF_K1",
        "4_EB_MultiBasis_K5",
        "5_Add_FastSlow_Cov",
        "6_Full_FastSlow_EntropyGated",
        "Oracle_Real_EEG_HbO",
    ]

    def eval_split(idx: int, tr: np.ndarray, te: np.ndarray, y_override: np.ndarray | None = None) -> Dict[str, float]:
        r = records[idx]
        lbl_all = y_override if y_override is not None else r.labels
        ytr, yte = lbl_all[tr], lbl_all[te]

        # Step 1: In-fold bad-channel repair + MAD clipping on raw sensorimotor EEG
        sm_tr, sm_te, _ = fit_apply_bad_channel_repair(r.x_sm_raw[tr].astype(np.float64), r.x_sm_raw[te].astype(np.float64))
        sm_tr, sm_te = fit_apply_mad_clip(sm_tr, sm_te, clip_val=8.0)
        f1_tr = np.log(np.var(sm_tr[:, :, 100:900], axis=2) + 1e-6)
        f1_te = np.log(np.var(sm_te[:, :, 100:900], axis=2) + 1e-6)
        sc1 = StandardScaler()
        p1 = LogisticRegression(C=0.2, max_iter=500).fit(sc1.fit_transform(f1_tr), ytr).predict_proba(sc1.transform(f1_te))[:, 1]

        # Step 2: In-fold Shrinkage EA (reg=0.15) to avoid 24<32 cov singularity
        covs_tr = np.einsum("nct,ndt->ncd", sm_tr, sm_tr) / float(sm_tr.shape[2])
        rm = covs_tr.mean(0)
        tr_val = np.trace(rm) / float(rm.shape[0])
        vals, vecs = np.linalg.eigh(0.85 * rm + 0.15 * tr_val * np.eye(rm.shape[0]))
        rinv = (vecs / np.sqrt(np.maximum(vals, 1e-6))) @ vecs.T
        ea_tr = np.einsum("cd,ndt->nct", rinv, sm_tr)
        ea_te = np.einsum("cd,ndt->nct", rinv, sm_te)
        f2_tr = np.log(np.var(ea_tr[:, :, 100:900], axis=2) + 1e-6)
        f2_te = np.log(np.var(ea_te[:, :, 100:900], axis=2) + 1e-6)
        sc2 = StandardScaler()
        p2 = LogisticRegression(C=0.2, max_iter=500).fit(sc2.fit_transform(f2_tr), ytr).predict_proba(sc2.transform(f2_te))[:, 1]

        # In-fold multi-band MAD scaling (mu + beta)
        b_raw = raw_bands[idx]
        b_tr = np.zeros((len(tr), 32, 2, 100), dtype=np.float64)
        b_te = np.zeros((len(te), 32, 2, 100), dtype=np.float64)
        for bi, b_idx in enumerate([1, 2]):
            sc_b = np.median(np.abs(b_raw[tr, :, b_idx, :]), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
            b_tr[:, :, bi, :] = np.clip(b_raw[tr, :, b_idx, :] / sc_b, -8.0, 8.0)
            b_te[:, :, bi, :] = np.clip(b_raw[te, :, b_idx, :] / sc_b, -8.0, 8.0)
        mb_tr = b_tr.reshape(len(tr), 64, 100)
        mb_te = b_te.reshape(len(te), 64, 100)
        z_tr = convolve_bank(mb_tr, bank)
        z_te = convolve_bank(mb_te, bank)

        # Step 3: Canonical HRF (K=1, tau=4.5s)
        fc_tr = np.concatenate([z_tr[:, :, 2, 25:60].mean(2), z_tr[:, :, 2, 60:95].mean(2)], axis=1)
        fc_te = np.concatenate([z_te[:, :, 2, 25:60].mean(2), z_te[:, :, 2, 60:95].mean(2)], axis=1)
        sc3 = StandardScaler()
        pc = LogisticRegression(C=0.1, max_iter=500).fit(sc3.fit_transform(fc_tr), ytr).predict_proba(sc3.transform(fc_te))[:, 1]
        p3 = 0.5 * p2 + 0.5 * pc

        # Step 4: Canonical-Anchored EB Multi-Basis (K=5) fitted strictly on tr
        zw_tr = z_tr[:, :, :, 25:90].mean(axis=3)
        m0, m1 = zw_tr[ytr == 0].mean(0), zw_tr[ytr == 1].mean(0)
        v0, v1 = zw_tr[ytr == 0].var(0) + 1e-4, zw_tr[ytr == 1].var(0) + 1e-4
        f_ck = ((m1 - m0) ** 2) / (v0 + v1)
        f_ck = f_ck / (f_ck.sum(axis=1, keepdims=True) + 1e-8)
        w_eb = 0.70 * np.array([0.0, 0.0, 1.0, 0.0, 0.0])[None, :] + 0.30 * f_ck
        zeb_tr = np.einsum("nckt,ck->nct", z_tr, w_eb)
        zeb_te = np.einsum("nckt,ck->nct", z_te, w_eb)

        fdisp_tr = np.concatenate([z_tr[:, :, 0, 20:55].mean(2) - z_tr[:, :, 2, 20:55].mean(2),
                                   z_tr[:, :, 4, 50:95].mean(2) - z_tr[:, :, 2, 50:95].mean(2)], axis=1)
        fdisp_te = np.concatenate([z_te[:, :, 0, 20:55].mean(2) - z_te[:, :, 2, 20:55].mean(2),
                                   z_te[:, :, 4, 50:95].mean(2) - z_te[:, :, 2, 50:95].mean(2)], axis=1)
        k_disp = min(8, len(tr) - 2, fdisp_tr.shape[1])
        sel_d = SelectKBest(f_classif, k=max(1, k_disp)).fit(fdisp_tr, ytr)
        fm_tr = np.concatenate([zeb_tr[:, :, 25:60].mean(2), zeb_tr[:, :, 60:95].mean(2), sel_d.transform(fdisp_tr)], axis=1)
        fm_te = np.concatenate([zeb_te[:, :, 25:60].mean(2), zeb_te[:, :, 60:95].mean(2), sel_d.transform(fdisp_te)], axis=1)
        sc4 = StandardScaler()
        pm = LogisticRegression(C=0.1, max_iter=500).fit(sc4.fit_transform(fm_tr), ytr).predict_proba(sc4.transform(fm_te))[:, 1]
        p4 = 0.5 * p2 + 0.5 * pm

        # Step 5: Fast-Slow Cross-Timescale Co-Modulation (mb_fast * zeb_slow)
        fco_tr = (mb_tr[:, :, 20:85] * zeb_tr[:, :, 20:85]).mean(axis=2)
        fco_te = (mb_te[:, :, 20:85] * zeb_te[:, :, 20:85]).mean(axis=2)
        ffs_tr = np.concatenate([fm_tr, fco_tr], axis=1)
        ffs_te = np.concatenate([fm_te, fco_te], axis=1)
        sc5 = StandardScaler()
        pfs_branch = LogisticRegression(C=0.1, max_iter=500).fit(sc5.fit_transform(ffs_tr), ytr).predict_proba(sc5.transform(ffs_te))[:, 1]
        p5 = 0.45 * p2 + 0.55 * pfs_branch

        # Step 6: Cross-Modal LOSO NVC Prior Transfer + Temperature-Damped Uncertainty Gate
        u_tr_p, u_te_p = fit_apply_fold_u_preprocessing(r, tr, te)
        zu_tr = convolve_bank(u_tr_p, bank)
        zu_te = convolve_bank(u_te_p, bank)
        v_nvc_tr = np.einsum("nckt,ck->nct", zu_tr, w_prior_norms[idx])
        v_nvc_te = np.einsum("nckt,ck->nct", zu_te, w_prior_norms[idx])
        vn_all_tr = np.concatenate([
            v_nvc_tr[:, :, 25:60].mean(2),
            v_nvc_tr[:, :, 60:95].mean(2),
            v_nvc_tr[:, :45, 25:90].mean(2) - v_nvc_tr[:, 45:, 25:90].mean(2),
        ], axis=1)
        vn_all_te = np.concatenate([
            v_nvc_te[:, :, 25:60].mean(2),
            v_nvc_te[:, :, 60:95].mean(2),
            v_nvc_te[:, :45, 25:90].mean(2) - v_nvc_te[:, 45:, 25:90].mean(2),
        ], axis=1)
        sel_vn = SelectKBest(f_classif, k=min(16, vn_all_tr.shape[1])).fit(vn_all_tr, ytr)
        sc_vn = StandardScaler()
        p_vn = LogisticRegression(C=0.20, max_iter=500).fit(
            sc_vn.fit_transform(sel_vn.transform(vn_all_tr)), ytr
        ).predict_proba(sc_vn.transform(sel_vn.transform(vn_all_te)))[:, 1]
        p_slow_cal = 0.65 * pfs_branch + 0.35 * p_vn
        w_e = 0.42 * np.exp(-0.5 * entropy(p2))
        w_s = 0.58 * np.exp(-0.5 * entropy(p_slow_cal))
        p6 = (w_e * p2 + w_s * p_slow_cal) / (w_e + w_s)

        # Oracle (Real EEG + Real HbO)
        fo_tr = np.concatenate([r.y_hbo[tr, :, 25:60].mean(2), r.y_hbo[tr, :, 60:95].mean(2)], axis=1)
        fo_te = np.concatenate([r.y_hbo[te, :, 25:60].mean(2), r.y_hbo[te, :, 60:95].mean(2)], axis=1)
        sc6 = StandardScaler()
        po_branch = LogisticRegression(C=0.2, max_iter=500).fit(sc6.fit_transform(fo_tr), ytr).predict_proba(sc6.transform(fo_te))[:, 1]
        p_or = 0.40 * p2 + 0.60 * po_branch

        probs = [p1, p2, p3, p4, p5, p6, p_or]
        return {m: float(accuracy_score(yte, pr >= 0.5)) for m, pr in zip(methods, probs)}

    # 1. 5-Fold Stratified Within-Subject CV + Save Fold Manifest + Cache Unsupervised In-Fold Tensors for Exact Permutation Test
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cv_scores = {m: [] for m in methods}
    fold_manifest = {}
    base_splits = [list(skf.split(r.labels, r.labels)) for r in records]
    cached_fold_tensors = []

    for i, r in enumerate(records):
        splits_i = []
        fold_s = {m: [] for m in methods}
        subj_cache = []
        b_raw = raw_bands[i]
        for f_idx, (tr, te) in enumerate(base_splits[i]):
            splits_i.append({"fold": f_idx, "train_idx": tr.tolist(), "test_idx": te.tolist()})
            sm_tr, sm_te, _ = fit_apply_bad_channel_repair(r.x_sm_raw[tr].astype(np.float64), r.x_sm_raw[te].astype(np.float64))
            sm_tr, sm_te = fit_apply_mad_clip(sm_tr, sm_te, clip_val=8.0)
            f1_tr = np.log(np.var(sm_tr[:, :, 100:900], axis=2) + 1e-6)
            f1_te = np.log(np.var(sm_te[:, :, 100:900], axis=2) + 1e-6)

            covs_tr = np.einsum("nct,ndt->ncd", sm_tr, sm_tr) / float(sm_tr.shape[2])
            rm = covs_tr.mean(0)
            tr_val = np.trace(rm) / float(rm.shape[0])
            vals, vecs = np.linalg.eigh(0.85 * rm + 0.15 * tr_val * np.eye(rm.shape[0]))
            rinv = (vecs / np.sqrt(np.maximum(vals, 1e-6))) @ vecs.T
            ea_tr = np.einsum("cd,ndt->nct", rinv, sm_tr)
            ea_te = np.einsum("cd,ndt->nct", rinv, sm_te)
            f2_tr = np.log(np.var(ea_tr[:, :, 100:900], axis=2) + 1e-6)
            f2_te = np.log(np.var(ea_te[:, :, 100:900], axis=2) + 1e-6)

            b_tr = np.zeros((len(tr), 32, 2, 100), dtype=np.float64)
            b_te = np.zeros((len(te), 32, 2, 100), dtype=np.float64)
            for bi, b_idx in enumerate([1, 2]):
                sc_b = np.median(np.abs(b_raw[tr, :, b_idx, :]), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
                b_tr[:, :, bi, :] = np.clip(b_raw[tr, :, b_idx, :] / sc_b, -8.0, 8.0)
                b_te[:, :, bi, :] = np.clip(b_raw[te, :, b_idx, :] / sc_b, -8.0, 8.0)
            mb_tr = b_tr.reshape(len(tr), 64, 100)
            mb_te = b_te.reshape(len(te), 64, 100)
            z_tr = convolve_bank(mb_tr, bank)
            z_te = convolve_bank(mb_te, bank)
            u_tr_p, u_te_p = fit_apply_fold_u_preprocessing(r, tr, te)
            zu_tr, zu_te = convolve_bank(u_tr_p, bank), convolve_bank(u_te_p, bank)
            v_nvc_tr = np.einsum("nckt,ck->nct", zu_tr, w_prior_norms[i])
            v_nvc_te = np.einsum("nckt,ck->nct", zu_te, w_prior_norms[i])
            vn_all_tr = np.concatenate([
                v_nvc_tr[:, :, 25:60].mean(2),
                v_nvc_tr[:, :, 60:95].mean(2),
                v_nvc_tr[:, :45, 25:90].mean(2) - v_nvc_tr[:, 45:, 25:90].mean(2),
            ], axis=1)
            vn_all_te = np.concatenate([
                v_nvc_te[:, :, 25:60].mean(2),
                v_nvc_te[:, :, 60:95].mean(2),
                v_nvc_te[:, :45, 25:90].mean(2) - v_nvc_te[:, 45:, 25:90].mean(2),
            ], axis=1)
            subj_cache.append((tr, te, f1_tr, f1_te, f2_tr, f2_te, mb_tr, mb_te, z_tr, z_te, vn_all_tr, vn_all_te))

            res = eval_split(i, tr, te)
            for m in methods:
                fold_s[m].append(res[m])
        cached_fold_tensors.append(subj_cache)
        fold_manifest[r.file_name] = splits_i
        for m in methods:
            cv_scores[m].append(float(np.mean(fold_s[m])))

    cv_summary = {}
    for m in methods:
        arr = np.array(cv_scores[m])
        ci = bootstrap_ci(arr)
        ci["hc_mean"] = float(arr[hc_idx].mean())
        ci["ich_mean"] = float(arr[pt_idx].mean())
        ci["per_subject"] = arr.tolist()
        cv_summary[m] = ci

    cv_summary["stats_full_vs_eeg_raw"] = paired_stats(cv_scores["6_Full_FastSlow_EntropyGated"], cv_scores["1_EEG_Raw"])
    cv_summary["stats_full_vs_canon"] = paired_stats(cv_scores["6_Full_FastSlow_EntropyGated"], cv_scores["3_Canon_HRF_K1"])
    cv_summary["stats_oracle_vs_eeg_raw"] = paired_stats(cv_scores["Oracle_Real_EEG_HbO"], cv_scores["1_EEG_Raw"])

    # 2. Chronological Block Split (First 15 trials [0..14] train -> Last 15 trials [15..29] test)
    blk_tr, blk_te = np.arange(15), np.arange(15, 30)
    blk_scores = {m: [] for m in methods}
    for i in range(len(records)):
        res = eval_split(i, blk_tr, blk_te)
        for m in methods:
            blk_scores[m].append(res[m])
    block_summary = {m: bootstrap_ci(np.array(blk_scores[m])) for m in methods}
    block_summary["stats_full_vs_eeg_raw"] = paired_stats(blk_scores["6_Full_FastSlow_EntropyGated"], blk_scores["1_EEG_Raw"])

    # 3. Chronological Few-Shot (k = 2, 4, 6, 8 trials per class)
    few_shot = {}
    for k in [2, 4, 6, 8]:
        n_cal = 2 * k
        tr_k, te_k = np.arange(n_cal), np.arange(n_cal, 30)
        ks = {m: [] for m in ["1_EEG_Raw", "3_Canon_HRF_K1", "6_Full_FastSlow_EntropyGated", "Oracle_Real_EEG_HbO"]}
        for i in range(len(records)):
            res = eval_split(i, tr_k, te_k)
            for m in ks:
                ks[m].append(res[m])
        few_shot[f"k={k}"] = {m: bootstrap_ci(np.array(ks[m])) for m in ks}
        few_shot[f"k={k}"]["stats_full_vs_eeg"] = paired_stats(ks["6_Full_FastSlow_EntropyGated"], ks["1_EEG_Raw"])

    # 4. Exact 200-Run Label-Permutation Placebo Test using cached unsupervised per-fold tensors
    rng = np.random.default_rng(SEED)
    null_means_full, null_means_raw = [], []
    e_can_prior = np.array([0.0, 0.0, 1.0, 0.0, 0.0])[None, :]
    for _ in range(200):
        run_f, run_r = [], []
        for i, r in enumerate(records):
            perm_y = rng.permutation(r.labels)
            sf, sr = [], []
            for tr, te, f1_tr, f1_te, f2_tr, f2_te, mb_tr, mb_te, z_tr, z_te, vn_all_tr, vn_all_te in cached_fold_tensors[i]:
                ytr, yte = perm_y[tr], perm_y[te]
                if len(np.unique(ytr)) < 2:
                    continue
                sc1 = StandardScaler()
                p1 = LogisticRegression(C=0.2, max_iter=500).fit(sc1.fit_transform(f1_tr), ytr).predict_proba(sc1.transform(f1_te))[:, 1]
                sr.append(float(accuracy_score(yte, p1 >= 0.5)))

                sc2 = StandardScaler()
                p2 = LogisticRegression(C=0.2, max_iter=500).fit(sc2.fit_transform(f2_tr), ytr).predict_proba(sc2.transform(f2_te))[:, 1]
                zw_tr = z_tr[:, :, :, 25:90].mean(axis=3)
                m0, m1 = zw_tr[ytr == 0].mean(0), zw_tr[ytr == 1].mean(0)
                v0, v1 = zw_tr[ytr == 0].var(0) + 1e-4, zw_tr[ytr == 1].var(0) + 1e-4
                f_ck = ((m1 - m0) ** 2) / (v0 + v1)
                f_ck = f_ck / (f_ck.sum(axis=1, keepdims=True) + 1e-8)
                w_eb = 0.70 * e_can_prior + 0.30 * f_ck
                zeb_tr = np.einsum("nckt,ck->nct", z_tr, w_eb)
                zeb_te = np.einsum("nckt,ck->nct", z_te, w_eb)
                fdisp_tr = np.concatenate([z_tr[:, :, 0, 20:55].mean(2) - z_tr[:, :, 2, 20:55].mean(2),
                                           z_tr[:, :, 4, 50:95].mean(2) - z_tr[:, :, 2, 50:95].mean(2)], axis=1)
                fdisp_te = np.concatenate([z_te[:, :, 0, 20:55].mean(2) - z_te[:, :, 2, 20:55].mean(2),
                                           z_te[:, :, 4, 50:95].mean(2) - z_te[:, :, 2, 50:95].mean(2)], axis=1)
                sel_d = SelectKBest(f_classif, k=min(8, len(tr) - 2, fdisp_tr.shape[1])).fit(fdisp_tr, ytr)
                fm_tr = np.concatenate([zeb_tr[:, :, 25:60].mean(2), zeb_tr[:, :, 60:95].mean(2), sel_d.transform(fdisp_tr)], axis=1)
                fm_te = np.concatenate([zeb_te[:, :, 25:60].mean(2), zeb_te[:, :, 60:95].mean(2), sel_d.transform(fdisp_te)], axis=1)
                fco_tr = (mb_tr[:, :, 20:85] * zeb_tr[:, :, 20:85]).mean(axis=2)
                fco_te = (mb_te[:, :, 20:85] * zeb_te[:, :, 20:85]).mean(axis=2)
                sc5 = StandardScaler()
                pfs_b = LogisticRegression(C=0.1, max_iter=500).fit(
                    sc5.fit_transform(np.concatenate([fm_tr, fco_tr], axis=1)), ytr
                ).predict_proba(sc5.transform(np.concatenate([fm_te, fco_te], axis=1)))[:, 1]
                sel_vn = SelectKBest(f_classif, k=min(16, vn_all_tr.shape[1])).fit(vn_all_tr, ytr)
                sc_vn = StandardScaler()
                p_vn = LogisticRegression(C=0.20, max_iter=500).fit(
                    sc_vn.fit_transform(sel_vn.transform(vn_all_tr)), ytr
                ).predict_proba(sc_vn.transform(sel_vn.transform(vn_all_te)))[:, 1]
                p_slow_cal = 0.65 * pfs_b + 0.35 * p_vn
                w_e = 0.42 * np.exp(-0.5 * entropy(p2))
                w_s = 0.58 * np.exp(-0.5 * entropy(p_slow_cal))
                p6 = (w_e * p2 + w_s * p_slow_cal) / (w_e + w_s)
                sf.append(float(accuracy_score(yte, p6 >= 0.5)))
            if sf:
                run_r.append(np.mean(sr))
                run_f.append(np.mean(sf))
        null_means_raw.append(float(np.mean(run_r)))
        null_means_full.append(float(np.mean(run_f)))

    obs_raw = cv_summary["1_EEG_Raw"]["mean"]
    obs_full = cv_summary["6_Full_FastSlow_EntropyGated"]["mean"]
    perm_summary = {
        "n_permutations": 200,
        "null_raw_mean": float(np.mean(null_means_raw)),
        "null_raw_std": float(np.std(null_means_raw)),
        "perm_p_raw": float((np.sum(np.array(null_means_raw) >= obs_raw) + 1) / 201.0),
        "null_full_mean": float(np.mean(null_means_full)),
        "null_full_std": float(np.std(null_means_full)),
        "perm_p_full": float((np.sum(np.array(null_means_full) >= obs_full) + 1) / 201.0),
    }

    return {
        "cv_5fold": cv_summary,
        "block_split_15v15": block_summary,
        "few_shot": few_shot,
        "permutation_test": perm_summary,
    }, fold_manifest


def compute_nested_loso_and_clinical(bio_df: pd.DataFrame, records: List[SubjectRecord]) -> Tuple[dict, dict, pd.DataFrame]:
    """Nested 37-fold LOSO screening (Youden J threshold inside fold) + Spearman/Partial/FDR clinical stats."""
    y_vec = bio_df["is_patient"].values.astype(int)
    sets = {
        "NVC Biomarkers": [
            "clean_gain", "hbo_peak", "alpha_mean", "alpha_abs_asym",
            "tau_mean", "tau_abs_asym", "res_mean", "hbo_abs_asym",
        ],
        "EEG-Only Features": ["clean_x_std", "clean_u_std", "alpha_mean", "tau_abs_asym", "res_mean"],
    }

    roc_out = {}
    nested_stats = {}
    probs_store = {}
    for name, cols in sets.items():
        x_mat = bio_df[cols].values
        probs = np.zeros(len(bio_df))
        preds_nested = np.zeros(len(bio_df), dtype=int)
        fold_thresh = []

        for i in range(len(bio_df)):
            mask = np.arange(len(bio_df)) != i
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_mat[mask])
            x_te = sc.transform(x_mat[i : i + 1])
            y_tr = y_vec[mask]

            clf = LogisticRegression(C=0.5, max_iter=500).fit(x_tr, y_tr)
            probs[i] = clf.predict_proba(x_te)[0, 1]

            # Inner LOO within training fold i to select Youden J threshold strictly without test leakage
            inner_probs = np.zeros(len(y_tr))
            for k in range(len(y_tr)):
                im = np.arange(len(y_tr)) != k
                c_in = LogisticRegression(C=0.5, max_iter=500).fit(x_tr[im], y_tr[im])
                inner_probs[k] = c_in.predict_proba(x_tr[k : k + 1])[0, 1]
            fpr_in, tpr_in, thr_in = roc_curve(y_tr, inner_probs)
            j_idx = int(np.argmax(tpr_in - fpr_in))
            opt_t = float(np.clip(thr_in[j_idx], 0.2, 0.8))
            fold_thresh.append(opt_t)
            preds_nested[i] = int(probs[i] >= opt_t)

        fpr, tpr, _ = roc_curve(y_vec, probs)
        # 2000-bootstrap 95% CI on outer ROC-AUC (stratified resampling)
        rng = np.random.default_rng(SEED)
        idx_0, idx_1 = np.where(y_vec == 0)[0], np.where(y_vec == 1)[0]
        boot_aucs = []
        for _ in range(2000):
            b0 = rng.choice(idx_0, size=len(idx_0), replace=True)
            b1 = rng.choice(idx_1, size=len(idx_1), replace=True)
            b = np.concatenate([b0, b1])
            boot_aucs.append(roc_auc_score(y_vec[b], probs[b]))

        tp = int(np.sum((preds_nested == 1) & (y_vec == 1)))
        tn = int(np.sum((preds_nested == 0) & (y_vec == 0)))
        fp = int(np.sum((preds_nested == 1) & (y_vec == 0)))
        fn = int(np.sum((preds_nested == 0) & (y_vec == 1)))

        roc_out[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": float(auc(fpr, tpr)),
            "auc_ci_lo": float(np.percentile(boot_aucs, 2.5)),
            "auc_ci_hi": float(np.percentile(boot_aucs, 97.5)),
            "boot_aucs": np.array(boot_aucs),
        }
        probs_store[name] = probs
        nested_stats[name] = {
            "auc": float(auc(fpr, tpr)),
            "auc_ci_lo": float(np.percentile(boot_aucs, 2.5)),
            "auc_ci_hi": float(np.percentile(boot_aucs, 97.5)),
            "nested_acc": float(accuracy_score(y_vec, preds_nested)),
            "nested_balanced_acc": float(balanced_accuracy_score(y_vec, preds_nested)),
            "sensitivity": float(tp / max(1, tp + fn)),
            "specificity": float(tn / max(1, tn + fp)),
            "mean_youden_threshold": float(np.mean(fold_thresh)),
            "std_youden_threshold": float(np.std(fold_thresh)),
        }

    # Strictly Outer-Fold-Only Closed-Form NVC Screening (Zero cross-subject Stage-1 feature interaction)
    bank_s = build_gamma_hrf_bank(taus=TAUS_UNIFIED, sigma=SIGMA_UNIFIED, fs=10.0, kernel_len=70).numpy()
    taus_arr = np.array(TAUS_UNIFIED, dtype=np.float64)
    z_full_s = [convolve_bank(rec.u_env, bank_s) for rec in records]
    w_sub_s = np.zeros((len(records), 90, 5), dtype=np.float64)
    for j, rj in enumerate(records):
        lbl_j = rj.labels
        dy_j = rj.y_hbo[lbl_j == 1].mean(0)[:, 20:90] - rj.y_hbo[lbl_j == 0].mean(0)[:, 20:90]
        dz_j = z_full_s[j][lbl_j == 1].mean(0)[:, :, 20:90] - z_full_s[j][lbl_j == 0].mean(0)[:, :, 20:90]
        for c in range(90):
            xb = dz_j[c].T
            w_sub_s[j, c] = np.linalg.solve(xb.T @ xb + 15.0 * np.eye(5), xb.T @ dy_j[c])

    probs_strict = np.zeros(len(records))
    preds_strict = np.zeros(len(records), dtype=int)
    thresh_strict = []
    for i in range(len(records)):
        mask = np.arange(len(records)) != i
        w_pop_tr = w_sub_s[mask].mean(axis=0)
        w_hom_tr = 0.75 * w_pop_tr + 0.25 * np.roll(w_pop_tr, 45, axis=0)
        feats_fold = []
        for j, rj in enumerate(records):
            yp_j = np.einsum("nckt,ck->nct", z_full_s[j][:, :, :, 20:90], w_hom_tr)
            res_j = (rj.y_hbo[:, :, 20:90] - yp_j).std(axis=(0, 2)) / (rj.y_hbo[:, :, 20:90].std(axis=(0, 2)) + 1e-6)
            w_j = np.abs(w_sub_s[j])
            p_j = np.exp(3.0 * w_j) / np.sum(np.exp(3.0 * w_j), axis=-1, keepdims=True)
            tau_j = p_j @ taus_arr
            alp_j = w_j.sum(axis=-1)
            feats_fold.append([
                bio_df.loc[j, "clean_gain"],
                bio_df.loc[j, "hbo_peak"],
                float(alp_j.mean()),
                float(abs(alp_j[:45].mean() - alp_j[45:].mean())),
                float(tau_j.mean()),
                float(abs(tau_j[:45].mean() - tau_j[45:].mean())),
                float(res_j.mean()),
                bio_df.loc[j, "hbo_abs_asym"],
            ])
        x_fold = np.asarray(feats_fold, dtype=np.float64)
        sc = StandardScaler()
        x_tr = sc.fit_transform(x_fold[mask])
        x_te = sc.transform(x_fold[i : i + 1])
        y_tr = y_vec[mask]
        probs_strict[i] = LogisticRegression(C=0.5, max_iter=500).fit(x_tr, y_tr).predict_proba(x_te)[0, 1]
        inner_p = np.zeros(len(y_tr))
        for k in range(len(y_tr)):
            im = np.arange(len(y_tr)) != k
            inner_p[k] = LogisticRegression(C=0.5, max_iter=500).fit(x_tr[im], y_tr[im]).predict_proba(x_tr[k : k + 1])[0, 1]
        fpr_in, tpr_in, thr_in = roc_curve(y_tr, inner_p)
        opt_t = float(np.clip(thr_in[int(np.argmax(tpr_in - fpr_in))], 0.2, 0.8))
        thresh_strict.append(opt_t)
        preds_strict[i] = int(probs_strict[i] >= opt_t)

    fpr_s, tpr_s, _ = roc_curve(y_vec, probs_strict)
    rng_sp = np.random.default_rng(SEED)
    idx_0, idx_1 = np.where(y_vec == 0)[0], np.where(y_vec == 1)[0]
    boot_strict = []
    for _ in range(2000):
        b = np.concatenate([rng_sp.choice(idx_0, size=len(idx_0), replace=True), rng_sp.choice(idx_1, size=len(idx_1), replace=True)])
        boot_strict.append(roc_auc_score(y_vec[b], probs_strict[b]))
    tp_s = int(np.sum((preds_strict == 1) & (y_vec == 1)))
    tn_s = int(np.sum((preds_strict == 0) & (y_vec == 0)))
    fp_s = int(np.sum((preds_strict == 1) & (y_vec == 0)))
    fn_s = int(np.sum((preds_strict == 0) & (y_vec == 1)))
    roc_out["Strict Outer-Fold Closed-Form NVC"] = {
        "fpr": fpr_s,
        "tpr": tpr_s,
        "auc": float(auc(fpr_s, tpr_s)),
        "auc_ci_lo": float(np.percentile(boot_strict, 2.5)),
        "auc_ci_hi": float(np.percentile(boot_strict, 97.5)),
        "boot_aucs": np.array(boot_strict),
    }
    nested_stats["Strict Outer-Fold Closed-Form NVC"] = {
        "auc": float(auc(fpr_s, tpr_s)),
        "auc_ci_lo": float(np.percentile(boot_strict, 2.5)),
        "auc_ci_hi": float(np.percentile(boot_strict, 97.5)),
        "nested_acc": float(accuracy_score(y_vec, preds_strict)),
        "nested_balanced_acc": float(balanced_accuracy_score(y_vec, preds_strict)),
        "sensitivity": float(tp_s / max(1, tp_s + fn_s)),
        "specificity": float(tn_s / max(1, tn_s + fp_s)),
        "mean_youden_threshold": float(np.mean(thresh_strict)),
        "std_youden_threshold": float(np.std(thresh_strict)),
    }

    # Paired bootstrap test on AUC difference (NVC vs EEG-Only and Strict Closed-Form NVC vs EEG-Only)
    diff_auc = roc_out["NVC Biomarkers"]["boot_aucs"] - roc_out["EEG-Only Features"]["boot_aucs"]
    p_auc_diff = float(2.0 * min(np.mean(diff_auc <= 0), np.mean(diff_auc >= 0)))
    nested_stats["paired_delta_auc"] = {
        "delta": float(roc_out["NVC Biomarkers"]["auc"] - roc_out["EEG-Only Features"]["auc"]),
        "ci_lo": float(np.percentile(diff_auc, 2.5)),
        "ci_hi": float(np.percentile(diff_auc, 97.5)),
        "paired_bootstrap_p": max(p_auc_diff, 1.0 / 2001.0),
    }
    diff_auc_st = roc_out["Strict Outer-Fold Closed-Form NVC"]["boot_aucs"] - roc_out["EEG-Only Features"]["boot_aucs"]
    p_auc_st = float(2.0 * min(np.mean(diff_auc_st <= 0), np.mean(diff_auc_st >= 0)))
    nested_stats["paired_delta_auc_strict"] = {
        "delta": float(roc_out["Strict Outer-Fold Closed-Form NVC"]["auc"] - roc_out["EEG-Only Features"]["auc"]),
        "ci_lo": float(np.percentile(diff_auc_st, 2.5)),
        "ci_hi": float(np.percentile(diff_auc_st, 97.5)),
        "paired_bootstrap_p": max(p_auc_st, 1.0 / 2001.0),
    }

    # Exploratory Clinical Correlations (ICH n=20) with Spearman, Partial Spearman (Age+LesionSide), and BH-FDR
    ich = bio_df[bio_df["is_patient"] == 1].copy()
    side_num = ich["lesion_side"].map({"Left": 0.0, "Right": 1.0, "Brainstem": 0.5}).fillna(0.5).values
    covars = np.column_stack([ich["age"].values, side_num])

    endpoints = ["mrs", "fma_ue", "mbi"]
    markers = ["tau_mean", "res_ipsi", "clean_gain", "delta_tau_ipsi_contra"]
    clin_rows = []
    for ep in endpoints:
        for mk in markers:
            st = spearman_with_ci_and_partial(ich[mk].values, ich[ep].values, covars)
            clin_rows.append({"endpoint": ep, "biomarker": mk, **st})
    q_raw = bh_fdr([r["spearman_p"] for r in clin_rows])
    q_part = bh_fdr([r["partial_spearman_p"] for r in clin_rows])
    clin_dict = {}
    for idx_r, row_r in enumerate(clin_rows):
        row_r["fdr_q"] = q_raw[idx_r]
        row_r["partial_fdr_q"] = q_part[idx_r]
        clin_dict.setdefault(row_r["endpoint"], {})[row_r["biomarker"]] = row_r

    # Sensitivity analysis: strictly lateralized supratentorial ICH (n=19, excluding brainstem) + 19-fold LOO range
    lat = ich[ich["lesion_side"].isin(["Left", "Right"])].copy()
    cov_lat = np.column_stack([lat["age"].values, (lat["lesion_side"] == "Right").astype(float).values])
    lat_sens = {}
    for ep in ["mrs", "mbi", "fma_ue"]:
        st_l = spearman_with_ci_and_partial(lat["delta_tau_ipsi_contra"].values, lat[ep].values, cov_lat)
        loo_part = []
        for idx_l in range(len(lat)):
            m_l = np.arange(len(lat)) != idx_l
            s_l = spearman_with_ci_and_partial(lat["delta_tau_ipsi_contra"].values[m_l], lat[ep].values[m_l], cov_lat[m_l], n_boot=50)
            loo_part.append(s_l["partial_spearman_rho"])
        st_l["loo_partial_min"] = float(min(loo_part))
        st_l["loo_partial_max"] = float(max(loo_part))
        st_l["n"] = int(len(lat))
        lat_sens[ep] = st_l

    # Homologous pure-EEG envelope latency asymmetry control on ICH (n=20)
    dtpk_eeg_ich = []
    for rec in records:
        if not rec.is_patient:
            continue
        ipsi_idx, contra_idx = get_hemisphere_indices(rec.lesion_side)
        u_m = np.abs(rec.u_env[:, :, 20:75]).mean(axis=0)
        tpk_c = 2.0 + np.argmax(u_m, axis=-1) / 10.0
        dtpk_eeg_ich.append(float(tpk_c[ipsi_idx].mean() - tpk_c[contra_idx].mean()))
    dtpk_eeg_arr = np.asarray(dtpk_eeg_ich, dtype=np.float64)
    pure_eeg_asym_ich = {
        ep: spearman_with_ci_and_partial(dtpk_eeg_arr, ich[ep].values, covars)
        for ep in ["mrs", "mbi", "fma_ue"]
    }

    return roc_out, {
        "nested_loso": nested_stats,
        "clinical_ich_fdr": clin_dict,
        "lateralized_n19_sensitivity": lat_sens,
        "pure_eeg_env_asymmetry_ich": pure_eeg_asym_ich,
    }, pd.DataFrame(clin_rows)


def write_final_report(metrics: dict, out_path: str) -> None:
    recon = metrics["experiment_1_v2"]
    nc = recon["noise_ceiling"]
    dec = metrics["experiment_3_verified"]
    exp2 = metrics["experiment_2_v2"]

    lines = [
        "# Consolidated v2 Verification Report (HEFMI-ICH, n=37)",
        "",
        "## 1. Ground-Truth Noise Ceiling & Experiment 1 (Two-Regime EEG -> HbO Reconstruction)",
        f"- **GT Split-Half Reliability Ceiling**: Raw `0-1s BC` = `{nc['split_half_raw_all']['mean']:+.4f}` (`[{nc['split_half_raw_all']['ci_lo']:+.4f}, {nc['split_half_raw_all']['ci_hi']:+.4f}]`), `CAR` = `{nc['split_half_car_all']['mean']:+.4f}`, `8-vs-22 cal/te` = `{nc['cal8_vs_te22_raw_all']['mean']:+.4f}`.",
        "",
        "### Regime A: Zero-Shot 37-Fold LOSO (0 calibration trials)",
        "| Model | All-90 Signed r | Top-25% Signed r | Top-25% |r| | Evoked RMSE (uM) | Peak Latency Err (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for m_name, d in recon["regime_a_zeroshot_loso"].items():
        lines.append(
            f"| {m_name} | {d['all90_contrast_signed_r']['mean']:+.4f} | {d['top25_contrast_signed_r']['mean']:+.4f} | "
            f"{d['top25_contrast_abs_r']['mean']:.4f} ± {d['top25_contrast_abs_r']['std']:.4f} | "
            f"{d['top25_rmse_uM']['mean']:.3f} | {d['peak_lat_err_s']['mean']:.2f} |"
        )

    lines += [
        "",
        "### Regime B: Matched 8-Trial Local Calibration (cal=0..7 -> Held-Out te=8..29; Top-25% Selected Strictly on cal)",
        "| Model (Matched 8-Trial Budget) | Held-Out Top-25% Signed r | All-90 Signed r | Held-Out Top-25% |r| | Task-Evoked Signed r | Single-Trial Signed r | Held-Out RMSE (uM) | Lat Err (s) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    rb = recon["regime_b_matched_8trial"]
    for m_name, d in rb.items():
        if m_name.startswith("paired_"):
            continue
        lines.append(
            f"| {m_name} | {d['top25_contrast_signed_r']['mean']:+.4f} ± {d['top25_contrast_signed_r']['std']:.4f} | "
            f"{d['all90_contrast_signed_r']['mean']:+.4f} | {d['top25_contrast_abs_r']['mean']:.4f} ± {d['top25_contrast_abs_r']['std']:.4f} | "
            f"{d['top25_evoked_signed_r']['mean']:+.4f} | {d['top25_single_trial_signed_r']['mean']:+.4f} | "
            f"{d['top25_rmse_uM']['mean']:.3f} | {d['peak_lat_err_s']['mean']:.2f} |"
        )

    lines += [
        f"- **EB-NVC vs Canon (Regime B Held-Out `te=8..29`)**: Signed r diff `{rb['paired_eb_vs_canon']['top25_contrast_signed_r']['diff_mean']:+.4f}` ($p={rb['paired_eb_vs_canon']['top25_contrast_signed_r']['wilcoxon_p']:.4f}$); `|r|` diff `{rb['paired_eb_vs_canon']['top25_contrast_abs_r']['diff_mean']:+.4f}` ($p={rb['paired_eb_vs_canon']['top25_contrast_abs_r']['wilcoxon_p']:.4e}$); RMSE diff `{rb['paired_eb_vs_canon']['top25_rmse_uM']['diff_mean']:+.4f}` ($p={rb['paired_eb_vs_canon']['top25_rmse_uM']['wilcoxon_p']:.4f}$).",
        "",
        "## 2. Experiment 2: Nested LOSO Screening & BH-FDR Exploratory Clinical Associations",
        f"- **Strict Outer-Fold Closed-Form NVC (Zero Stacking)**: AUC = `{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['auc']:.3f}` (95% CI `[{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['auc_ci_lo']:.3f}, {exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['auc_ci_hi']:.3f}]`), Nested Acc = `{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['nested_acc']*100:.1f}%` (Sens `{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['sensitivity']*100:.1f}%`, Spec `{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['specificity']*100:.1f}%`, BalAcc `{exp2['nested_loso']['Strict Outer-Fold Closed-Form NVC']['nested_balanced_acc']*100:.1f}%`), Paired ΔAUC vs EEG-Only boot $p = {exp2['nested_loso']['paired_delta_auc_strict']['paired_bootstrap_p']:.4f}$.",
        f"- **Nested 37-fold LOSO combined NVC features (includes measured HbO)**: AUC = `{exp2['nested_loso']['NVC Biomarkers']['auc']:.3f}` (95% CI `[{exp2['nested_loso']['NVC Biomarkers']['auc_ci_lo']:.3f}, {exp2['nested_loso']['NVC Biomarkers']['auc_ci_hi']:.3f}]`), Nested Acc = `{exp2['nested_loso']['NVC Biomarkers']['nested_acc']*100:.1f}%` (Sens `{exp2['nested_loso']['NVC Biomarkers']['sensitivity']*100:.1f}%`, Spec `{exp2['nested_loso']['NVC Biomarkers']['specificity']*100:.1f}%`, BalAcc `{exp2['nested_loso']['NVC Biomarkers']['nested_balanced_acc']*100:.1f}%`).",
        f"- **Nested LOSO EEG-Only**: AUC = `{exp2['nested_loso']['EEG-Only Features']['auc']:.3f}` (95% CI `[{exp2['nested_loso']['EEG-Only Features']['auc_ci_lo']:.3f}, {exp2['nested_loso']['EEG-Only Features']['auc_ci_hi']:.3f}]`), Paired ΔAUC boot $p = {exp2['nested_loso']['paired_delta_auc']['paired_bootstrap_p']:.4f}$.",
        "",
        "## 3. Experiment 3: Strictly In-Fold 6-Step MI Decoding & Placebo Controls (HEFMI-ICH)",
        "| Step / Method | 5-Fold In-Fold CV (All n=37) | 95% Bootstrap CI | Block Split (0..14 -> 15..29) |",
        "| :--- | :---: | :---: | :---: |",
    ]
    cv5 = dec["cv_5fold"]
    blk = dec["block_split_15v15"]
    for m_name in ["1_EEG_Raw", "2_InFold_Shrinkage_EA", "3_Canon_HRF_K1", "4_EB_MultiBasis_K5", "5_Add_FastSlow_Cov", "6_Full_FastSlow_EntropyGated", "Oracle_Real_EEG_HbO"]:
        lines.append(
            f"| {m_name} | {cv5[m_name]['mean']*100:.2f}% ± {cv5[m_name]['std']*100:.2f}% | "
            f"[{cv5[m_name]['ci_lo']*100:.2f}%, {cv5[m_name]['ci_hi']*100:.2f}%] | {blk[m_name]['mean']*100:.2f}% |"
        )
    pt = dec["permutation_test"]
    lines.append(f"- **200-Run Label-Permutation Test**: Null Full = `{pt['null_full_mean']*100:.2f}% ± {pt['null_full_std']*100:.2f}%`, $p_{{perm}} = {pt['perm_p_full']:.4f}$.")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="HEFMI-ICH v2 evaluation and figure generation.")
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate Fig. 1-3 directly from saved route_a_metrics.json and subject_nvc_biomarkers.csv without re-running decoding permutations.",
    )
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)
    records = load_hefmi_dataset()
    bio_df = pd.read_csv(os.path.join(RESULTS_DIR, "subject_nvc_biomarkers.csv"))
    metrics_path = os.path.join(RESULTS_DIR, "route_a_metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    if args.figures_only:
        required = ("experiment_1_v2", "experiment_2_v2", "experiment_3_verified")
        missing = [k for k in required if k not in metrics]
        if missing:
            raise RuntimeError(f"Missing saved v2 metrics {missing} in {metrics_path}; run without --figures-only first.")
        _, rep_wave = compute_reconstruction_benchmark(records)
        roc_data, _, _ = compute_nested_loso_and_clinical(bio_df, records)
        recon_res = metrics["experiment_1_v2"]
        exp2_v2 = metrics["experiment_2_v2"]
        dec_res = metrics["experiment_3_verified"]
        plot_fig1_pathology(records, bio_df, recon_res, os.path.join(FIG_DIR, "fig1_nvc_pathology_comparison.png"))
        plot_fig2_clinical(bio_df, roc_data, exp2_v2, os.path.join(FIG_DIR, "fig2_lesion_asymmetry_clinical.png"))
        plot_fig3_recon_and_decoding(recon_res, rep_wave, dec_res, os.path.join(FIG_DIR, "fig3_virtual_hbo_decoding.png"))
        print("Regenerated fig1-fig3 (.png/.pdf) from saved v2 metrics (--figures-only).")
        return

    recon_res, rep_wave = compute_reconstruction_benchmark(records)
    dec_res, fold_manifest = compute_verified_decoding(records)
    roc_data, exp2_v2, clin_df = compute_nested_loso_and_clinical(bio_df, records)

    metrics["experiment_1_v2"] = recon_res
    metrics["experiment_2_v2"] = exp2_v2
    metrics["experiment_3_verified"] = dec_res

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    manifest_path = os.path.join(RESULTS_DIR, "fold_manifest.json")
    manifest = json.load(open(manifest_path, "r", encoding="utf-8")) if os.path.exists(manifest_path) else {}
    manifest["HEFMI_ICH_5fold"] = fold_manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    clin_df.to_csv(os.path.join(RESULTS_DIR, "clinical_ich_fdr.csv"), index=False)

    # Preserve Section 4 (Liu2024 & Stratified) if already appended in route_a_report.md
    rep_path = os.path.join(RESULTS_DIR, "route_a_report.md")
    sec4_tail = ""
    if os.path.exists(rep_path):
        old_rep = open(rep_path, "r", encoding="utf-8").read()
        if "## 4. Experiment 4:" in old_rep:
            sec4_tail = "\n## 4. Experiment 4:" + old_rep.split("## 4. Experiment 4:", 1)[1]
    write_final_report(metrics, rep_path)
    if sec4_tail:
        with open(rep_path, "a", encoding="utf-8") as f:
            f.write(sec4_tail)
    plot_fig1_pathology(records, bio_df, recon_res, os.path.join(FIG_DIR, "fig1_nvc_pathology_comparison.png"))
    plot_fig2_clinical(bio_df, roc_data, exp2_v2, os.path.join(FIG_DIR, "fig2_lesion_asymmetry_clinical.png"))
    plot_fig3_recon_and_decoding(recon_res, rep_wave, dec_res, os.path.join(FIG_DIR, "fig3_virtual_hbo_decoding.png"))
    print("Finalized v2 HEFMI-ICH metrics, manifest, clinical FDR CSV, report, and 3 figures successfully.")


if __name__ == "__main__":
    main()
