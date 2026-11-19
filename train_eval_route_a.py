"""Stage-2/3 Complete Training & Evaluation Pipeline for Route A (Physio-NVC-Net).

Executes:
1. Experiment 1: 37-fold LOSO + HC->ICH cross-group EEG->HbO dynamical reconstruction
   comparing Canonical-HRF, Ridge-FIR, BlackBox-TCN, Physio-NVC-Net (w/o Asym Graph),
   and Physio-NVC-Net (Full).
2. Experiment 2: Biophysical NVC uncoupling biomarker extraction, HC vs ICH pathology
   detection (LOSO ROC-AUC), and clinical scale association (FMA-UE, mRS, MBI) +
   Leave-One-Patient-Out regression.
3. Experiment 3: Chronological Few-Shot Calibration (k=2,4,6,8 trials/class) and 5-fold CV
   comparing EEG-Only, EEG + Canonical HRF, EEG + Physio-NVC-Net Virtual HbO (single-EEG
   at inference), and Oracle Hybrid (Real EEG + Real HbO).

Outputs:
  - e:\\shujushujujj\\results\\route_a_metrics.json
  - e:\\shujushujujj\\results\\route_a_predictions.npz (cached out-of-fold predictions for plotting)
  - e:\\shujushujujj\\results\\route_a_report.md
"""

from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mne.decoding import CSP
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from data_hefmi import SubjectRecord, get_hemisphere_indices, load_hefmi_dataset
from models import BlackBoxTCN, CanonicalHRFBaseline, PhysioNVCNet
from models.physio_nvc_ssm import build_gamma_hrf_bank

RESULTS_DIR = r"e:\shujushujujj\results"
SEED = 42


def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lesion_to_onehot(lesion_side: str) -> np.ndarray:
    mapping = {"None": 0, "Left": 1, "Right": 2, "Brainstem": 3}
    vec = np.zeros(4, dtype=np.float32)
    vec[mapping.get(lesion_side, 0)] = 1.0
    return vec


def nvc_reconstruction_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    labels: torch.Tensor,
    subj_offsets: List[Tuple[int, int]],
) -> torch.Tensor:
    """Composite loss: trial SmoothL1 + temporal cosine + subject class-contrast evoked alignment."""
    yp = y_pred[:, :, 20:90]
    yt = y_true[:, :, 20:90]

    l1_trial = F.smooth_l1_loss(yp, yt, beta=0.25)
    cos_trial = (1.0 - F.cosine_similarity(yp, yt, dim=-1, eps=1e-6)).mean()

    evoked_losses = []
    for st, en in subj_offsets:
        yp_s = yp[st:en]
        yt_s = yt[st:en]
        lbl_s = labels[st:en]
        if (lbl_s == 0).any() and (lbl_s == 1).any():
            dy_p = yp_s[lbl_s == 1].mean(dim=0) - yp_s[lbl_s == 0].mean(dim=0)
            dy_t = yt_s[lbl_s == 1].mean(dim=0) - yt_s[lbl_s == 0].mean(dim=0)
            ev_l1 = F.smooth_l1_loss(dy_p, dy_t, beta=0.15)
            ev_cos = (1.0 - F.cosine_similarity(dy_p, dy_t, dim=-1, eps=1e-6)).mean()
            evoked_losses.append(ev_l1 + 0.5 * ev_cos)

    evoked_loss = torch.stack(evoked_losses).mean() if evoked_losses else torch.tensor(0.0, device=yp.device)
    return l1_trial + 0.35 * cos_trial + 1.25 * evoked_loss


def compute_recon_metrics(y_pred: np.ndarray, y_true: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Evaluate single-subject reconstruction metrics on [2.0, 9.0] s window."""
    yp = y_pred[:, :, 20:90]
    yt = y_true[:, :, 20:90]

    # Evoked contrast (Class 1 - Class 0): (90, 70)
    dy_p = yp[labels == 1].mean(axis=0) - yp[labels == 0].mean(axis=0)
    dy_t = yt[labels == 1].mean(axis=0) - yt[labels == 0].mean(axis=0)

    # Channel-wise evoked correlation
    r_signed_list, r_abs_list = [], []
    ch_power = np.sqrt(np.mean(dy_t ** 2, axis=1))
    top25_thresh = np.percentile(ch_power, 75)
    r_top25_signed, r_top25_abs = [], []
    lat_errs = []

    for c in range(90):
        a, b = dy_p[c], dy_t[c]
        if a.std() > 1e-8 and b.std() > 1e-8:
            rc = float(np.corrcoef(a, b)[0, 1])
            r_signed_list.append(rc)
            r_abs_list.append(abs(rc))
            if ch_power[c] >= top25_thresh:
                r_top25_signed.append(rc)
                r_top25_abs.append(abs(rc))
                # Peak latency error in seconds (10 Hz)
                t_p = np.argmax(np.abs(a)) / 10.0
                t_t = np.argmax(np.abs(b)) / 10.0
                lat_errs.append(abs(t_p - t_t))

    # Global evoked NRMSE (normalized by true std)
    nrmse_evoked = float(np.sqrt(np.mean((dy_p - dy_t) ** 2)) / (dy_t.std() + 1e-8))
    # Trial-level temporal correlation across responsive channels
    top_ch = np.where(ch_power >= top25_thresh)[0]
    trial_rs = []
    for n in range(yp.shape[0]):
        for c in top_ch:
            a, b = yp[n, c], yt[n, c]
            if a.std() > 1e-8 and b.std() > 1e-8:
                trial_rs.append(float(np.corrcoef(a, b)[0, 1]))

    return {
        "r_evoked_signed": float(np.mean(r_signed_list)) if r_signed_list else 0.0,
        "r_evoked_abs": float(np.mean(r_abs_list)) if r_abs_list else 0.0,
        "r_top25_signed": float(np.mean(r_top25_signed)) if r_top25_signed else 0.0,
        "r_top25_abs": float(np.mean(r_top25_abs)) if r_top25_abs else 0.0,
        "nrmse_evoked": nrmse_evoked,
        "peak_lat_err_s": float(np.mean(lat_errs)) if lat_errs else 0.0,
        "r_trial_top25": float(np.mean(trial_rs)) if trial_rs else 0.0,
    }


def make_lagged_features(u: np.ndarray, lags: Tuple[int, ...]) -> np.ndarray:
    """Create causal multi-lag FIR design tensor (N, C, T, L) for Ridge-FIR baseline."""
    n_tr, n_ch, n_t = u.shape
    feats = []
    for lag in lags:
        if lag == 0:
            feats.append(u)
        else:
            pad = np.zeros((n_tr, n_ch, lag), dtype=u.dtype)
            feats.append(np.concatenate([pad, u[:, :, :-lag]], axis=2))
    return np.stack(feats, axis=-1)


def run_ridge_fir_loso(records: List[SubjectRecord]) -> Tuple[List[Dict[str, float]], np.ndarray]:
    """Evaluate closed-form Ridge-FIR baseline under 37-fold LOSO."""
    lags = (0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60)
    lag_feats = [make_lagged_features(r.u_env, lags) for r in records]
    oof_preds = np.zeros((len(records), 30, 90, 100), dtype=np.float32)
    metrics_list = []

    for i in range(len(records)):
        x_tr = np.concatenate([lag_feats[j][:, :, 20:90, :] for j in range(len(records)) if j != i], axis=0)
        y_tr = np.concatenate([records[j].y_hbo[:, :, 20:90] for j in range(len(records)) if j != i], axis=0)
        # Fit Ridge across flattened observations
        reg = Ridge(alpha=500.0)
        reg.fit(x_tr.reshape(-1, len(lags)), y_tr.reshape(-1))
        pred_full = reg.predict(lag_feats[i].reshape(-1, len(lags))).reshape(30, 90, 100)
        pred_full = pred_full - pred_full[:, :, :10].mean(axis=2, keepdims=True)
        oof_preds[i] = pred_full.astype(np.float32)
        metrics_list.append(compute_recon_metrics(oof_preds[i], records[i].y_hbo, records[i].labels))

    return metrics_list, oof_preds


def train_torch_model_on_subjects(
    model_fn,
    train_recs: List[SubjectRecord],
    device: torch.device,
    epochs: int = 45,
    lr: float = 2e-3,
    batch_subjects: int = 6,
    seed: int = SEED,
) -> nn.Module:
    """Train a reconstruction network with subject-grouped batches and gradient clipping."""
    set_seed(seed)
    model = model_fn().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-move tensors to GPU (37 subjects total is ~50 MB, fits completely on RTX 4060!)
    u_tensors = [torch.from_numpy(r.u_env).to(device) for r in train_recs]
    e_tensors = [torch.from_numpy(r.eeg_bands).to(device) for r in train_recs]
    y_tensors = [torch.from_numpy(r.y_hbo).to(device) for r in train_recs]
    l_tensors = [torch.from_numpy(r.labels).to(device) for r in train_recs]
    les_tensors = [
        torch.from_numpy(lesion_to_onehot(r.lesion_side)).unsqueeze(0).expand(30, -1).to(device)
        for r in train_recs
    ]

    n_sub = len(train_recs)
    rng = np.random.default_rng(seed)
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n_sub)
        for start in range(0, n_sub, batch_subjects):
            idx_batch = perm[start : start + batch_subjects]
            u_b = torch.cat([u_tensors[k] for k in idx_batch], dim=0)
            e_b = torch.cat([e_tensors[k] for k in idx_batch], dim=0)
            y_b = torch.cat([y_tensors[k] for k in idx_batch], dim=0)
            lbl_b = torch.cat([l_tensors[k] for k in idx_batch], dim=0)
            # Randomly drop lesion conditioning 50% of steps so blind inference works seamlessly
            if rng.random() < 0.5:
                les_b = torch.cat([les_tensors[k] for k in idx_batch], dim=0)
            else:
                les_b = None

            offsets = [(m * 30, (m + 1) * 30) for m in range(len(idx_batch))]
            optimizer.zero_grad()
            out = model(u_b, e_b, lesion_onehot=les_b)
            loss = nvc_reconstruction_loss(out["y_pred"], y_b, lbl_b, offsets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()
    return model


@torch.no_grad()
def predict_subject(
    model: nn.Module,
    rec: SubjectRecord,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Run strictly blind inference (lesion_onehot=None, no labels) on one held-out subject."""
    model.eval()
    u_t = torch.from_numpy(rec.u_env).to(device)
    e_t = torch.from_numpy(rec.eeg_bands).to(device)
    out = model(u_t, e_t, lesion_onehot=None)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


def run_experiment_1(records: List[SubjectRecord], device: torch.device) -> Tuple[dict, dict]:
    """Run Experiment 1: 37-fold LOSO and HC->ICH transfer evaluation."""
    print("\n=== Experiment 1: EEG -> HbO Dynamical Reconstruction ===")
    model_specs = {
        "Canonical-HRF": (lambda: CanonicalHRFBaseline(), 35, 5e-3),
        "BlackBox-TCN": (lambda: BlackBoxTCN(), 45, 2e-3),
        "Physio-NVC-Net (w/o Asym Graph)": (lambda: PhysioNVCNet(use_asymmetry_graph=False), 45, 2e-3),
        "Physio-NVC-Net (Full)": (lambda: PhysioNVCNet(use_asymmetry_graph=True), 45, 2e-3),
    }

    exp1_summary = {}
    oof_store = {}

    # 1. Ridge-FIR baseline
    ridge_metrics, ridge_oof = run_ridge_fir_loso(records)
    oof_store["Ridge-FIR"] = ridge_oof
    exp1_summary["Ridge-FIR"] = summarize_recon_metrics(ridge_metrics, records)

    # 2. PyTorch models 37-fold LOSO
    physio_bio_params = []  # store alpha_gain, tau_eff, res_ratio, contra_gate from blind LOSO Physio-NVC-Net
    for name, (factory, epochs, lr) in model_specs.items():
        t0 = time.time()
        fold_metrics = []
        oof_y = np.zeros((len(records), 30, 90, 100), dtype=np.float32)

        for i in range(len(records)):
            tr_recs = [records[j] for j in range(len(records)) if j != i]
            model = train_torch_model_on_subjects(
                factory, tr_recs, device=device, epochs=epochs, lr=lr, seed=SEED + i
            )
            out_i = predict_subject(model, records[i], device=device)
            oof_y[i] = out_i["y_pred"]
            fold_metrics.append(compute_recon_metrics(out_i["y_pred"], records[i].y_hbo, records[i].labels))
            if name == "Physio-NVC-Net (Full)":
                physio_bio_params.append({
                    "alpha_gain": out_i["alpha_gain"].mean(axis=0),   # (90,)
                    "tau_eff": out_i["tau_eff"].mean(axis=0),         # (90,)
                    "res_ratio": out_i["res_ratio"].mean(axis=0),     # (90,)
                    "contra_gate": out_i["contra_gate"].mean(axis=0), # (90,)
                })

        oof_store[name] = oof_y
        exp1_summary[name] = summarize_recon_metrics(fold_metrics, records)
        dt = time.time() - t0
        s = exp1_summary[name]["all"]
        print(
            f"  {name:32s} | {dt:5.1f}s | r_evk_abs={s['r_evoked_abs']['mean']:.3f} | "
            f"r_top25_abs={s['r_top25_abs']['mean']:.3f} | r_top25_signed={s['r_top25_signed']['mean']:+.3f} | "
            f"lat_err={s['peak_lat_err_s']['mean']:.2f}s | nrmse={s['nrmse_evoked']['mean']:.3f}"
        )

    # 3. HC (17) -> ICH (20) zero-shot cross-group transfer
    hc_recs = [r for r in records if not r.is_patient]
    pt_recs = [r for r in records if r.is_patient]
    hc_to_ich = {}
    for name, (factory, epochs, lr) in model_specs.items():
        m_hc = train_torch_model_on_subjects(factory, hc_recs, device=device, epochs=epochs, lr=lr, seed=SEED)
        pt_ms = []
        for pr in pt_recs:
            out_p = predict_subject(m_hc, pr, device=device)
            pt_ms.append(compute_recon_metrics(out_p["y_pred"], pr.y_hbo, pr.labels))
        hc_to_ich[name] = {
            k: {"mean": float(np.mean([m[k] for m in pt_ms])), "std": float(np.std([m[k] for m in pt_ms]))}
            for k in pt_ms[0]
        }

    exp1_summary["HC_to_ICH_Transfer"] = hc_to_ich
    oof_store["physio_bio_params"] = physio_bio_params
    return exp1_summary, oof_store


def summarize_recon_metrics(fold_metrics: List[Dict[str, float]], records: List[SubjectRecord]) -> dict:
    keys = list(fold_metrics[0].keys())
    hc_idx = [i for i, r in enumerate(records) if not r.is_patient]
    pt_idx = [i for i, r in enumerate(records) if r.is_patient]

    def agg(indices: List[int]) -> dict:
        return {
            k: {
                "mean": float(np.mean([fold_metrics[i][k] for i in indices])),
                "std": float(np.std([fold_metrics[i][k] for i in indices])),
            }
            for k in keys
        }

    return {
        "all": agg(list(range(len(records)))),
        "hc": agg(hc_idx),
        "ich": agg(pt_idx),
        "per_subject": fold_metrics,
    }


def run_experiment_2(records: List[SubjectRecord], physio_bio_params: List[dict]) -> Tuple[dict, pd.DataFrame]:
    """Run Experiment 2: NVC uncoupling biomarkers, HC vs ICH detection, and clinical scale prediction."""
    print("\n=== Experiment 2: NVC Uncoupling Biomarkers & Clinical Association ===")
    hrf = build_gamma_hrf_bank(taus=(4.5,), sigma=1.1, fs=10.0, kernel_len=70)[0].numpy()

    rows = []
    for i, r in enumerate(records):
        bp = physio_bio_params[i]
        ipsi_idx, contra_idx = get_hemisphere_indices(r.lesion_side)

        # Empirical NVC metrics from real signals
        clean_gain = r.hbo_std / (r.clean_u_std + 1e-8)
        hbo_peak = float(np.abs(r.y_hbo[:, :, 20:80].mean(axis=2)).mean())

        # Per-hemisphere HbO amplitude & band-specific coupling (mu/beta -> HbO evoked correlation)
        hbo_ch_std = r.y_hbo.std(axis=(0, 2))
        hbo_left = float(hbo_ch_std[:45].mean())
        hbo_right = float(hbo_ch_std[45:].mean())
        hbo_abs_asym = abs(hbo_left - hbo_right) / (hbo_left + hbo_right + 1e-8)

        # Model-inverted biophysical parameters (extracted purely from blind LOSO EEG inference!)
        alpha = bp["alpha_gain"]
        tau = bp["tau_eff"]
        res_r = bp["res_ratio"]
        c_gate = bp["contra_gate"]

        alpha_mean = float(alpha.mean())
        alpha_abs_asym = float(abs(alpha[:45].mean() - alpha[45:].mean()) / (alpha.mean() + 1e-8))
        tau_mean = float(tau.mean())
        tau_abs_asym = float(abs(tau[:45].mean() - tau[45:].mean()))
        res_mean = float(res_r.mean())
        contra_gate_abs = float(np.abs(c_gate).mean())

        # Lesion-relative features (meaningful for lateralized ICH; for HC defaults to Left vs Right)
        alpha_ipsi = float(alpha[ipsi_idx].mean())
        alpha_contra = float(alpha[contra_idx].mean())
        li_alpha = (alpha_contra - alpha_ipsi) / (alpha_contra + alpha_ipsi + 1e-8)

        tau_ipsi = float(tau[ipsi_idx].mean())
        tau_contra = float(tau[contra_idx].mean())
        delta_tau_ipsi_contra = tau_ipsi - tau_contra

        res_ipsi = float(res_r[ipsi_idx].mean())
        res_contra = float(res_r[contra_idx].mean())

        hbo_ipsi = float(hbo_ch_std[ipsi_idx].mean())
        hbo_contra = float(hbo_ch_std[contra_idx].mean())
        li_hbo = (hbo_contra - hbo_ipsi) / (hbo_contra + hbo_ipsi + 1e-8)

        # Band-specific evoked HRF coupling (theta=0, mu=1, beta=2)
        band_ev_rs = []
        for b_idx in range(3):
            b_env = r.eeg_bands[:, :, b_idx, :]  # (30, 32, 100)
            b_diff = b_env[r.labels == 1].mean(axis=0) - b_env[r.labels == 0].mean(axis=0)
            b_conv = np.array([np.convolve(b_diff[c], hrf, mode="full")[:100] for c in range(32)])
            # Match to top responsive HbO channels via max absolute correlation across EEG motor channels
            h_diff = r.y_hbo[r.labels == 1].mean(axis=0) - r.y_hbo[r.labels == 0].mean(axis=0)
            top_h = h_diff[np.argmax(np.std(h_diff[:, 20:80], axis=1)), 20:90]
            best_r = max(
                abs(np.corrcoef(b_conv[c, 20:90], top_h)[0, 1])
                for c in range(32)
                if b_conv[c, 20:90].std() > 1e-8 and top_h.std() > 1e-8
            )
            band_ev_rs.append(float(best_r))

        row = {
            "file": r.file_name,
            "is_patient": int(r.is_patient),
            "subject_id": r.subject_id,
            "lesion_side": r.lesion_side,
            "clean_x_std": r.clean_x_std,
            "clean_u_std": r.clean_u_std,
            "hbo_std": r.hbo_std,
            "hbo_peak": hbo_peak,
            "clean_gain": clean_gain,
            "hbo_abs_asym": hbo_abs_asym,
            "alpha_mean": alpha_mean,
            "alpha_abs_asym": alpha_abs_asym,
            "tau_mean": tau_mean,
            "tau_abs_asym": tau_abs_asym,
            "res_mean": res_mean,
            "contra_gate_abs": contra_gate_abs,
            "alpha_ipsi": alpha_ipsi,
            "alpha_contra": alpha_contra,
            "li_alpha": li_alpha,
            "tau_ipsi": tau_ipsi,
            "tau_contra": tau_contra,
            "delta_tau_ipsi_contra": delta_tau_ipsi_contra,
            "res_ipsi": res_ipsi,
            "res_contra": res_contra,
            "hbo_ipsi": hbo_ipsi,
            "hbo_contra": hbo_contra,
            "li_hbo": li_hbo,
            "band_r_theta": band_ev_rs[0],
            "band_r_mu": band_ev_rs[1],
            "band_r_beta": band_ev_rs[2],
            "fma_ue": r.clinical.get("fma_ue", np.nan),
            "mrs": r.clinical.get("mrs", np.nan),
            "mbi": r.clinical.get("mbi", np.nan),
            "age": r.clinical.get("age", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    hc_df = df[df["is_patient"] == 0]
    ich_df = df[df["is_patient"] == 1]

    # 1. HC vs ICH group stats & Cohen's d
    group_tests = {}
    test_cols = [
        "clean_gain", "hbo_std", "hbo_peak", "clean_x_std",
        "alpha_mean", "tau_mean", "tau_abs_asym", "res_mean", "hbo_abs_asym", "band_r_mu", "band_r_beta",
    ]
    for col in test_cols:
        _, p_mwu = stats.mannwhitneyu(hc_df[col], ich_df[col], alternative="two-sided")
        pooled_sd = np.sqrt(
            ((len(hc_df) - 1) * hc_df[col].var() + (len(ich_df) - 1) * ich_df[col].var())
            / (len(hc_df) + len(ich_df) - 2)
        )
        cohens_d = float((hc_df[col].mean() - ich_df[col].mean()) / (pooled_sd + 1e-8))
        group_tests[col] = {
            "hc_mean": float(hc_df[col].mean()),
            "hc_std": float(hc_df[col].std()),
            "ich_mean": float(ich_df[col].mean()),
            "ich_std": float(ich_df[col].std()),
            "mwu_p": float(p_mwu),
            "cohens_d": cohens_d,
        }
        if p_mwu < 0.05:
            print(f"  [HC vs ICH sig] {col:16s}: HC={hc_df[col].mean():.4f} vs ICH={ich_df[col].mean():.4f}, p={p_mwu:.3e}, d={cohens_d:+.2f}")

    # 2. Strict LOSO pathology classification (lesion-blind features only!)
    blind_nvc_feats = [
        "clean_gain", "hbo_peak", "alpha_mean", "alpha_abs_asym",
        "tau_mean", "tau_abs_asym", "res_mean", "hbo_abs_asym",
    ]
    eeg_only_feats = ["clean_x_std", "clean_u_std", "alpha_mean", "tau_abs_asym", "res_mean"]

    def loso_pathology_auc(feat_cols: List[str]) -> Tuple[float, float]:
        x_mat = df[feat_cols].values
        y_vec = df["is_patient"].values
        probs = np.zeros(len(df))
        for i in range(len(df)):
            mask = np.arange(len(df)) != i
            sc = StandardScaler()
            x_tr = sc.fit_transform(x_mat[mask])
            x_te = sc.transform(x_mat[i : i + 1])
            clf = LogisticRegression(C=0.5, max_iter=500).fit(x_tr, y_vec[mask])
            probs[i] = clf.predict_proba(x_te)[0, 1]
        return float(roc_auc_score(y_vec, probs)), float(accuracy_score(y_vec, probs >= 0.5))

    auc_nvc, acc_nvc = loso_pathology_auc(blind_nvc_feats)
    auc_eeg, acc_eeg = loso_pathology_auc(eeg_only_feats)
    print(f"  [LOSO Pathology Detection] NVC Biomarkers: AUC={auc_nvc:.3f}, Acc={acc_nvc*100:.1f}% | EEG-only bio: AUC={auc_eeg:.3f}, Acc={acc_eeg*100:.1f}%")

    # 3. Clinical scale univariate correlations + LOPO regression (ICH n=20)
    clinical_corrs = {}
    bio_cand_cols = [
        "clean_gain", "hbo_std", "hbo_peak", "alpha_mean", "tau_mean", "tau_abs_asym",
        "res_mean", "li_alpha", "delta_tau_ipsi_contra", "li_hbo", "res_ipsi", "band_r_mu", "band_r_beta",
    ]
    for target in ["fma_ue", "mrs", "mbi"]:
        clinical_corrs[target] = {}
        for col in bio_cand_cols:
            rho, p_sp = stats.spearmanr(ich_df[col], ich_df[target])
            r_pe, p_pe = stats.pearsonr(ich_df[col], ich_df[target])
            clinical_corrs[target][col] = {
                "spearman_rho": float(rho),
                "spearman_p": float(p_sp),
                "pearson_r": float(r_pe),
                "pearson_p": float(p_pe),
            }

    # LOPO Ridge prediction of FMA-UE and mRS
    lopo_reg = {}
    nvc_reg_cols = [
        "clean_gain", "hbo_peak", "alpha_mean", "tau_mean", "tau_abs_asym",
        "res_mean", "delta_tau_ipsi_contra", "li_hbo", "band_r_mu", "band_r_beta",
    ]
    eeg_reg_cols = ["clean_x_std", "clean_u_std", "alpha_mean", "tau_abs_asym", "res_mean"]

    for target in ["fma_ue", "mrs"]:
        lopo_reg[target] = {}
        y_t = ich_df[target].values
        for fset_name, cols in [("EEG-Only Features", eeg_reg_cols), ("NVC Biomarker Vector", nvc_reg_cols)]:
            x_m = ich_df[cols].values
            preds = np.zeros(len(ich_df))
            for i in range(len(ich_df)):
                mask = np.arange(len(ich_df)) != i
                sc = StandardScaler()
                reg = Ridge(alpha=15.0).fit(sc.fit_transform(x_m[mask]), y_t[mask])
                preds[i] = reg.predict(sc.transform(x_m[i : i + 1]))[0]
            r_p, p_p = stats.pearsonr(preds, y_t)
            rho_s, p_s = stats.spearmanr(preds, y_t)
            mae = float(mean_absolute_error(y_t, preds))
            lopo_reg[target][fset_name] = {
                "pearson_r": float(r_p), "pearson_p": float(p_p),
                "spearman_rho": float(rho_s), "spearman_p": float(p_s), "mae": mae,
            }
            print(f"  [LOPO Clinical Reg {target:6s}] {fset_name:22s}: r={r_p:+.3f} (p={p_p:.4f}), rho={rho_s:+.3f}, MAE={mae:.2f}")

    return {
        "group_tests": group_tests,
        "loso_pathology": {
            "nvc_auc": auc_nvc, "nvc_acc": acc_nvc,
            "eeg_auc": auc_eeg, "eeg_acc": acc_eeg,
        },
        "clinical_corrs": clinical_corrs,
        "lopo_clinical_regression": lopo_reg,
    }, df


def extract_eeg_features(rec: SubjectRecord, tr_idx: np.ndarray, te_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit CSP + multi-band log-power purely on calibration indices tr_idx and transform both."""
    x_tr = rec.x_sm_ea[tr_idx, :, 100:900].astype(np.float64)
    y_tr = rec.labels[tr_idx]
    x_te = rec.x_sm_ea[te_idx, :, 100:900].astype(np.float64)

    n_comp = min(4, max(2, len(tr_idx) - 2))
    csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True, norm_trace=False)
    f_csp_tr = csp.fit_transform(x_tr, y_tr)
    f_csp_te = csp.transform(x_te)

    # Compact band envelope summaries (mu & beta in early 1-5s and late 5-9s windows)
    def band_sum(idx: np.ndarray) -> np.ndarray:
        mu = rec.eeg_bands[idx, :, 1, :]  # (N, 32, 100)
        be = rec.eeg_bands[idx, :, 2, :]
        return np.concatenate(
            [mu[:, :, 15:50].mean(2), mu[:, :, 50:90].mean(2), be[:, :, 15:50].mean(2), be[:, :, 50:90].mean(2)],
            axis=1,
        )

    f_b_tr, f_b_te = band_sum(tr_idx), band_sum(te_idx)
    n_pca = min(6, max(2, len(tr_idx) - 1))
    sc_b = StandardScaler()
    pca_b = PCA(n_components=n_pca, random_state=SEED)
    fb_tr_p = pca_b.fit_transform(sc_b.fit_transform(f_b_tr))
    fb_te_p = pca_b.transform(sc_b.transform(f_b_te))

    return np.concatenate([f_csp_tr, fb_tr_p], axis=1), np.concatenate([f_csp_te, fb_te_p], axis=1)


def extract_hemo_features(
    hbo_tensor: np.ndarray, tr_idx: np.ndarray, te_idx: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract window-averaged HbO features + calibration-only PCA compression."""
    def raw_win(idx: np.ndarray) -> np.ndarray:
        h = hbo_tensor[idx]
        return np.concatenate(
            [h[:, :, 20:45].mean(2), h[:, :, 45:70].mean(2), h[:, :, 70:95].mean(2)],
            axis=1,
        )

    w_tr, w_te = raw_win(tr_idx), raw_win(te_idx)
    sc = StandardScaler()
    n_pca = min(8, max(2, len(tr_idx) - 1))
    pca = PCA(n_components=n_pca, random_state=SEED)
    return pca.fit_transform(sc.fit_transform(w_tr)), pca.transform(sc.transform(w_te))


def eval_split_decoding(
    rec: SubjectRecord,
    y_canon: np.ndarray,
    y_physio: np.ndarray,
    tr_idx: np.ndarray,
    te_idx: np.ndarray,
) -> Dict[str, float]:
    """Evaluate 4 decoding regimes on a single calibration/test split."""
    y_tr, y_te = rec.labels[tr_idx], rec.labels[te_idx]
    fe_tr, fe_te = extract_eeg_features(rec, tr_idx, te_idx)
    fc_tr, fc_te = extract_hemo_features(y_canon, tr_idx, te_idx)
    fp_tr, fp_te = extract_hemo_features(y_physio, tr_idx, te_idx)
    fo_tr, fo_te = extract_hemo_features(rec.y_hbo, tr_idx, te_idx)

    feature_sets = {
        "EEG-Only": (fe_tr, fe_te),
        "EEG + Canonical HRF": (np.concatenate([fe_tr, fc_tr], axis=1), np.concatenate([fe_te, fc_te], axis=1)),
        "EEG + Physio-NVC-Net Virtual HbO": (np.concatenate([fe_tr, fp_tr], axis=1), np.concatenate([fe_te, fp_te], axis=1)),
        "Oracle Hybrid (Real EEG + Real HbO)": (np.concatenate([fe_tr, fo_tr], axis=1), np.concatenate([fe_te, fo_te], axis=1)),
    }

    out = {}
    for m_name, (x_tr, x_te) in feature_sets.items():
        sc = StandardScaler()
        lda = LDA(solver="lsqr", shrinkage="auto")
        lda.fit(sc.fit_transform(x_tr), y_tr)
        preds = lda.predict(sc.transform(x_te))
        out[m_name] = float(accuracy_score(y_te, preds))
    return out


def run_experiment_3(
    records: List[SubjectRecord],
    oof_canon: np.ndarray,
    oof_physio: np.ndarray,
) -> dict:
    """Run Experiment 3: Chronological Few-Shot Calibration & 5-Fold Stratified CV MI Decoding."""
    print("\n=== Experiment 3: Single-EEG + Virtual-HbO MI Decoding ===")
    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"

    ks = [2, 4, 6, 8]
    methods = [
        "EEG-Only",
        "EEG + Canonical HRF",
        "EEG + Physio-NVC-Net Virtual HbO",
        "Oracle Hybrid (Real EEG + Real HbO)",
    ]
    hc_idx = [i for i, r in enumerate(records) if not r.is_patient]
    pt_idx = [i for i, r in enumerate(records) if r.is_patient]

    fewshot_res = {}
    for k in ks:
        n_cal = 2 * k
        tr_idx = np.arange(n_cal)
        te_idx = np.arange(n_cal, 30)
        per_sub = [eval_split_decoding(records[i], oof_canon[i], oof_physio[i], tr_idx, te_idx) for i in range(len(records))]

        summary_k = {}
        for m in methods:
            vals_all = np.array([s[m] for s in per_sub])
            vals_hc = vals_all[hc_idx]
            vals_pt = vals_all[pt_idx]
            summary_k[m] = {
                "all_mean": float(vals_all.mean()), "all_std": float(vals_all.std()),
                "hc_mean": float(vals_hc.mean()), "hc_std": float(vals_hc.std()),
                "ich_mean": float(vals_pt.mean()), "ich_std": float(vals_pt.std()),
                "per_subject": vals_all.tolist(),
            }
        # Paired Wilcoxon test: Physio-NVC-Net Virtual HbO vs EEG-Only
        v_ours = np.array(summary_k["EEG + Physio-NVC-Net Virtual HbO"]["per_subject"])
        v_eeg = np.array(summary_k["EEG-Only"]["per_subject"])
        _, p_all = stats.wilcoxon(v_ours, v_eeg, zero_method="wilcox")
        _, p_ich = stats.wilcoxon(v_ours[pt_idx], v_eeg[pt_idx], zero_method="wilcox")
        summary_k["wilcoxon_p_all_vs_eeg"] = float(p_all)
        summary_k["wilcoxon_p_ich_vs_eeg"] = float(p_ich)
        fewshot_res[f"k={k}"] = summary_k
        print(
            f"  k={k} ({n_cal:2d} cal / {30-n_cal:2d} test) | EEG-Only={v_eeg.mean()*100:.2f}% | "
            f"+Canon={summary_k['EEG + Canonical HRF']['all_mean']*100:.2f}% | "
            f"+VirtualHbO(Ours)={v_ours.mean()*100:.2f}% (p={p_all:.4f}; ICH={v_ours[pt_idx].mean()*100:.2f}% vs {v_eeg[pt_idx].mean()*100:.2f}%, p={p_ich:.4f}) | "
            f"Oracle={summary_k['Oracle Hybrid (Real EEG + Real HbO)']['all_mean']*100:.2f}%"
        )

    # 5-fold Stratified CV (24 train / 6 test per fold)
    cv_per_sub = []
    for i, r in enumerate(records):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        fold_acc = {m: [] for m in methods}
        for tr_idx, te_idx in skf.split(r.x_sm_ea, r.labels):
            res_f = eval_split_decoding(r, oof_canon[i], oof_physio[i], tr_idx, te_idx)
            for m in methods:
                fold_acc[m].append(res_f[m])
        cv_per_sub.append({m: float(np.mean(fold_acc[m])) for m in methods})

    cv5_res = {}
    for m in methods:
        va = np.array([s[m] for s in cv_per_sub])
        cv5_res[m] = {
            "all_mean": float(va.mean()), "all_std": float(va.std()),
            "hc_mean": float(va[hc_idx].mean()), "ich_mean": float(va[pt_idx].mean()),
            "per_subject": va.tolist(),
        }
    v_ours_cv = np.array(cv5_res["EEG + Physio-NVC-Net Virtual HbO"]["per_subject"])
    v_eeg_cv = np.array(cv5_res["EEG-Only"]["per_subject"])
    _, p_cv_all = stats.wilcoxon(v_ours_cv, v_eeg_cv, zero_method="wilcox")
    _, p_cv_ich = stats.wilcoxon(v_ours_cv[pt_idx], v_eeg_cv[pt_idx], zero_method="wilcox")
    cv5_res["wilcoxon_p_all_vs_eeg"] = float(p_cv_all)
    cv5_res["wilcoxon_p_ich_vs_eeg"] = float(p_cv_ich)
    print(
        f"  5-Fold CV             | EEG-Only={v_eeg_cv.mean()*100:.2f}% | "
        f"+Canon={cv5_res['EEG + Canonical HRF']['all_mean']*100:.2f}% | "
        f"+VirtualHbO(Ours)={v_ours_cv.mean()*100:.2f}% (p={p_cv_all:.4f}; ICH={v_ours_cv[pt_idx].mean()*100:.2f}%, p={p_cv_ich:.4f}) | "
        f"Oracle={cv5_res['Oracle Hybrid (Real EEG + Real HbO)']['all_mean']*100:.2f}%"
    )

    return {"few_shot": fewshot_res, "cv_5fold": cv5_res}


def write_markdown_report(metrics: dict, out_path: str) -> None:
    lines = [
        "# Route A Evaluation Report: Physio-NVC-Net on HEFMI-ICH-cache",
        "",
        "## 1. Experiment 1: EEG -> HbO Dynamical Reconstruction (37-fold Blind LOSO)",
        "| Model | r_evoked_abs | r_top25_abs | r_top25_signed | NRMSE_evoked | Peak Lat Error (s) | HC->ICH r_top25_abs |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    exp1 = metrics["experiment_1"]
    hc_ich = exp1["HC_to_ICH_Transfer"]
    for m in ["Ridge-FIR", "Canonical-HRF", "BlackBox-TCN", "Physio-NVC-Net (w/o Asym Graph)", "Physio-NVC-Net (Full)"]:
        s = exp1[m]["all"]
        tr_val = hc_ich[m]["r_top25_abs"]["mean"] if m in hc_ich else float("nan")
        lines.append(
            f"| {m} | {s['r_evoked_abs']['mean']:.3f} ± {s['r_evoked_abs']['std']:.3f} | "
            f"{s['r_top25_abs']['mean']:.3f} ± {s['r_top25_abs']['std']:.3f} | "
            f"{s['r_top25_signed']['mean']:+.3f} ± {s['r_top25_signed']['std']:.3f} | "
            f"{s['nrmse_evoked']['mean']:.3f} | {s['peak_lat_err_s']['mean']:.2f} | "
            f"{tr_val:.3f} |"
        )

    lines += [
        "",
        "## 2. Experiment 2: NVC Uncoupling Biomarkers & Clinical Correlation",
        f"- **LOSO Pathology Detection (HC n=17 vs ICH n=20, lesion-blind)**: "
        f"NVC Biomarkers ROC-AUC = `{metrics['experiment_2']['loso_pathology']['nvc_auc']:.3f}` "
        f"(Acc = `{metrics['experiment_2']['loso_pathology']['nvc_acc']*100:.1f}%`) vs EEG-only bio AUC = "
        f"`{metrics['experiment_2']['loso_pathology']['eeg_auc']:.3f}`",
        "",
        "| Metric | HC (n=17) | ICH (n=20) | MWU p | Cohen's d |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for k, v in metrics["experiment_2"]["group_tests"].items():
        lines.append(
            f"| {k} | {v['hc_mean']:.4f} ± {v['hc_std']:.4f} | {v['ich_mean']:.4f} ± {v['ich_std']:.4f} | "
            f"{v['mwu_p']:.3e} | {v['cohens_d']:+.2f} |"
        )

    lines += [
        "",
        "## 3. Experiment 3: Single-EEG + Virtual-HbO MI Decoding",
        "| Setting | EEG-Only | EEG + Canon HRF | EEG + Physio-NVC-Net Virtual HbO | Oracle (Real EEG+HbO) | Wilcoxon p (Ours vs EEG) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]
    for k_lbl, k_dict in metrics["experiment_3"]["few_shot"].items():
        lines.append(
            f"| Few-Shot {k_lbl} (All) | {k_dict['EEG-Only']['all_mean']*100:.2f}% | "
            f"{k_dict['EEG + Canonical HRF']['all_mean']*100:.2f}% | "
            f"**{k_dict['EEG + Physio-NVC-Net Virtual HbO']['all_mean']*100:.2f}%** | "
            f"{k_dict['Oracle Hybrid (Real EEG + Real HbO)']['all_mean']*100:.2f}% | "
            f"{k_dict['wilcoxon_p_all_vs_eeg']:.4f} |"
        )
    cv5 = metrics["experiment_3"]["cv_5fold"]
    lines.append(
        f"| 5-Fold CV (All) | {cv5['EEG-Only']['all_mean']*100:.2f}% | "
        f"{cv5['EEG + Canonical HRF']['all_mean']*100:.2f}% | "
        f"**{cv5['EEG + Physio-NVC-Net Virtual HbO']['all_mean']*100:.2f}%** | "
        f"{cv5['Oracle Hybrid (Real EEG + Real HbO)']['all_mean']*100:.2f}% | "
        f"{cv5['wilcoxon_p_all_vs_eeg']:.4f} |"
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.environ["MNE_LOGGING_LEVEL"] = "WARNING"
    set_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    records = load_hefmi_dataset()
    exp1_summary, oof_store = run_experiment_1(records, device=device)
    exp2_summary, bio_df = run_experiment_2(records, oof_store["physio_bio_params"])
    exp3_summary = run_experiment_3(
        records,
        oof_canon=oof_store["Canonical-HRF"],
        oof_physio=oof_store["Physio-NVC-Net (Full)"],
    )

    # Save CSV, JSON, NPZ, and Markdown report
    bio_df.to_csv(os.path.join(RESULTS_DIR, "subject_nvc_biomarkers.csv"), index=False)
    np.savez_compressed(
        os.path.join(RESULTS_DIR, "route_a_predictions.npz"),
        y_true=np.stack([r.y_hbo for r in records]),
        labels=np.stack([r.labels for r in records]),
        oof_ridge=oof_store["Ridge-FIR"],
        oof_canon=oof_store["Canonical-HRF"],
        oof_tcn=oof_store["BlackBox-TCN"],
        oof_no_asym=oof_store["Physio-NVC-Net (w/o Asym Graph)"],
        oof_physio=oof_store["Physio-NVC-Net (Full)"],
    )

    metrics_path = os.path.join(RESULTS_DIR, "route_a_metrics.json")
    full_metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            full_metrics = json.load(f)
    full_metrics.update({
        "experiment_1": exp1_summary,
        "experiment_2": exp2_summary,
        "experiment_3": exp3_summary,
    })
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(full_metrics, f, indent=2)

    if "experiment_1_v2" not in full_metrics:
        write_markdown_report(full_metrics, os.path.join(RESULTS_DIR, "route_a_report.md"))
    print(f"\nStage-1 LOSO artifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
