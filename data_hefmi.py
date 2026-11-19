"""Data loader and physiological preprocessing pipeline for HEFMI-ICH-cache.

Handles:
1. Loading all 37 subjects (17 Healthy Controls, 20 ICH Patients) + verified clinical metadata.
2. Detecting & repairing extreme bad channels (> 5.0 * median channel std) and MAD winsorizing.
3. Euclidean Alignment (EA) on cleaned sensorimotor EEG.
4. Multi-band (theta 4-8 Hz, mu 8-13 Hz, beta 13-30 Hz) 10 Hz EEG envelope extraction.
5. Mandatory uM scaling (x1e6) and 0-1 s baseline correction on HbO to prevent alternating-trial tail leakage.
6. Hemispheric (Left 0:45, Right 45:90) and lesion-relative (Ipsilesional vs Contralesional) indexing.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CACHE_DIR = r"E:\bci_data\HEFMI-ICH-cache"
META_CSV = os.path.join(CACHE_DIR, "dataset_metadata", "clinical_metadata_verified.csv")

EEG_FS = 100.0
FNIRS_FS = 10.0
BANDS = {
    "theta": (4.0, 8.0),
    "mu": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


@dataclass
class SubjectRecord:
    file_name: str
    group: str                 # "Normals" or "Patients"
    is_patient: bool
    subject_id: int
    lesion_side: str           # "None", "Left", "Right", "Brainstem"
    x_broadband: np.ndarray    # (30, 32, 1000) cleaned & MAD-clipped
    x_sm_ea: np.ndarray        # (30, 32, 1000) cleaned, MAD-clipped & EA-aligned
    eeg_bands: np.ndarray      # (30, 32, 3, 100) 10 Hz theta/mu/beta envelopes (baseline-corr)
    u_env: np.ndarray          # (30, 90, 100) cleaned, MAD-scaled, 0-1s baseline-corr
    u_env_raw_clean: np.ndarray  # (30, 90, 100) cleaned physical scale (for gain ratio)
    y_hbo: np.ndarray          # (30, 90, 100) uM, 0-1s baseline-corrected & lightly smoothed
    y_hbo_raw_bc: np.ndarray   # (30, 90, 100) uM, 0-1s baseline-corrected (unsmoothed)
    labels: np.ndarray         # (30,) int {0, 1}
    good_x_mask: np.ndarray    # (32,) bool
    good_u_mask: np.ndarray    # (90,) bool
    clean_x_std: float
    clean_u_std: float
    hbo_std: float
    clinical: Dict[str, object]
    x_broadband_raw: Optional[np.ndarray] = None  # (30, 32, 1000) unclipped raw broadband EEG
    x_sm_raw: Optional[np.ndarray] = None         # (30, 32, 1000) unaligned raw sensorimotor EEG
    u_raw: Optional[np.ndarray] = None            # (30, 90, 100) unscaled raw EEG projection
    y_hbo_raw_uM: Optional[np.ndarray] = None     # (30, 90, 100) uM before baseline subtraction


def fit_apply_bad_channel_repair(
    arr_tr: np.ndarray, arr_te: np.ndarray, threshold_mult: float = 5.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit bad-channel mask strictly on training fold arr_tr and repair both tr and te."""
    ch_std = arr_tr.std(axis=(0, 2))
    med_std = float(np.median(ch_std))
    good_mask = ch_std <= (threshold_mult * max(med_std, 1e-8))
    out_tr = arr_tr.copy()
    out_te = arr_te.copy()
    if not np.all(good_mask) and np.any(good_mask):
        ref_tr = np.median(out_tr[:, good_mask, :], axis=1, keepdims=True)
        ref_te = np.median(out_te[:, good_mask, :], axis=1, keepdims=True)
        bad_idx = np.where(~good_mask)[0]
        for idx in bad_idx:
            out_tr[:, idx, :] = ref_tr[:, 0, :]
            out_te[:, idx, :] = ref_te[:, 0, :]
    return out_tr, out_te, good_mask


def fit_apply_mad_clip(
    arr_tr: np.ndarray, arr_te: np.ndarray, clip_val: float = 8.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit per-channel MAD center and scale strictly on training fold and apply to tr and te."""
    med = np.median(arr_tr, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(arr_tr - med), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
    tr_norm = np.clip((arr_tr - med) / mad, -clip_val, clip_val)
    te_norm = np.clip((arr_te - med) / mad, -clip_val, clip_val)
    return tr_norm, te_norm


def fit_apply_euclidean_alignment(
    x_tr: np.ndarray, x_te: np.ndarray, reg: float = 1e-4
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit EA reference whitening matrix strictly on training trials x_tr and transform both."""
    covs_tr = np.einsum("nct,ndt->ncd", x_tr, x_tr) / float(x_tr.shape[2])
    r_mean = covs_tr.mean(axis=0) + reg * np.eye(x_tr.shape[1], dtype=x_tr.dtype)
    vals, vecs = np.linalg.eigh(r_mean)
    vals = np.maximum(vals, 1e-6)
    r_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
    return (
        np.einsum("cd,ndt->nct", r_inv_sqrt, x_tr),
        np.einsum("cd,ndt->nct", r_inv_sqrt, x_te),
    )


def _compute_single_trial_band_rms_bc(
    x_bb: np.ndarray, fs: float = EEG_FS, target_len: int = 100
) -> np.ndarray:
    """Compute per-trial zero-phase FFT bandpass + 10 Hz RMS + 0-1s baseline subtraction."""
    n_trials, n_ch, n_t = x_bb.shape
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    spec = np.fft.rfft(x_bb, axis=-1)
    bin_size = n_t // target_len
    band_envs = []
    for lo, hi in BANDS.values():
        mask = np.zeros_like(freqs)
        mask[(freqs >= lo) & (freqs <= hi)] = 1.0
        low_edge = (freqs >= max(0.5, lo - 1.0)) & (freqs < lo)
        high_edge = (freqs > hi) & (freqs <= hi + 1.0)
        if np.any(low_edge):
            mask[low_edge] = 0.5 * (1.0 - np.cos(np.pi * (freqs[low_edge] - (lo - 1.0))))
        if np.any(high_edge):
            mask[high_edge] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_edge] - hi)))
        filtered = np.fft.irfft(spec * mask[None, None, :], n=n_t, axis=-1)
        rms_10hz = np.sqrt(
            np.mean(filtered.reshape(n_trials, n_ch, target_len, bin_size) ** 2, axis=-1) + 1e-10
        )
        rms_bc = rms_10hz - rms_10hz[:, :, :10].mean(axis=2, keepdims=True)
        band_envs.append(rms_bc)
    return np.stack(band_envs, axis=2)


def fit_apply_fold_eeg_preprocessing(
    rec: SubjectRecord, tr: np.ndarray, te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Strictly inductive in-fold EEG bad-channel repair, MAD clipping, EA, and band scaling."""
    x_sm_raw = rec.x_sm_raw if rec.x_sm_raw is not None else rec.x_sm_ea
    x_bb_raw = rec.x_broadband_raw if rec.x_broadband_raw is not None else rec.x_broadband

    sm_tr_c, sm_te_c, _ = fit_apply_bad_channel_repair(x_sm_raw[tr].astype(np.float64), x_sm_raw[te].astype(np.float64))
    sm_tr_m, sm_te_m = fit_apply_mad_clip(sm_tr_c, sm_te_c, clip_val=8.0)
    sm_tr_ea, sm_te_ea = fit_apply_euclidean_alignment(sm_tr_m, sm_te_m, reg=1e-4)

    bb_tr_c, bb_te_c, _ = fit_apply_bad_channel_repair(x_bb_raw[tr].astype(np.float64), x_bb_raw[te].astype(np.float64))
    bb_tr_m, bb_te_m = fit_apply_mad_clip(bb_tr_c, bb_te_c, clip_val=8.0)
    b_tr_raw = _compute_single_trial_band_rms_bc(bb_tr_m)
    b_te_raw = _compute_single_trial_band_rms_bc(bb_te_m)
    b_tr_out = np.zeros_like(b_tr_raw, dtype=np.float32)
    b_te_out = np.zeros_like(b_te_raw, dtype=np.float32)
    for b_idx in range(b_tr_raw.shape[2]):
        scale = np.median(np.abs(b_tr_raw[:, :, b_idx, :]), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
        b_tr_out[:, :, b_idx, :] = np.clip(b_tr_raw[:, :, b_idx, :] / scale, -8.0, 8.0)
        b_te_out[:, :, b_idx, :] = np.clip(b_te_raw[:, :, b_idx, :] / scale, -8.0, 8.0)

    return sm_tr_ea.astype(np.float64), sm_te_ea.astype(np.float64), b_tr_out, b_te_out


def fit_apply_fold_u_preprocessing(
    rec: SubjectRecord, tr: np.ndarray, te: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Strictly inductive in-fold bad-channel repair and MAD scaling for 90-ch EEG envelopes."""
    u_raw = rec.u_raw if rec.u_raw is not None else rec.u_env
    u_tr_c, u_te_c, _ = fit_apply_bad_channel_repair(u_raw[tr].astype(np.float64), u_raw[te].astype(np.float64))
    u_tr_bc = u_tr_c - u_tr_c[:, :, :10].mean(axis=2, keepdims=True)
    u_te_bc = u_te_c - u_te_c[:, :, :10].mean(axis=2, keepdims=True)
    scale = np.median(np.abs(u_tr_bc), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
    return (
        np.clip(u_tr_bc / scale, -8.0, 8.0).astype(np.float32),
        np.clip(u_te_bc / scale, -8.0, 8.0).astype(np.float32),
    )


def _detect_and_repair_bad_channels(
    arr: np.ndarray, threshold_mult: float = 5.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Detect channels with std > threshold_mult * median(channel_std) and replace with normal median."""
    ch_std = arr.std(axis=(0, 2))
    med_std = float(np.median(ch_std))
    good_mask = ch_std <= (threshold_mult * max(med_std, 1e-8))
    out = arr.copy()
    if not np.all(good_mask) and np.any(good_mask):
        ref_signal = np.median(out[:, good_mask, :], axis=1, keepdims=True)
        bad_idx = np.where(~good_mask)[0]
        for idx in bad_idx:
            out[:, idx, :] = ref_signal[:, 0, :]
    return out, good_mask


def _mad_normalize_and_clip(
    arr: np.ndarray, clip_val: float = 8.0
) -> np.ndarray:
    """Robust per-channel MAD standardization followed by symmetric clipping."""
    med = np.median(arr, axis=(0, 2), keepdims=True)
    mad = np.median(np.abs(arr - med), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
    return np.clip((arr - med) / mad, -clip_val, clip_val)


def _euclidean_align(x: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Apply Euclidean Alignment (EA) across trials of a single subject."""
    # x: (N, C, T)
    covs = np.einsum("nct,ndt->ncd", x, x) / float(x.shape[2])
    r_mean = covs.mean(axis=0) + reg * np.eye(x.shape[1], dtype=x.dtype)
    vals, vecs = np.linalg.eigh(r_mean)
    vals = np.maximum(vals, 1e-6)
    r_inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
    return np.einsum("cd,ndt->nct", r_inv_sqrt, x)


def _extract_multiband_envelopes(
    x_bb: np.ndarray, fs: float = EEG_FS, target_len: int = 100
) -> np.ndarray:
    """Extract 10 Hz baseline-corrected RMS envelopes for theta, mu, beta bands via FFT bandpass."""
    n_trials, n_ch, n_t = x_bb.shape
    freqs = np.fft.rfftfreq(n_t, d=1.0 / fs)
    spec = np.fft.rfft(x_bb, axis=-1)
    bin_size = n_t // target_len

    band_envs = []
    for lo, hi in BANDS.values():
        # Smooth cosine-tapered frequency mask to minimize ringing
        mask = np.zeros_like(freqs)
        in_band = (freqs >= lo) & (freqs <= hi)
        mask[in_band] = 1.0
        # 1 Hz transition edges
        low_edge = (freqs >= max(0.5, lo - 1.0)) & (freqs < lo)
        high_edge = (freqs > hi) & (freqs <= hi + 1.0)
        if np.any(low_edge):
            mask[low_edge] = 0.5 * (1.0 - np.cos(np.pi * (freqs[low_edge] - (lo - 1.0))))
        if np.any(high_edge):
            mask[high_edge] = 0.5 * (1.0 + np.cos(np.pi * (freqs[high_edge] - hi)))

        filtered = np.fft.irfft(spec * mask[None, None, :], n=n_t, axis=-1)
        rms_10hz = np.sqrt(
            np.mean(filtered.reshape(n_trials, n_ch, target_len, bin_size) ** 2, axis=-1) + 1e-10
        )
        # Subtract 0-1s baseline and normalize by subject-channel robust scale
        rms_bc = rms_10hz - rms_10hz[:, :, :10].mean(axis=2, keepdims=True)
        scale = np.median(np.abs(rms_bc), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
        band_envs.append(np.clip(rms_bc / scale, -8.0, 8.0))

    return np.stack(band_envs, axis=2).astype(np.float32)  # (N, 32, 3, 100)


def get_hemisphere_indices(lesion_side: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (ipsilesional_idx, contralesional_idx) within the 90 fNIRS channels.

    Channels 0..44 correspond to Left hemisphere, 45..89 correspond to Right hemisphere.
    For HC ('None') or 'Brainstem', returns (0:45, 45:90) by convention.
    """
    left = np.arange(0, 45)
    right = np.arange(45, 90)
    if lesion_side == "Left":
        return left, right
    if lesion_side == "Right":
        return right, left
    return left, right


def load_hefmi_dataset(cache_dir: str = CACHE_DIR, meta_csv: str = META_CSV) -> List[SubjectRecord]:
    """Load and preprocess all 37 HEFMI-ICH subjects with verified clinical metadata."""
    meta_df = pd.read_csv(meta_csv)
    ich_meta = meta_df[meta_df["dataset"] == "HEFMI-ICH"].set_index("subject_num")

    npz_paths = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
    if not npz_paths:
        raise FileNotFoundError(f"No .npz files found in {cache_dir}")

    records: List[SubjectRecord] = []
    for p in npz_paths:
        d = np.load(p, allow_pickle=True)
        fn = os.path.basename(p)
        group = str(d["subject_group"])
        is_patient = group == "Patients"
        sub_id = int(d["subject_id"])
        lesion_side = str(d["lesion_side"])
        labels = d["y"].astype(np.int64)

        x_bb_raw = d["x_broadband"].astype(np.float64)
        x_sm_raw = d["x_sensorimotor"].astype(np.float64)
        u_raw = d["u_eeg_env"].astype(np.float64)
        y_hbo_uM = d["y_hbo"].astype(np.float64) * 1e6

        # 1. Bad-channel detection & repair
        x_bb_clean, good_bb = _detect_and_repair_bad_channels(x_bb_raw)
        x_sm_clean, good_sm = _detect_and_repair_bad_channels(x_sm_raw)
        good_x = good_bb & good_sm
        u_clean, good_u = _detect_and_repair_bad_channels(u_raw)

        clean_x_std = float(x_sm_raw[:, good_sm, :].std())
        clean_u_std = float(u_raw[:, good_u, :].std())

        # 2. Robust normalization + Euclidean Alignment on EEG
        x_bb_rob = _mad_normalize_and_clip(x_bb_clean, clip_val=8.0)
        x_sm_rob = _mad_normalize_and_clip(x_sm_clean, clip_val=8.0)
        x_sm_ea = _euclidean_align(x_sm_rob)

        # 3. 10 Hz Multi-band envelopes from cleaned broadband EEG
        eeg_bands = _extract_multiband_envelopes(x_bb_rob, fs=EEG_FS, target_len=100)

        # 4. u_eeg_env baseline subtraction & robust scale (preserving sign & relative dynamics)
        u_bc = u_clean - u_clean[:, :, :10].mean(axis=2, keepdims=True)
        u_scale = np.median(np.abs(u_bc), axis=(0, 2), keepdims=True) * 1.4826 + 1e-6
        u_env_norm = np.clip(u_bc / u_scale, -8.0, 8.0)

        # 5. Mandatory 0-1 s baseline subtraction on y_hbo (uM) + Savitzky-Golay smoothing
        y_hbo_bc = y_hbo_uM - y_hbo_uM[:, :, :10].mean(axis=2, keepdims=True)
        y_hbo_smooth = savgol_filter(y_hbo_bc, window_length=5, polyorder=2, axis=-1)
        hbo_std = float(y_hbo_bc.std())

        clinical: Dict[str, object] = {}
        if is_patient and sub_id in ich_meta.index:
            row = ich_meta.loc[sub_id]
            clinical = {
                "age": float(row["age"]),
                "gender": str(row["gender"]),
                "paralysis_side": str(row["paralysis_side"]),
                "lesion_hemisphere": str(row["lesion_hemisphere"]),
                "stroke_location": str(row["stroke_location"]),
                "fma_ue": float(row["score_primary"]),
                "mrs": float(row["score_secondary"]),
                "mbi": float(row["mbi"]),
            }

        records.append(
            SubjectRecord(
                file_name=fn,
                group=group,
                is_patient=is_patient,
                subject_id=sub_id,
                lesion_side=lesion_side,
                x_broadband=x_bb_rob.astype(np.float32),
                x_sm_ea=x_sm_ea.astype(np.float32),
                eeg_bands=eeg_bands.astype(np.float32),
                u_env=u_env_norm.astype(np.float32),
                u_env_raw_clean=u_bc.astype(np.float32),
                y_hbo=y_hbo_smooth.astype(np.float32),
                y_hbo_raw_bc=y_hbo_bc.astype(np.float32),
                labels=labels,
                good_x_mask=good_x,
                good_u_mask=good_u,
                clean_x_std=clean_x_std,
                clean_u_std=clean_u_std,
                hbo_std=hbo_std,
                clinical=clinical,
                x_broadband_raw=x_bb_raw.astype(np.float32),
                x_sm_raw=x_sm_raw.astype(np.float32),
                u_raw=u_raw.astype(np.float32),
                y_hbo_raw_uM=y_hbo_uM.astype(np.float32),
            )
        )

    return records


if __name__ == "__main__":
    recs = load_hefmi_dataset()
    n_hc = sum(not r.is_patient for r in recs)
    n_pt = sum(r.is_patient for r in recs)
    print(f"Loaded {len(recs)} subjects (HC={n_hc}, ICH={n_pt})")
    r0 = recs[0]
    print(
        f"Sample {r0.file_name}: x_sm_ea={r0.x_sm_ea.shape}, eeg_bands={r0.eeg_bands.shape}, "
        f"u_env={r0.u_env.shape}, y_hbo={r0.y_hbo.shape}, clean_gain={r0.hbo_std / r0.clean_u_std:.4f}"
    )
    pts = [r for r in recs if r.is_patient]
    print(f"Patients with clinical metadata: {sum(bool(r.clinical) for r in pts)}/{len(pts)}")
