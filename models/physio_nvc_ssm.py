"""Physio-NVC / EB-NVC: Mode-Exact Neurovascular Coupling Model.

Implements:
1. build_gamma_hrf_bank & MultiBasisHRFKernel: exact causal convolution against K=5
   Gamma hemodynamic bases with exact peak latencies (modes) tau_k in {2.0, 3.25, 4.5, 5.75, 7.0} s
   and unified dispersion sigma = 1.1 s:
     b_k = (-tau_k + sqrt(tau_k^2 + 4*sigma^2)) / 2,  a_k = 1 + tau_k / b_k
   guaranteeing mode (a_k - 1)*b_k = tau_k and variance a_k*b_k^2 = sigma^2.
2. LesionAwareSpatialGraphModulator: predicts channel-wise NVC basis weights (yielding
   interpretable gain alpha_c and mode latency tau_c) and models homologous inter-hemispheric
   (Left 0:45 <-> Right 45:90) compensation with sharp identity-anchored graph weights.
3. StateSpaceUncouplingResidual: compact bounded causal dilated 1D-CNN / diagonal SSM
   residual capturing pathological vascular uncoupling beyond linear HRF convolution.
4. Comparison baselines: CanonicalHRFBaseline and BlackBoxTCN.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_gamma_hrf_bank(
    taus: Tuple[float, ...] = (2.0, 3.25, 4.5, 5.75, 7.0),
    sigma: float = 1.1,
    fs: float = 10.0,
    kernel_len: int = 70,
) -> torch.Tensor:
    """Build K normalized causal Gamma HRF kernels with exact peak latencies (modes) taus."""
    t = torch.arange(kernel_len, dtype=torch.float32) / fs
    t = torch.clamp(t, min=1e-4)
    kernels = []
    for tau in taus:
        # Exact mode parameterization: (a - 1) * b = tau and a * b^2 = sigma^2
        b = (-tau + math.sqrt(tau * tau + 4.0 * sigma * sigma)) / 2.0
        a = 1.0 + tau / b
        log_k = (a - 1.0) * torch.log(t) - (t / b)
        log_k = log_k - log_k.max()
        k = torch.exp(log_k)
        k[0] = 0.0
        k = k / (k.sum() + 1e-8)
        kernels.append(k)
    return torch.stack(kernels, dim=0)  # (K, L)


class MultiBasisHRFKernel(nn.Module):
    """Causal multi-basis Gamma HRF convolution operator."""

    def __init__(
        self,
        taus: Tuple[float, ...] = (2.0, 3.25, 4.5, 5.75, 7.0),
        sigma: float = 1.1,
        fs: float = 10.0,
        kernel_len: int = 70,
    ) -> None:
        super().__init__()
        self.kernel_len = kernel_len
        bank = build_gamma_hrf_bank(taus=taus, sigma=sigma, fs=fs, kernel_len=kernel_len)
        self.register_buffer("basis", bank)  # (K, L)
        self.register_buffer("basis_flipped", torch.flip(bank, dims=[-1]).unsqueeze(1))  # (K, 1, L)
        self.register_buffer("taus", torch.tensor(taus, dtype=torch.float32))  # (K,)

    def forward(self, driver: torch.Tensor) -> torch.Tensor:
        """Convolve neural driver (B, C, T) causally with K HRF bases -> (B, C, K, T)."""
        b_sz, n_ch, n_t = driver.shape
        x = driver.reshape(b_sz * n_ch, 1, n_t)
        x_pad = F.pad(x, (self.kernel_len - 1, 0))
        out = F.conv1d(x_pad, self.basis_flipped)  # (B*C, K, T)
        return out.reshape(b_sz, n_ch, len(self.taus), n_t)


class LesionAwareSpatialGraphModulator(nn.Module):
    """Predicts per-channel HRF weights and models inter-hemispheric compensation."""

    def __init__(self, n_fnirs: int = 90, n_bases: int = 5, hidden_dim: int = 48) -> None:
        super().__init__()
        self.n_fnirs = n_fnirs
        self.n_hemi = n_fnirs // 2  # 45 Left, 45 Right
        self.n_bases = n_bases

        self.ch_embed = nn.Parameter(torch.randn(1, n_fnirs, 8) * 0.02)
        self.param_mlp = nn.Sequential(
            nn.Linear(4 + 8 + 4, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bases),
        )
        # Initialize close to canonical HRF centered at middle basis (tau=4.5 s)
        nn.init.zeros_(self.param_mlp[-1].weight)
        init_bias = torch.tensor([0.05, 0.15, 0.60, 0.15, 0.05])
        with torch.no_grad():
            self.param_mlp[-1].bias.copy_(init_bias)

        # Residual intra-hemisphere spatial smoothing anchored strongly at identity
        self.intra_left = nn.Parameter(torch.eye(self.n_hemi) * 6.0)
        self.intra_right = nn.Parameter(torch.eye(self.n_hemi) * 6.0)
        self.contra_gate = nn.Sequential(
            nn.Linear(4 + 8 + 4, 24),
            nn.GELU(),
            nn.Linear(24, 1),
            nn.Tanh(),
        )
        nn.init.zeros_(self.contra_gate[-2].weight)
        nn.init.zeros_(self.contra_gate[-2].bias)

    def forward(
        self,
        z_hrf: torch.Tensor,
        u_env: torch.Tensor,
        lesion_onehot: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        b_sz = z_hrf.shape[0]
        if lesion_onehot is None:
            lesion_feat = torch.zeros(b_sz, self.n_fnirs, 4, device=z_hrf.device, dtype=z_hrf.dtype)
        else:
            lesion_feat = lesion_onehot.unsqueeze(1).expand(b_sz, self.n_fnirs, 4)

        # Unsupervised channel descriptors (B, 90, 4)
        u_std = u_env.std(dim=-1, keepdim=True)
        u_rms = torch.sqrt(torch.mean(u_env ** 2, dim=-1, keepdim=True) + 1e-6)
        u_late = u_env[:, :, 20:80].mean(dim=-1, keepdim=True)
        u_homolog = torch.cat([u_env[:, self.n_hemi :, :], u_env[:, : self.n_hemi, :]], dim=1)
        u_asym = u_std - u_homolog.std(dim=-1, keepdim=True)

        ch_desc = torch.cat(
            [u_std, u_rms, u_late, u_asym, self.ch_embed.expand(b_sz, -1, -1), lesion_feat],
            dim=-1,
        )  # (B, 90, 16)

        weights = self.param_mlp(ch_desc)  # (B, 90, K)
        y_local = torch.einsum("bckt,bck->bct", z_hrf, weights)

        # Anchor intra-hemispheric graph strongly on self-channel (~90% self, ~10% neighbors)
        w_l = F.softmax(self.intra_left, dim=-1)
        w_r = F.softmax(self.intra_right, dim=-1)
        y_l = torch.einsum("ij,bjt->bit", w_l, y_local[:, : self.n_hemi, :])
        y_r = torch.einsum("ij,bjt->bit", w_r, y_local[:, self.n_hemi :, :])
        y_intra = torch.cat([y_l, y_r], dim=1)

        y_homolog = torch.cat([y_intra[:, self.n_hemi :, :], y_intra[:, : self.n_hemi, :]], dim=1)
        g_contra = self.contra_gate(ch_desc)  # (B, 90, 1)
        y_graph = y_intra + 0.25 * g_contra * y_homolog

        return y_graph, {
            "weights": weights,
            "contra_gate": g_contra.squeeze(-1),
        }


class DiagonalSSMBlock(nn.Module):
    """Lightweight causal dilated residual block approximating a diagonal state-space filter."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = 5
        self.pad = (self.kernel_size - 1) * dilation
        self.conv_depth = nn.Conv1d(
            channels, channels, kernel_size=self.kernel_size, dilation=dilation, groups=channels
        )
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.pw_in = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.pw_out = nn.Conv1d(channels, channels, kernel_size=1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.pad(x, (self.pad, 0))
        h = self.conv_depth(h)
        h = self.norm(h)
        a, b = self.pw_in(h).chunk(2, dim=1)
        h = a * torch.sigmoid(b)
        h = self.drop(self.pw_out(h))
        return x + h


class StateSpaceUncouplingResidual(nn.Module):
    """Causal multi-scale SSM residual capturing pathological neurovascular uncoupling."""

    def __init__(self, n_fnirs: int = 90, d_model: int = 48) -> None:
        super().__init__()
        self.in_proj = nn.Conv1d(n_fnirs, d_model, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                DiagonalSSMBlock(d_model, dilation=1, dropout=0.15),
                DiagonalSSMBlock(d_model, dilation=2, dropout=0.15),
                DiagonalSSMBlock(d_model, dilation=4, dropout=0.15),
            ]
        )
        self.out_proj = nn.Conv1d(d_model, n_fnirs, kernel_size=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, driver: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(driver)
        for blk in self.blocks:
            h = blk(h)
        return self.out_proj(h)


class PhysioNVCNet(nn.Module):
    """Lesion-Aware Differentiable Neurovascular Coupling & EEG->HbO Reconstruction Network."""

    def __init__(
        self,
        n_eeg: int = 32,
        n_bands: int = 3,
        n_fnirs: int = 90,
        taus: Tuple[float, ...] = (2.0, 3.25, 4.5, 5.75, 7.0),
        use_asymmetry_graph: bool = True,
        use_ssm_residual: bool = True,
    ) -> None:
        super().__init__()
        self.n_fnirs = n_fnirs
        self.use_asymmetry_graph = use_asymmetry_graph
        self.use_ssm_residual = use_ssm_residual

        # Residual spatial refinement from multi-band EEG (anchored cleanly on u_env)
        self.eeg_spatial_proj = nn.Sequential(
            nn.Conv1d(n_eeg * n_bands, n_fnirs, kernel_size=1),
            nn.Tanh(),
        )
        self.eeg_res_gate = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ~ 0.12 initial blend

        self.hrf_kernel = MultiBasisHRFKernel(taus=taus)
        self.canon_hrf = MultiBasisHRFKernel(taus=(4.5,), sigma=1.1)
        self.canon_scale = nn.Parameter(torch.ones(1, n_fnirs, 1) * 0.5)
        self.graph_mod = LesionAwareSpatialGraphModulator(n_fnirs=n_fnirs, n_bases=len(taus))
        self.ablation_weights = nn.Parameter(torch.tensor([[[0.05, 0.15, 0.60, 0.15, 0.05]]]).repeat(1, n_fnirs, 1))

        self.ssm_residual = StateSpaceUncouplingResidual(n_fnirs=n_fnirs, d_model=48)
        self.res_gate = nn.Parameter(torch.ones(1, n_fnirs, 1) * -1.5)  # bounded residual gate <= 0.35

    def forward(
        self,
        u_env: torch.Tensor,
        eeg_bands: torch.Tensor,
        lesion_onehot: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        b_sz, n_eeg, n_bands, n_t = eeg_bands.shape
        e_flat = eeg_bands.reshape(b_sz, n_eeg * n_bands, n_t)
        e_drv = self.eeg_spatial_proj(e_flat)

        # Keep u_env as primary clean spatial-aligned driver; multi-band EEG adds small bounded refinement
        driver = u_env + 0.25 * torch.sigmoid(self.eeg_res_gate) * e_drv

        z_hrf = self.hrf_kernel(driver)  # (B, 90, K, 100)

        if self.use_asymmetry_graph:
            y_graph, mod_info = self.graph_mod(z_hrf, driver, lesion_onehot=lesion_onehot)
            y_back = torch.einsum("bckt,bck->bct", z_hrf, self.ablation_weights.expand(b_sz, -1, -1))
            hemi_diff = y_graph - torch.roll(y_graph, shifts=self.n_fnirs // 2, dims=1)
            y_hrf = y_back - 0.15 * hemi_diff
            weights = mod_info["weights"]
        else:
            weights = self.ablation_weights.expand(b_sz, -1, -1)
            y_hrf = torch.einsum("bckt,bck->bct", z_hrf, weights)
            mod_info = {"contra_gate": torch.zeros(b_sz, self.n_fnirs, device=u_env.device)}

        if self.use_ssm_residual:
            res_raw = self.ssm_residual(driver)
            # Bounded residual modulation so HRF physics stays the dominant backbone
            y_res = 0.35 * torch.sigmoid(self.res_gate) * res_raw
        else:
            y_res = torch.zeros_like(y_hrf)

        y_pred = y_hrf + y_res
        y_pred = y_pred - y_pred[:, :, :10].mean(dim=-1, keepdim=True)
        y_hrf_bc = y_hrf - y_hrf[:, :, :10].mean(dim=-1, keepdim=True)

        abs_w = torch.abs(weights)
        alpha_gain = abs_w.sum(dim=-1)
        tau_probs = F.softmax(abs_w * 3.0, dim=-1)
        tau_eff = torch.einsum("bck,k->bc", tau_probs, self.hrf_kernel.taus)
        res_ratio = y_res.std(dim=-1) / (y_pred.std(dim=-1) + 1e-6)

        return {
            "y_pred": y_pred,
            "y_hrf": y_hrf_bc,
            "y_res": y_res,
            "driver": driver,
            "alpha_gain": alpha_gain,
            "tau_eff": tau_eff,
            "res_ratio": res_ratio,
            "contra_gate": mod_info["contra_gate"],
        }


class CanonicalHRFBaseline(nn.Module):
    """Fixed single Gamma HRF (mode tau=4.5 s, sigma=1.1 s) + learnable per-channel scalar gain."""

    def __init__(self, n_fnirs: int = 90) -> None:
        super().__init__()
        self.hrf = MultiBasisHRFKernel(taus=(4.5,), sigma=1.1)
        self.scale = nn.Parameter(torch.ones(1, n_fnirs, 1) * 0.5)

    def forward(
        self, u_env: torch.Tensor, eeg_bands: torch.Tensor, lesion_onehot: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        z = self.hrf(u_env).squeeze(2)
        y_pred = self.scale * z
        y_pred = y_pred - y_pred[:, :, :10].mean(dim=-1, keepdim=True)
        return {"y_pred": y_pred}


class BlackBoxTCN(nn.Module):
    """Unconstrained temporal CNN mapping EEG bands + u_env -> HbO without HRF prior."""

    def __init__(self, n_eeg: int = 32, n_bands: int = 3, n_fnirs: int = 90, hidden: int = 96) -> None:
        super().__init__()
        in_ch = n_fnirs + n_eeg * n_bands
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=7, padding=3),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=6, dilation=2),
            nn.GELU(),
            nn.GroupNorm(8, hidden),
            nn.Conv1d(hidden, hidden, kernel_size=7, padding=12, dilation=4),
            nn.GELU(),
            nn.Conv1d(hidden, n_fnirs, kernel_size=1),
        )

    def forward(
        self, u_env: torch.Tensor, eeg_bands: torch.Tensor, lesion_onehot: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        b_sz, n_eeg, n_bands, n_t = eeg_bands.shape
        x = torch.cat([u_env, eeg_bands.reshape(b_sz, n_eeg * n_bands, n_t)], dim=1)
        y_pred = self.net(x)
        y_pred = y_pred - y_pred[:, :, :10].mean(dim=-1, keepdim=True)
        return {"y_pred": y_pred}
