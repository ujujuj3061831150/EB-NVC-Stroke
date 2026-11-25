"""Publication-grade IEEE TBME figures for the HEFMI-ICH and Liu2024 manuscript."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from matplotlib.ticker import FixedLocator, NullLocator, StrMethodFormatter
import numpy as np
from scipy.stats import sem, theilslopes


# IEEE Transactions double-column width: 7.16 in = 515.52 pt
WIDTH_IN = 7.16
BLACK = "#1F1F1F"
GRAY = "#6B6B6B"
LIGHT = "#C8C8C8"
FAINT = "#ECECEC"
TEAL = "#007276"
ORANGE = "#D55E00"
PURPLE = "#7B52AB"
RED = "#C0392B"
BLUE = "#005AB5"

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8.5,
    "mathtext.fontset": "dejavusans",
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.70,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.labelpad": 2.5,
    "axes.unicode_minus": False,
    "xtick.labelsize": 7.8,
    "ytick.labelsize": 7.8,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.major.width": 0.70,
    "ytick.major.width": 0.70,
    "xtick.major.pad": 2.2,
    "ytick.major.pad": 2.2,
    "legend.fontsize": 7.6,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.borderaxespad": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def save(fig: plt.Figure, output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    try:
        from PIL import Image
        gray_path = path.with_name(f"{path.stem}_grayscale.png")
        Image.open(png_path).convert("L").save(gray_path)
    except ImportError:
        pass
    plt.close(fig)


def add_ieee_label(
    ax: plt.Axes,
    letter: str,
    title: str,
    x_offset_pt: float = -24.0,
    y_offset_pt: float = 5.0,
) -> None:
    """Place an aligned IEEE bold `(a)` panel tag and short heading above `ax`."""
    ax.annotate(
        f"({letter})",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_offset_pt, y_offset_pt),
        textcoords="offset points",
        fontsize=9.2,
        fontweight="bold",
        color=BLACK,
        ha="left",
        va="bottom",
        annotation_clip=False,
    )
    ax.annotate(
        title,
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(x_offset_pt + 18.0, y_offset_pt),
        textcoords="offset points",
        fontsize=8.6,
        color=BLACK,
        ha="left",
        va="bottom",
        annotation_clip=False,
    )


def sig_bracket(
    ax: plt.Axes,
    x1: float,
    x2: float,
    y: float,
    h: float,
    text: str,
    fs: float = 7.4,
    lw: float = 0.70,
) -> None:
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], color=BLACK, lw=lw, clip_on=False)
    ax.text((x1 + x2) / 2.0, y + h * 1.15, text, ha="center", va="bottom", fontsize=fs, color=BLACK)


def beeswarm_offsets(
    values: np.ndarray,
    y_span: float,
    dot_dx: float = 0.065,
    dot_dy: float = 0.045,
    max_half_w: float = 0.28,
) -> np.ndarray:
    """Deterministic greedy beeswarm x-offsets in data coordinates (zero vertical distortion)."""
    n = len(values)
    if n <= 1:
        return np.zeros(n, dtype=np.float64)
    order = np.argsort(values)
    offsets = np.zeros(n, dtype=np.float64)
    placed: list[tuple[float, float]] = []
    scale_y = max(float(y_span), 1e-9)
    # Candidate horizontal slots radiating outward from center
    steps = [0.0]
    k = 1
    while k * dot_dx <= max_half_w + 1e-9:
        steps.extend([-k * dot_dx, k * dot_dx])
        k += 1

    for idx in order:
        yv = float(values[idx]) / scale_y
        chosen = 0.0
        best_score = -1.0
        for cand in steps:
            ok = True
            min_dist = 1e9
            for px, py in placed:
                d = np.hypot((cand - px) / dot_dx, (yv - py) / dot_dy)
                if d < 1.0:
                    ok = False
                if d < min_dist:
                    min_dist = d
            if ok:
                chosen = cand
                break
            if min_dist > best_score:
                best_score = min_dist
                chosen = cand
        offsets[idx] = chosen
        placed.append((chosen, yv))
    return offsets


def ordinal_beeswarm(
    ax: plt.Axes,
    grades: np.ndarray,
    values: np.ndarray,
    color: str,
    y_span: float,
    marker: str = "o",
    size: float = 13.0,
) -> None:
    for grade in np.unique(grades):
        mask = grades == grade
        grp = values[mask]
        offs = beeswarm_offsets(grp, y_span=y_span, dot_dx=0.075, dot_dy=0.055, max_half_w=0.30)
        if len(grp) > 1:
            ax.hlines(
                np.median(grp),
                grade - 0.28,
                grade + 0.28,
                color=BLACK,
                lw=1.1,
                zorder=4,
            )
        ax.scatter(
            grade + offs,
            grp,
            s=size,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            lw=0.78,
            zorder=3,
        )


def styled_box_strip(
    ax: plt.Axes,
    groups: list[np.ndarray],
    colors: list[str],
    markers: list[str],
    is_log: bool = False,
) -> None:
    """Draw clean boxplot (median + IQR + 1.5*IQR whiskers) overlaid with beeswarm raw points."""
    for i, (vals, col, mk) in enumerate(zip(groups, colors, markers)):
        bp = ax.boxplot(
            [vals],
            positions=[i],
            widths=0.42,
            patch_artist=True,
            showfliers=False,
            whis=1.5,
            zorder=2,
        )
        for box in bp["boxes"]:
            box.set(facecolor=col, alpha=0.16, edgecolor=col, linewidth=0.75)
        for med in bp["medians"]:
            med.set(color=BLACK, linewidth=1.25)
        for whisk in bp["whiskers"]:
            whisk.set(color=col, linewidth=0.7, linestyle="-")
        for cap in bp["caps"]:
            cap.set(color=col, linewidth=0.7)

        y_for_swarm = np.log10(np.clip(vals, 1e-9, None)) if is_log else vals
        span = float(np.ptp(np.concatenate([np.log10(np.clip(g, 1e-9, None)) if is_log else g for g in groups])))
        offs = beeswarm_offsets(y_for_swarm, y_span=span, dot_dx=0.055, dot_dy=0.048, max_half_w=0.22)
        ax.scatter(
            i + offs,
            vals,
            s=11,
            marker=mk,
            facecolor="white",
            edgecolor=col,
            lw=0.75,
            alpha=0.95,
            zorder=3,
        )


@mpl.rc_context(STYLE)
def plot_fig1_pathology(records, bio_df, recon_res, out_path):
    """Fig. 1: 4-panel electro-vascular disparity in HEFMI-ICH (n=37)."""
    fig = plt.figure(figsize=(WIDTH_IN, 2.22))
    gs = GridSpec(
        1, 4,
        figure=fig,
        width_ratios=[1.08, 0.96, 0.96, 1.62],
        left=0.072, right=0.988, bottom=0.22, top=0.80, wspace=0.48,
    )
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[0, 2])
    d = fig.add_subplot(gs[0, 3])

    hc = bio_df[bio_df["is_patient"] == 0]
    ich = bio_df[bio_df["is_patient"] == 1]
    cohorts = [(0, "Healthy", hc, TEAL, "o"), (1, "ICH", ich, ORANGE, "^")]

    # (a) NVC ratio: half-violin (right) + IQR + median + raw points (left)
    rng = np.random.default_rng(42)
    for i, name, sub, color, marker in cohorts:
        nvc = sub["clean_gain"].to_numpy()
        violin = a.violinplot(nvc, positions=[i + 0.06], widths=0.58, showmeans=False, showmedians=False, showextrema=False)
        body = violin["bodies"][0]
        verts = body.get_paths()[0].vertices
        verts[:, 0] = np.maximum(verts[:, 0], i + 0.06)
        body.set(facecolor=color, edgecolor=color, alpha=0.22, linewidth=0.65)
        q1, med, q3 = np.quantile(nvc, [0.25, 0.5, 0.75])
        a.vlines(i + 0.06, q1, q3, color=color, lw=2.8, zorder=3)
        a.scatter(i + 0.06, med, s=16, color="white", edgecolor=color, lw=0.9, zorder=4)
        jit = rng.uniform(-0.07, 0.07, len(nvc))
        a.scatter(
            i - 0.14 + jit,
            nvc,
            s=10,
            marker=marker,
            facecolor="white",
            edgecolor=color,
            lw=0.72,
            zorder=3,
        )
    a.set(
        xlim=(-0.48, 1.48),
        ylim=(0.0, 0.47),
        yticks=[0.0, 0.1, 0.2, 0.3, 0.4],
        xticks=[0, 1],
        xticklabels=["Healthy\n(n = 17)", "ICH\n(n = 20)"],
        ylabel=r"$\hat{R}_{\mathrm{amp}}$",
    )
    sig_bracket(a, 0, 1, 0.395, 0.015, r"$p = 1.31 \times 10^{-4}$", fs=7.4)
    add_ieee_label(a, "a", "NVC ratio", x_offset_pt=-32)

    # (b) Normalized EEG amplitude (log y-axis handles 5.86 outlier without compressing main cluster)
    eeg_hc = hc["clean_x_std"].to_numpy() / hc["clean_x_std"].mean()
    eeg_ich = ich["clean_x_std"].to_numpy() / hc["clean_x_std"].mean()
    b.axhline(1.0, color=LIGHT, lw=0.7, ls="--", zorder=1)
    styled_box_strip(b, [eeg_hc, eeg_ich], [TEAL, ORANGE], ["o", "^"], is_log=True)
    b.set_yscale("log")
    b.set(
        xlim=(-0.50, 1.50),
        ylim=(0.50, 8.8),
        xticks=[0, 1],
        xticklabels=["Healthy\n(n = 17)", "ICH\n(n = 20)"],
        ylabel="EEG / healthy mean (log)",
    )
    b.yaxis.set_major_locator(FixedLocator([0.6, 1, 2, 4, 6]))
    b.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    b.yaxis.set_minor_locator(NullLocator())
    sig_bracket(b, 0, 1, 6.4, 0.65, r"$p = 0.0154$", fs=7.4)
    add_ieee_label(b, "b", "EEG amplitude", x_offset_pt=-28)

    # (c) Normalized HbO amplitude
    hbo_hc = hc["hbo_peak"].to_numpy() / hc["hbo_peak"].mean()
    hbo_ich = ich["hbo_peak"].to_numpy() / hc["hbo_peak"].mean()
    c.axhline(1.0, color=LIGHT, lw=0.7, ls="--", zorder=1)
    styled_box_strip(c, [hbo_hc, hbo_ich], [TEAL, ORANGE], ["o", "^"], is_log=False)
    c.set(
        xlim=(-0.50, 1.50),
        ylim=(0.0, 2.22),
        yticks=[0.0, 0.5, 1.0, 1.5, 2.0],
        xticks=[0, 1],
        xticklabels=["Healthy\n(n = 17)", "ICH\n(n = 20)"],
        ylabel="HbO / healthy mean",
    )
    sig_bracket(c, 0, 1, 1.88, 0.07, r"$p = 0.0100$", fs=7.4)
    add_ieee_label(c, "c", "HbO amplitude", x_offset_pt=-28)

    # (d) Hemodynamic time course
    d.axvspan(0.0, 1.0, color=FAINT, zorder=0)
    d.text(0.50, 1.48, "Base", ha="center", va="top", fontsize=6.6, color=GRAY)
    d.axvline(2.0, color=LIGHT, ls=":", lw=0.7)
    d.axvline(9.0, color=LIGHT, ls=":", lw=0.7)
    d.plot([2.0, 9.0], [1.43, 1.43], color=GRAY, lw=0.75, solid_capstyle="butt")
    d.text(5.5, 1.45, "Motor imagery", ha="center", va="bottom", fontsize=7.4, color=BLACK)

    for i, name, _, color, marker in cohorts:
        rms = np.asarray([
            np.sqrt(np.mean(r.y_hbo**2, axis=1)).mean(0)
            for r in records if int(r.is_patient) == i
        ])
        t = np.arange(rms.shape[1]) / 10.0
        center = rms.mean(axis=0)
        spread = sem(rms, axis=0)
        d.fill_between(t, center - spread, center + spread, color=color, alpha=0.18, lw=0)
        ls = "-" if i == 0 else "--"
        d.plot(
            t, center, color=color, lw=1.25, ls=ls,
            marker=marker, markevery=[20, 50, 80], markersize=3.2,
            markerfacecolor="white", markeredgewidth=0.8,
            label=f"{name} (n = {len(rms)})",
        )
    d.legend(loc="lower right", bbox_to_anchor=(0.99, 0.03), handlelength=1.8, labelspacing=0.25)
    d.set(
        xlim=(0.0, 9.9),
        ylim=(0.0, 1.62),
        xticks=[0, 2, 4, 6, 8],
        yticks=[0.0, 0.5, 1.0, 1.5],
        xlabel="Trial time (s)",
        ylabel=r"RMS $\Delta$HbO ($\mu$M)",
    )
    add_ieee_label(d, "d", "Hemodynamic time course", x_offset_pt=-26)

    save(fig, out_path)


@mpl.rc_context(STYLE)
def plot_fig2_clinical(bio_df, roc_data, nested_res, out_path):
    """Fig. 2: Nested LOSO screening & exploratory FDR clinical associations."""
    fig = plt.figure(figsize=(WIDTH_IN, 2.78))
    gs = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1.10, 1.24, 1.10],
        left=0.070, right=0.988, bottom=0.16, top=0.87, wspace=0.44,
    )
    a = fig.add_subplot(gs[0, 0])
    gs_mid = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0, 1], height_ratios=[1.05, 1.0], hspace=0.82)
    gs_hm = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_mid[0, 0], width_ratios=[1.0, 0.065], wspace=0.08)
    b = fig.add_subplot(gs_hm[0, 0])
    cax = fig.add_subplot(gs_hm[0, 1])
    c = fig.add_subplot(gs_mid[1, 0])
    d = fig.add_subplot(gs[0, 2])

    # (a) LOSO screening ROC
    a.plot([0, 1], [0, 1], color=LIGHT, ls=":", lw=0.8)
    for key, name, color, ls in [
        ("Strict Outer-Fold Closed-Form NVC", "Closed-form", TEAL, "-"),
        ("NVC Biomarkers", "Combined NVC", PURPLE, "-."),
        ("HbO-Only Features", "HbO only", ORANGE, ":"),
        ("EEG-Only Features", "EEG only", GRAY, "--"),
    ]:
        curve = roc_data[key]
        auc_v = nested_res["nested_loso"][key]["auc"]
        a.step(curve["fpr"], curve["tpr"], where="post", color=color, ls=ls, lw=1.25, label=f"{name} ({auc_v:.3f})")

    # Operating point of Strict Outer-Fold Closed-Form NVC (Sens=70.0%, Spec=94.1%)
    cf_stats = nested_res["nested_loso"]["Strict Outer-Fold Closed-Form NVC"]
    op_x = 1.0 - cf_stats["specificity"]
    op_y = cf_stats["sensitivity"]
    a.scatter([op_x], [op_y], s=26, color=TEAL, edgecolor="white", lw=0.7, zorder=5)
    a.annotate(
        f"Acc {cf_stats['nested_acc']*100:.1f}%",
        xy=(op_x, op_y),
        xytext=(0.14, 0.88),
        fontsize=7.3,
        color=TEAL,
        arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.55, shrinkA=0, shrinkB=3),
    )
    a.legend(
        loc="lower right",
        bbox_to_anchor=(1.00, 0.02),
        labelspacing=0.24,
        fontsize=6.9,
        frameon=True,
        facecolor="white",
        edgecolor="none",
        framealpha=0.88,
    )
    a.set(
        xlim=(0.0, 1.02),
        ylim=(0.0, 1.02),
        xticks=[0, 0.25, 0.5, 0.75, 1],
        yticks=[0, 0.25, 0.5, 0.75, 1],
        xlabel="1 − Specificity",
        ylabel="Sensitivity",
    )
    for axis in (a.xaxis, a.yaxis):
        axis.set_major_formatter(StrMethodFormatter("{x:g}"))
    add_ieee_label(a, "a", "LOSO screening", x_offset_pt=-32)

    # (b) Partial Spearman heatmap WITH explicit colorbar (fixes P5)
    fdr = nested_res["clinical_ich_fdr"]
    keys = ["tau_mean", "res_ipsi", "clean_gain", "delta_tau_ipsi_contra"]
    endpoints = ["mrs", "mbi", "fma_ue"]
    mat = np.zeros((len(keys), len(endpoints)))
    qmat = np.ones((len(keys), len(endpoints)))
    for i, key in enumerate(keys):
        for j, ep in enumerate(endpoints):
            row = fdr[ep][key]
            mat[i, j] = row["partial_spearman_rho"]
            qmat[i, j] = row["partial_fdr_q"]

    signed_map = LinearSegmentedColormap.from_list("partial", [BLUE, "#F7F7F7", RED])
    im = b.imshow(mat, cmap=signed_map, vmin=-1.0, vmax=1.0, aspect="auto")
    for i in range(len(keys)):
        for j in range(len(endpoints)):
            rho = mat[i, j]
            q = qmat[i, j]
            suffix = "**" if q < 0.01 else "*" if q < 0.05 else ""
            b.text(
                j, i, f"{rho:+.2f}{suffix}",
                ha="center", va="center", fontsize=7.1,
                fontweight="bold" if suffix else "normal",
                color="white" if abs(rho) > 0.62 else BLACK,
            )
    b.set_xticks(range(3))
    b.set_xticklabels(["mRS", "MBI", "FMA-UE"])
    b.set_yticks(range(4))
    b.set_yticklabels(["Mean centroid", "Ipsi. residual", "NVC ratio", "Basis-shift asym."])
    b.tick_params(length=0, pad=2)
    b.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    b.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    b.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
    b.tick_params(which="minor", bottom=False, left=False)
    for sp in b.spines.values():
        sp.set_visible(False)

    cb = fig.colorbar(im, cax=cax)
    cb.set_ticks([-1, 0, 1])
    cb.ax.tick_params(labelsize=7.0, length=2, pad=1.5)
    cb.outline.set_linewidth(0.5)
    cb.set_label(r"Partial $\rho$", fontsize=7.4, labelpad=1.5)
    add_ieee_label(b, "b", "Adjusted associations", x_offset_pt=-54, y_offset_pt=4)

    # (c) mRS in ICH (n=20: mRS 1->10, 2->0, 3->5, 4->5)
    ich = bio_df[bio_df["is_patient"] == 1]
    x_ms = ich["delta_tau_ipsi_contra"].to_numpy() * 1000.0
    y_mbi = ich["mbi"].to_numpy()
    mrs_vals = ich["mrs"].to_numpy()

    c.axhline(0.0, color=LIGHT, lw=0.65, ls="--")
    ordinal_beeswarm(c, mrs_vals, x_ms, RED, y_span=32.0, size=13.0)
    c.set(
        xlim=(0.45, 4.55),
        ylim=(-16.5, 20.5),
        xticks=[1, 2, 3, 4],
        xticklabels=["1\n(n=10)", "2\n(n=0)", "3\n(n=5)", "4\n(n=5)"],
        yticks=[-10, 0, 10],
        xlabel="mRS grade",
        ylabel="Basis shift (ms)",
    )
    sig_bracket(c, 1.0, 3.5, 14.5, 1.4, r"1 vs 3–4: $p = 0.045$", fs=7.1)
    add_ieee_label(c, "c", "mRS in ICH", x_offset_pt=-54, y_offset_pt=6)

    # (d) MBI in ICH with Theil-Sen robust regression (avoids single-point OLS distortion)
    d.axvline(0.0, color=LIGHT, lw=0.65, ls=":")
    fit_x = np.linspace(x_ms.min(), x_ms.max(), 80)
    ts_res = theilslopes(y_mbi, x_ms)
    rng = np.random.default_rng(42)
    boot_ts = []
    for _ in range(1000):
        ids = rng.integers(0, len(x_ms), size=len(x_ms))
        if np.std(x_ms[ids]) > 1e-5:
            b_res = theilslopes(y_mbi[ids], x_ms[ids])
            boot_ts.append(b_res.slope * fit_x + b_res.intercept)
    lo, hi = np.percentile(boot_ts, [2.5, 97.5], axis=0)
    d.fill_between(fit_x, lo, hi, color=RED, lw=0, alpha=0.16)
    d.plot(fit_x, ts_res.slope * fit_x + ts_res.intercept, color=RED, lw=1.15, label="Theil–Sen")
    for side, marker in [("Left", "o"), ("Right", "s"), ("Brainstem", "^")]:
        sub = ich[ich["lesion_side"] == side]
        d.scatter(
            sub["delta_tau_ipsi_contra"] * 1000.0,
            sub["mbi"],
            marker=marker,
            s=16,
            edgecolor=RED,
            facecolor="white",
            lw=0.85,
            label=side,
            zorder=3,
        )
    d.legend(loc="upper right", bbox_to_anchor=(0.99, 0.99), labelspacing=0.22, handlelength=1.1, fontsize=7.2)
    d.text(
        0.04, 0.06,
        r"$\rho_{\mathrm{part}}=-0.75^{**}$" + "\n" + r"$q=0.0036$",
        transform=d.transAxes,
        fontsize=7.3,
        color=BLACK,
        va="bottom",
        ha="left",
    )
    d.set(
        xlim=(-15.0, 14.0),
        ylim=(0.0, 106.0),
        xticks=[-10, 0, 10],
        yticks=[0, 25, 50, 75, 100],
        xlabel="Basis-shift asymmetry (ms)",
        ylabel="MBI",
    )
    add_ieee_label(d, "d", "MBI in ICH", x_offset_pt=-28)

    save(fig, out_path)


HEFMI_KEYS = (
    "1_EEG_Raw", "2_InFold_Shrinkage_EA", "3_Canon_HRF_K1", "4_EB_MultiBasis_K5",
    "5_Add_FastSlow_Cov", "6_Full_FastSlow_EntropyGated", "Oracle_Real_EEG_HbO",
)
LIU_KEYS = (
    "1_EEG_Raw_CSP", "2_InFold_Shrinkage_EA", "3_Add_Canon_HRF_K1",
    "4_Add_MultiBasis_HRF_K5", "5_Add_Joint_FastSlow_sDGF", "6_Full_FastSlow_EntropyGated",
)
STEPS = ["EA", "$K$=1", "$K$=5", "F–S", "Full", "Oracle"]


def plot_ablation_dots(ax: plt.Axes, cv: dict, block: dict, keys: tuple, folds: int, has_oracle: bool) -> None:
    """Discrete ablation dot-plot WITHOUT connecting lines across categorical models (P6-safe)."""
    base = np.asarray(cv[keys[0]]["per_subject"])
    gains = [(np.asarray(cv[key]["per_subject"]) - base) * 100.0 for key in keys[1:]]
    block_gains = [100.0 * (block[key]["mean"] - block[keys[0]]["mean"]) for key in keys[1:]]
    positions = np.arange(len(gains), dtype=float)
    if has_oracle:
        # Visually separate pure-EEG ablation sequence from paired optical Oracle bound
        positions[-1] += 0.35
        ax.axvline(len(gains) - 1.325, color=LIGHT, ls=":", lw=0.8)

    ax.axhline(0.0, color=LIGHT, ls="--", lw=0.75)
    for i, (pos, gain, bg) in enumerate(zip(positions, gains, block_gains)):
        is_oracle = has_oracle and (i == len(gains) - 1)
        cv_col = PURPLE if is_oracle else TEAL
        cv_mk = "D" if is_oracle else "o"
        ax.errorbar(
            pos - 0.14,
            np.mean(gain),
            yerr=sem(gain),
            fmt=cv_mk,
            color=cv_col,
            ms=3.6,
            mfc=cv_col,
            lw=0.95,
            capsize=2.0,
            zorder=3,
        )
        ax.scatter(
            pos + 0.14,
            bg,
            marker="s",
            s=14,
            facecolor="white",
            edgecolor=ORANGE,
            lw=0.85,
            zorder=3,
        )

    ax.legend(
        handles=[
            Line2D([], [], color=TEAL, marker="o", ms=3.8, ls="", label=f"{folds}-fold CV (±SEM)"),
            Line2D([], [], color=ORANGE, marker="s", mfc="white", ms=3.8, ls="", label="Block split"),
        ],
        loc="upper left",
        bbox_to_anchor=(0.0, 1.15),
        ncol=2,
        columnspacing=0.8,
        fontsize=7.3,
    )
    ax.set(
        xlim=(-0.55, positions[-1] + 0.55),
        xticks=positions,
        xticklabels=STEPS[:len(gains)],
        ylim=(-5.5, 11.0),
        yticks=[-5, 0, 5, 10],
        ylabel="Gain vs. raw (pp)",
    )


@mpl.rc_context(STYLE)
def plot_fig3_recon_and_decoding(recon_res, rep_wave, decoding_res, out_path):
    """Fig. 3: Matched-budget 8-trial reconstruction & HEFMI-ICH decoding (n=37)."""
    fig = plt.figure(figsize=(WIDTH_IN, 3.06))
    gs = GridSpec(
        2, 2,
        figure=fig,
        height_ratios=[1.05, 1.0],
        width_ratios=[1.16, 1.0],
        left=0.125, right=0.988, bottom=0.11, top=0.88, hspace=0.90, wspace=0.30,
    )

    # (a) Subgridspec for All-90 Signed r (significant) + Top-25% RMSE (highly significant)
    gs_a = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0, 0], width_ratios=[1.0, 1.18], wspace=0.16)
    ax_r = fig.add_subplot(gs_a[0, 0])
    ax_rmse = fig.add_subplot(gs_a[0, 1])
    b = fig.add_subplot(gs[0, 1])
    c = fig.add_subplot(gs[1, 0])
    d = fig.add_subplot(gs[1, 1])

    rb = recon_res["regime_b_matched_8trial"]
    methods = [
        ("Ridge-FIR", "Ridge-FIR (8-Trial Cal)", "o"),
        ("TCN", "BlackBox-TCN (8-Trial Cal)", "^"),
        ("Canonical", "Canonical-HRF (K=1, 8-Trial Cal)", "s"),
        ("Local K=5", "Local Multi-Basis Ridge (K=5, 8-Trial Cal)", "v"),
        ("EB-NVC", "EB-NVC (K=5, LOSO Prior MAP, 8-Trial Cal)", "D"),
    ]
    ax_r.axvline(0.0, color=LIGHT, lw=0.7, ls="--")
    for i, (_, key, mk) in enumerate(methods):
        y = 4 - i
        is_ours = i == len(methods) - 1
        col = TEAL if is_ours else GRAY
        val_r = rb[key]["all90_contrast_signed_r"]
        ax_r.errorbar(
            val_r["mean"], y,
            xerr=val_r["std"] / np.sqrt(len(val_r["per_subject"])),
            fmt=mk, ms=3.5, color=col, mfc=col if is_ours else "white",
            mew=0.8, capsize=1.7, lw=0.9,
        )
        val_e = rb[key]["top25_rmse_uM"]
        ax_rmse.errorbar(
            val_e["mean"], y,
            xerr=val_e["std"] / np.sqrt(len(val_e["per_subject"])),
            fmt=mk, ms=3.5, color=col, mfc=col if is_ours else "white",
            mew=0.8, capsize=1.7, lw=0.9,
        )

    for ax in (ax_r, ax_rmse):
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=3)
        ax.set_ylim(-0.55, 4.55)
        ax.set_yticks(range(5))
    ax_r.set(
        xlim=(-0.032, 0.098),
        xticks=[-0.02, 0.02, 0.06],
        yticklabels=[name for name, _, _ in methods[::-1]],
    )
    ax_r.set_title("All-90 signed $r$", fontsize=8.0, pad=3.0)
    # Significance bracket between Canonical (y=2) and EB-NVC (y=0)
    bx_r = 0.068
    ax_r.plot([bx_r, bx_r + 0.004, bx_r + 0.004, bx_r], [2, 2, 0, 0], color=BLACK, lw=0.6)
    ax_r.text(bx_r + 0.006, 1.0, "*\n0.013", va="center", ha="left", fontsize=7.0, color=BLACK)

    ax_rmse.set(xlim=(0.50, 1.24), xticks=[0.6, 0.8, 1.0], yticklabels=[])
    ax_rmse.set_title(r"Top-25% RMSE ($\mu$M)", fontsize=8.0, pad=3.0)
    # Significance brackets without crossing Local K=5 errorbar
    bx_e1 = 0.995
    ax_rmse.plot([bx_e1, bx_e1 + 0.02, bx_e1 + 0.02, bx_e1], [1, 1, 0, 0], color=BLACK, lw=0.6)
    ax_rmse.text(bx_e1 + 0.026, 0.5, "****", va="center", ha="left", fontsize=7.0, color=BLACK)
    bx_e2 = 1.115
    ax_rmse.plot([bx_e2, bx_e2 + 0.02, bx_e2 + 0.02, bx_e2], [2, 2, 0, 0], color=BLACK, lw=0.6)
    ax_rmse.text(bx_e2 + 0.026, 1.0, "*", va="center", ha="left", fontsize=7.5, color=BLACK)
    add_ieee_label(ax_r, "a", "8-trial reconstruction (held-out trials 8–29)", x_offset_pt=-60, y_offset_pt=18)

    # (b) Held-out HbO contrast waveform
    b.axhline(0.0, color=LIGHT, lw=0.65, ls=":")
    b.axvline(2.0, color=FAINT, lw=0.75, ls="--", zorder=0)
    b.axvline(9.0, color=FAINT, lw=0.75, ls="--", zorder=0)
    if rep_wave is not None:
        t = np.arange(len(rep_wave["y_true"])) / 10.0
        for key, name, color, ls in [
            ("y_true", "Measured", BLUE, "-"),
            ("y_canon", "Canonical (r=+0.59)", GRAY, ":"),
            ("y_multi", "EB-NVC (r=+0.96)", RED, "--"),
        ]:
            b.plot(t, rep_wave[key], color=color, ls=ls, lw=1.25, label=name)
        b.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.02), ncol=3, columnspacing=0.55, handlelength=1.1, fontsize=7.0)
    b.set(
        xlim=(0.0, 9.9),
        ylim=(-0.22, 1.85),
        xticks=[0, 2, 4, 6, 8],
        yticks=[0.0, 0.5, 1.0, 1.5],
        xlabel="Trial time (s)",
        ylabel=r"$\Delta$HbO ($\mu$M)",
    )
    add_ieee_label(b, "b", "Held-out HbO contrast", x_offset_pt=-32, y_offset_pt=18)

    # (c) HEFMI-ICH component gain (P6-safe)
    plot_ablation_dots(c, decoding_res["cv_5fold"], decoding_res["block_split_15v15"], HEFMI_KEYS, 5, has_oracle=True)
    c.legend(
        handles=[
            Line2D([], [], color=TEAL, marker="o", ms=3.8, ls="", label="5-fold CV (±SEM)"),
            Line2D([], [], color=ORANGE, marker="s", mfc="white", ms=3.8, ls="", label="Block split"),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        ncol=2,
        columnspacing=0.8,
        fontsize=7.3,
    )
    c.set_xlabel("Decoding step")
    add_ieee_label(c, "c", "HEFMI-ICH component gain", x_offset_pt=-36, y_offset_pt=20)

    # (d) Few-shot calibration curve (k = 2, 4, 6, 8)
    fs = decoding_res["few_shot"]
    ks = np.array([2, 4, 6, 8])
    d.axhline(50.0, color=LIGHT, lw=0.7, ls="--")
    for idx, (key, name, color, ls, marker) in enumerate([
        (HEFMI_KEYS[5], "FastSlow-NVC", TEAL, "-", "o"),
        (HEFMI_KEYS[6], "Oracle", PURPLE, "-.", "D"),
        (HEFMI_KEYS[0], "Raw EEG", GRAY, "--", "^"),
    ]):
        means = [100.0 * fs[f"k={k}"][key]["mean"] for k in ks]
        spread = [100.0 * fs[f"k={k}"][key]["std"] / np.sqrt(37.0) for k in ks]
        d.errorbar(
            ks + (idx - 1) * 0.12,
            means,
            yerr=spread,
            color=color,
            ls=ls,
            marker=marker,
            ms=3.5,
            mfc=color if idx == 0 else "white",
            lw=1.0,
            capsize=1.8,
            label=name,
        )
    d.legend(loc="lower left", bbox_to_anchor=(0.0, 1.02), ncol=3, columnspacing=0.65, handlelength=1.3, fontsize=7.2)
    d.text(7.92, 61.3, r"$*\,p_{\mathrm{Bonf}}=0.032$", ha="right", va="bottom", fontsize=7.2, color=TEAL)
    d.set(
        xlim=(1.5, 8.5),
        ylim=(46.0, 64.5),
        xticks=ks,
        yticks=[50, 55, 60],
        xlabel="Calibration trials/class (k)",
        ylabel="Accuracy (%)",
    )
    add_ieee_label(d, "d", "Calibration size", x_offset_pt=-32, y_offset_pt=20)

    save(fig, out_path)


@mpl.rc_context(STYLE)
def plot_fig4_cross_stroke(liu_sum, liu_df, strata, out_path):
    """Fig. 4: External pure-EEG validation (Liu2024, n=47) & N=84 stratified summary."""
    fig = plt.figure(figsize=(WIDTH_IN, 2.34))
    gs = GridSpec(
        1, 3,
        figure=fig,
        width_ratios=[1.02, 1.92, 1.00],
        left=0.072, right=0.988, bottom=0.22, top=0.78, wspace=0.32,
    )
    a = fig.add_subplot(gs[0, 0])

    # Panel b split into label space + two forest columns
    gs_b = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs[0, 1], width_ratios=[1.18, 1.0, 1.0], wspace=0.10)
    ax_lbl = fig.add_subplot(gs_b[0, 0])
    b1 = fig.add_subplot(gs_b[0, 1])
    b2 = fig.add_subplot(gs_b[0, 2])
    c = fig.add_subplot(gs[0, 2])

    # (a) Liu2024 component gain (P6-safe, matching Fig. 3c)
    plot_ablation_dots(a, liu_sum["cv_10fold"], liu_sum["block_split_20v20"], LIU_KEYS, 10, has_oracle=False)
    a.set_ylim(-5.5, 9.5)
    a.set_yticks([-5, 0, 5])
    a.set_xlabel("Decoding step")
    sig_bracket(a, 1.0, 4.0, 5.2, 0.55, r"$+3.14\ \mathrm{pp}\ (p_t=0.040)$", fs=7.0)
    add_ieee_label(a, "a", "Liu2024 component gain", x_offset_pt=-32, y_offset_pt=16)

    # (b) Forest plot across N=84
    rows = [
        ("HEFMI-HC", 17, "HEFMI_HC_n17"),
        ("HEFMI-ICH", 20, "HEFMI_ICH_n20"),
        ("Liu2024 IS", 47, "Liu2024_IS_n47"),
        ("Stroke only", 67, "Pooled_Stroke_N67"),
        ("All cohorts", 84, "Pooled_All_N84"),
    ]
    ax_lbl.axis("off")
    ax_lbl.set_xlim(0, 1)
    ax_lbl.set_ylim(-0.6, 4.6)
    ax_lbl.text(0.95, 4.72, "$n$", ha="right", va="bottom", fontsize=7.6, color=BLACK)
    for i, (name, n, _) in enumerate(rows):
        y = 4 - i
        w = "bold" if i >= 3 else "normal"
        ax_lbl.text(0.0, y, name, ha="left", va="center", fontsize=7.5, fontweight=w, color=BLACK)
        ax_lbl.text(0.95, y, str(n), ha="right", va="center", fontsize=7.5, fontweight=w, color=BLACK)

    for ax, baseline, heading in [
        (b1, "vs_canon_hrf", "vs. canonical HRF"),
        (b2, "vs_eeg_raw", "vs. raw EEG"),
    ]:
        for y in range(5):
            ax.axhline(y, color=FAINT, lw=0.65, zorder=-2)
        ax.axvline(0.0, color=GRAY, ls="--", lw=0.7, zorder=0)
        ax.axhspan(-0.44, 0.44, color="#E7F3F2", zorder=-1)
        ax.axhline(1.5, color=LIGHT, lw=0.6, zorder=0)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        ax.set_title(heading, pad=3, fontsize=7.7, loc="center")
        ax.set(xlim=(-5.8, 13.8), ylim=(-0.6, 4.6), xticks=[-5, 0, 5, 10], xlabel="Gain (pp)")
        for i, (_, _, cohort) in enumerate(rows):
            row = strata[cohort][baseline]
            m, lo, hi = [100.0 * row["diff_" + k] for k in ("mean", "ci_lo", "ci_hi")]
            y = 4 - i
            if i < 3:
                ax.hlines(y, lo, hi, color=BLACK, lw=0.85, zorder=2)
                ax.scatter(m, y, color=BLACK, marker="s", s=14, linewidth=0, zorder=3)
            else:
                ax.add_patch(
                    Polygon(
                        [(lo, y), (m, y + 0.18), (hi, y), (m, y - 0.18)],
                        facecolor=TEAL,
                        edgecolor=TEAL,
                        lw=0.6,
                        zorder=3,
                    )
                )
        if baseline == "vs_canon_hrf":
            row_all = strata["Pooled_All_N84"][baseline]
            hi_a = 100.0 * row_all["diff_ci_hi"]
            ax.text(hi_a + 0.4, 0.0, r"$+2.51^*$", ha="left", va="center", fontsize=7.0, color=TEAL, fontweight="bold")
    add_ieee_label(ax_lbl, "b", "FastSlow-NVC accuracy gain (95% CI)", x_offset_pt=-4, y_offset_pt=16)

    # (c) IS severity (n=47): adaptive beeswarm + median bar (handles n=1,2,4,6,13,21 cleanly)
    ordinal_beeswarm(c, liu_df["mrs"].to_numpy(), liu_df["t_pk_ipsi"].to_numpy(), RED, y_span=6.5, size=12.0)
    counts = [int((liu_df["mrs"] == g).sum()) for g in range(6)]
    c.set(
        xlim=(-0.55, 5.55),
        ylim=(1.5, 8.3),
        xticks=np.arange(6),
        xticklabels=[f"{g}\n({n})" for g, n in enumerate(counts)],
        yticks=[2, 4, 6, 8],
        xlabel="mRS grade (n)",
        ylabel="EEG peak (s)",
    )
    c.text(
        0.04, 0.05,
        r"$\rho_{\mathrm{part}}=-0.46^*$" + "\n" + r"$q=0.0165$",
        transform=c.transAxes,
        fontsize=7.1,
        color=BLACK,
        va="bottom",
        ha="left",
    )
    add_ieee_label(c, "c", "IS severity", x_offset_pt=-26, y_offset_pt=16)

    save(fig, out_path)
