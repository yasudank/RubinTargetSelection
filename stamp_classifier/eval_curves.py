"""
Threshold Curves on the Evaluation Set
RF のスコアの閾値ごとに、母集団に対する純度 (purity = precision) と完全性 (completeness = recall)、
選ばれるソース数の推定値を、評価用セットの層の重みで求める。誤差は層内ブートストラップ。

対象 (target):
  real          : positive_real + negative_real (スコア = 1 - P(bogus))
  positive_real : スコア = P(positive_real)
  negative_real : スコア = P(negative_real)
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

TARGETS = {
    "real": ["positive_real", "negative_real"],
    "positive_real": ["positive_real"],
    "negative_real": ["negative_real"],
}

def targets_for(classes: List[str]) -> Dict[str, List[str]]:
    """
    クラス構成に応じた閾値分析の対象。step1 (positive_real / negative_real / bogus) では TARGETS を使い、
    それ以外 (step2: star / galaxy / no_host / bogus など) では bogus 以外の各クラスを単独の対象にする。
    """
    t = {k: v for k, v in TARGETS.items() if all(c in classes for c in v + ["bogus"])}
    return t or {c: [c] for c in classes if c != "bogus"}

def target_score(probs: np.ndarray, classes: List[str], target: str) -> np.ndarray:
    if target == "real":
        return 1.0 - probs[:, classes.index("bogus")]
    return probs[:, classes.index(target)]

def _curve(score, is_pos, w, thresholds):
    sel = score[None, :] >= thresholds[:, None]                    # (T, n)
    w_sel = (sel * w).sum(1)
    w_tp = (sel * (w * is_pos)).sum(1)
    w_pos = (w * is_pos).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        purity = np.where(w_sel > 0, w_tp / w_sel, np.nan)
        completeness = w_tp / w_pos if w_pos > 0 else np.full(len(thresholds), np.nan)
    return purity, completeness, w_sel

def threshold_curves(probs: np.ndarray, y_true: np.ndarray, strata_ids: np.ndarray, sizes: np.ndarray,
                     classes: List[str], target: str = "real", thresholds: Optional[np.ndarray] = None,
                     n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 1.0001, 0.05), 2)
    keep = strata_ids >= 0
    probs, y_true, h = probs[keep], np.asarray(y_true)[keep], np.asarray(strata_ids)[keep]
    n_h = np.bincount(h, minlength=len(sizes)).astype(float)
    w_h = np.where(n_h > 0, sizes / np.maximum(n_h, 1), 0.0)
    w = w_h[h]

    score = target_score(probs, classes, target)
    is_pos = np.isin(y_true, [classes.index(c) for c in targets_for(classes)[target]]).astype(float)
    purity, completeness, n_sel = _curve(score, is_pos, w, thresholds)

    rng = np.random.default_rng(seed)
    groups = [np.where(h == k)[0] for k in range(len(sizes)) if (h == k).any()]
    bp, bc, bn = [], [], []
    for _ in range(n_boot):
        i = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        p, c, n = _curve(score[i], is_pos[i], w[i], thresholds)
        bp.append(p); bc.append(c); bn.append(n)
    bp, bc, bn = np.array(bp), np.array(bc), np.array(bn)

    return pd.DataFrame({
        "target": target,
        "threshold": thresholds,
        "purity": purity,
        "purity_lo": np.nanpercentile(bp, 2.5, axis=0), "purity_hi": np.nanpercentile(bp, 97.5, axis=0),
        "completeness": completeness,
        "completeness_lo": np.nanpercentile(bc, 2.5, axis=0), "completeness_hi": np.nanpercentile(bc, 97.5, axis=0),
        "n_selected_est": n_sel,
        "n_selected_lo": np.nanpercentile(bn, 2.5, axis=0), "n_selected_hi": np.nanpercentile(bn, 97.5, axis=0),
        "n_eval_selected": [(score >= t).sum() for t in thresholds],
    })

def plot_threshold_curves(df: pd.DataFrame, out_png: str, title: str = "") -> None:
    import matplotlib
    import matplotlib.pyplot as plt
    targets = list(dict.fromkeys(df["target"]))
    fig, axes = plt.subplots(1, len(targets), figsize=(5.2 * len(targets), 4.3), squeeze=False)
    for ax, t in zip(axes[0], targets):
        d = df[df["target"] == t]
        ax.plot(d["threshold"], d["purity"], "-o", ms=3, color="#1f77b4", label="Purity")
        ax.fill_between(d["threshold"], d["purity_lo"], d["purity_hi"], color="#1f77b4", alpha=0.2)
        ax.plot(d["threshold"], d["completeness"], "-s", ms=3, color="#d62728", label="Completeness")
        ax.fill_between(d["threshold"], d["completeness_lo"], d["completeness_hi"], color="#d62728", alpha=0.2)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Score threshold")
        ax.set_title(f"target: {t}", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax2 = ax.twinx()
        ax2.plot(d["threshold"], d["n_selected_est"], ":", color="gray", label="N selected (est.)")
        ax2.set_ylabel("Estimated N selected (population)", color="gray")
        ax2.tick_params(axis="y", labelcolor="gray")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
