"""
Stratified Evaluation Set Support
固定評価用セットを「予測クラスによる層別抽出」で補強し、層の重みで母集団に対する偏りのない精度を推定する。

- 層 (stratum) は、作成時点の RF の予測クラスで母集団 (評価対象になり得る全ソース) を分割したもの。
  割り当ては diaSourceId ごとにファイルへ保存して固定するため、後でエンコーダや RF を更新しても層は変わらない。
- 一様ランダムに選んだ評価サンプルも、層の中では一様ランダムなので、層別に追加したサンプルと合わせて
  各層内の単純無作為標本として扱える。
- サンプル i (層 h) の重みは w_i = N_h / n_h (N_h: 層の大きさ, n_h: 層内の評価ラベル数)。
  重み付き混同行列から accuracy / recall / precision / balanced accuracy を推定し、
  層内ブートストラップで標準誤差を求める。
"""

import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

def strata_file_path(eval_labels_file: str) -> str:
    base, _ = os.path.splitext(eval_labels_file)
    return f"{base}_strata.npz"

def create_strata(strata_file: str, source_ids: np.ndarray, pred_classes: np.ndarray, eligible: np.ndarray,
                  class_names: List[str], stratifier_model_path: Optional[str] = None) -> Dict:
    """評価対象になり得るソース (eligible) を予測クラスで層に分けて保存"""
    sids = source_ids[eligible].astype(np.int64)
    strata = pred_classes[eligible].astype(np.int8)
    os.makedirs(os.path.dirname(strata_file) or ".", exist_ok=True)
    np.savez_compressed(
        strata_file,
        source_ids=sids,
        strata=strata,
        class_names=np.array(class_names),
        created=np.array(time.strftime("%Y-%m-%d %H:%M:%S"))
    )
    # 層を作ったモデルを記録用に保存
    if stratifier_model_path and os.path.exists(stratifier_model_path):
        shutil.copy2(stratifier_model_path, os.path.splitext(strata_file)[0] + "_stratifier.joblib")
    print(f"[+] Created evaluation strata ({len(sids)} eligible sources) -> {strata_file}")
    return load_strata(strata_file)

def load_strata(strata_file: str) -> Optional[Dict]:
    if not os.path.exists(strata_file):
        return None
    d = np.load(strata_file, allow_pickle=False)
    sids = d["source_ids"]
    strata = d["strata"]
    names = [str(c) for c in d["class_names"]]
    sizes = np.bincount(strata, minlength=len(names)).astype(np.int64)
    return {
        "source_ids": sids,
        "strata": strata,
        "class_names": names,
        "sizes": sizes,
        "created": str(d["created"]),
        "index": pd.Series(np.arange(len(sids)), index=sids),
    }

def stratum_of(strata: Dict, source_ids) -> np.ndarray:
    """diaSourceId -> 層 ID (層の母集団に含まれない場合は -1)"""
    pos = strata["index"].reindex(np.asarray(source_ids, dtype=np.int64)).values
    out = np.full(len(pos), -1, dtype=int)
    ok = ~pd.isna(pos)
    out[ok] = strata["strata"][pos[ok].astype(np.int64)]
    return out

def print_strata_summary(strata: Dict, eval_source_ids) -> None:
    h = stratum_of(strata, eval_source_ids)
    counts = np.bincount(h[h >= 0], minlength=len(strata["class_names"]))
    print(f"Evaluation strata (created {strata['created']}, stratified by predicted class):")
    for k, name in enumerate(strata["class_names"]):
        print(f"  pred:{name:15s} N_h={strata['sizes'][k]:8d}  labeled n_h={counts[k]:4d}")
    if (h < 0).any():
        print(f"  [!] {int((h < 0).sum())} evaluation labels are outside the strata population and are ignored in weighted metrics.")

def _metrics_from_confusion(C: np.ndarray) -> Dict[str, np.ndarray]:
    total = C.sum()
    row = C.sum(axis=1)
    col = C.sum(axis=0)
    diag = np.diag(C)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(row > 0, diag / row, np.nan)
        precision = np.where(col > 0, diag / col, np.nan)
    return {
        "acc": diag.sum() / total if total > 0 else np.nan,
        "bacc": np.nanmean(recall) if np.any(row > 0) else np.nan,
        "recall": recall,
        "precision": precision,
        "prevalence": row / total if total > 0 else row * np.nan,
    }

def _weighted_confusion(y_true, y_pred, h, sizes, n_classes) -> np.ndarray:
    C = np.zeros((n_classes, n_classes))
    for k in range(len(sizes)):
        m = h == k
        n_h = m.sum()
        if n_h == 0:
            continue
        np.add.at(C, (y_true[m], y_pred[m]), sizes[k] / n_h)
    return C

def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray, strata_ids: np.ndarray, sizes: np.ndarray,
                     n_classes: int, n_boot: int = 1000, seed: int = 0) -> Dict:
    """
    層の重み付きで母集団に対する精度を推定し、層内ブートストラップで標準誤差を付ける。
    strata_ids < 0 のサンプル、および評価ラベルが 1 件も無い層は推定から除外する。
    """
    keep = strata_ids >= 0
    y_true, y_pred, h = np.asarray(y_true)[keep], np.asarray(y_pred)[keep], np.asarray(strata_ids)[keep]
    empty = [k for k in range(len(sizes)) if not (h == k).any()]

    C = _weighted_confusion(y_true, y_pred, h, sizes, n_classes)
    point = _metrics_from_confusion(C)

    rng = np.random.default_rng(seed)
    groups = [np.where(h == k)[0] for k in range(len(sizes))]
    boot = {"acc": [], "bacc": [], "recall": [], "precision": []}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups if len(g) > 0])
        m = _metrics_from_confusion(_weighted_confusion(y_true[idx], y_pred[idx], h[idx], sizes, n_classes))
        for key in boot:
            boot[key].append(m[key])
    se = {key: np.nanstd(np.array(v, dtype=float), axis=0) for key, v in boot.items()}

    return {
        "confusion_weighted": C,
        "point": point,
        "se": se,
        "n_used": int(keep.sum()),
        "empty_strata": empty,
        "coverage": float(sum(sizes[k] for k in range(len(sizes)) if k not in empty) / sizes.sum()),
    }
