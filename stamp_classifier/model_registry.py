"""
Frozen Model Registry
運用に使う RF とクラス別の判定閾値を、バージョン名付きで固定保存・読み込みする。

- models/step{N}/rf_{version}.joblib : RF のコピー (active_loop で学び直されても影響を受けない)
- models/step{N}/rf_{version}.json   : 閾値、作成日時、学習ラベル数、エンコーダのハッシュ、
                                       評価用セットでの閾値ごとの純度・完全性 (固定時点の値)
RF は特定のエンコーダの埋め込みで学習されているため、エンコーダのハッシュを記録し、分類時に不一致を警告する。
"""

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import numpy as np

def file_md5(path: str, chunk: int = 1 << 20) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def parse_thresholds(text: str, classes: List[str]) -> Dict[str, float]:
    """'positive_real=0.55,negative_real=0.75' -> {'positive_real': 0.55, 'negative_real': 0.75}"""
    out = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, val = item.split("=")
        name = name.strip()
        if name not in classes:
            raise ValueError(f"Unknown class '{name}' in thresholds. Choose from {classes}")
        v = float(val)
        if not 0.0 < v <= 1.0:
            raise ValueError(f"Threshold for {name} must be in (0, 1]: {v}")
        out[name] = v
    if not out:
        raise ValueError("No thresholds given.")
    return out

def model_paths(checkpoint_dir: str, version: str) -> Dict[str, str]:
    return {
        "rf": os.path.join(checkpoint_dir, f"rf_{version}.joblib"),
        "meta": os.path.join(checkpoint_dir, f"rf_{version}.json"),
    }

def save_frozen_model(checkpoint_dir: str, version: str, rf_path: str, encoder_path: str,
                      classes: List[str], thresholds: Dict[str, float], extra: Dict) -> Dict[str, str]:
    paths = model_paths(checkpoint_dir, version)
    if os.path.exists(paths["rf"]) or os.path.exists(paths["meta"]):
        raise FileExistsError(f"Model version '{version}' already exists ({paths['meta']}). Choose another --model_version.")
    shutil.copy2(rf_path, paths["rf"])
    meta = {
        "version": version,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": classes,
        "thresholds": thresholds,
        "decision_rule": ("A source is assigned to a thresholded class if its probability >= the threshold "
                          "(highest probability wins if several qualify). Otherwise it is assigned to the argmax class "
                          "if that class has no threshold, else 'uncertain'."),
        "rf_md5": file_md5(paths["rf"]),
        "encoder_path": encoder_path,
        "encoder_md5": file_md5(encoder_path),
    }
    meta.update(extra)
    with open(paths["meta"], "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, default=float)
    return paths

def load_frozen_model_meta(checkpoint_dir: str, version: str) -> Dict:
    paths = model_paths(checkpoint_dir, version)
    if not os.path.exists(paths["meta"]):
        raise FileNotFoundError(f"Frozen model '{version}' not found ({paths['meta']}). Run --mode freeze_model first.")
    with open(paths["meta"]) as f:
        meta = json.load(f)
    meta["rf_path"] = paths["rf"]
    if file_md5(paths["rf"]) != meta.get("rf_md5"):
        raise RuntimeError(f"RF file {paths['rf']} does not match the recorded hash. It may have been modified.")
    current = file_md5(meta.get("encoder_path", ""))
    if current != meta.get("encoder_md5"):
        print(f"[!] WARNING: the encoder ({meta.get('encoder_path')}) has changed since model '{version}' was frozen. "
              f"The frozen RF expects embeddings from the original encoder; predictions and thresholds may be invalid.")
    return meta

def decide(probs: np.ndarray, classes: List[str], thresholds: Dict[str, float]) -> np.ndarray:
    """閾値ルールで各ソースの判定クラスを決める"""
    n = len(probs)
    thr = np.array([thresholds.get(c, np.inf) for c in classes])
    passed = probs >= thr[None, :]                                     # (n, C)
    masked = np.where(passed, probs, -1.0)
    best_passed = masked.argmax(axis=1)
    any_passed = passed.any(axis=1)
    argmax = probs.argmax(axis=1)
    unthresholded = np.array([c not in thresholds for c in classes])
    out = np.full(n, "uncertain", dtype=object)
    out[any_passed] = np.array(classes, dtype=object)[best_passed[any_passed]]
    fallback = ~any_passed & unthresholded[argmax]
    out[fallback] = np.array(classes, dtype=object)[argmax[fallback]]
    return out
