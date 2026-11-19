"""
Active Learning Sampler
Combines uncertainty sampling, high-real-probability (exploitation), anomaly (IsolationForest),
uniform random, and latent space diversity sampling to select informative cutouts for visual inspection.

境界事例 (margin 最小) だけを選ぶとラベル集合が難しい事例に偏り、
ラベル分布・精度推定が歪むため、fink と同様に複数の選択枠を混ぜる。
"""

import numpy as np
from typing import List, Set, Tuple, Optional, Sequence
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def valid_embedding_mask(embeddings: np.ndarray) -> np.ndarray:
    """空データや異常値（ノルムが極小または非有限の埋め込み）を除外するマスク"""
    emb_norms = np.linalg.norm(embeddings, axis=1)
    return (emb_norms > 1e-4) & np.isfinite(emb_norms)

def _allocate(n_samples: int, ratios: Sequence[Tuple[str, float]]) -> List[Tuple[str, int]]:
    """比率から各枠の件数を決める (端数は比率の大きい枠から順に配分)"""
    total = sum(r for _, r in ratios)
    raw = [(name, n_samples * r / total) for name, r in ratios]
    counts = {name: int(np.floor(v)) for name, v in raw}
    rest = n_samples - sum(counts.values())
    for name, v in sorted(raw, key=lambda t: t[1] - np.floor(t[1]), reverse=True)[:rest]:
        counts[name] += 1
    return [(name, counts[name]) for name, _ in ratios]

def _diverse_top(embeddings: np.ndarray, candidates: np.ndarray, scores: np.ndarray, k: int, pool_factor: int = 5) -> List[int]:
    """スコア上位 k*pool_factor 件のプールを KMeans で k クラスタに分け、各クラスタの最高スコアを選ぶ"""
    if k <= 0 or len(candidates) == 0:
        return []
    order = np.argsort(-scores)
    pool = candidates[order[:min(len(candidates), k * pool_factor)]]
    pool_scores = scores[order[:len(pool)]]
    if len(pool) <= k:
        return pool.tolist()
    labels = KMeans(n_clusters=k, random_state=42, n_init=4).fit_predict(embeddings[pool])
    picks = []
    for c in range(k):
        members = np.where(labels == c)[0]
        if len(members) > 0:
            picks.append(int(pool[members[np.argmax(pool_scores[members])]]))
    return picks

def anomaly_scores(embeddings: np.ndarray, candidates: np.ndarray, fit_size: int = 50000, seed: int = 42) -> np.ndarray:
    """IsolationForest による異常度 (大きいほど異常)。学習はサブサンプルで高速化"""
    rng = np.random.default_rng(seed)
    fit_idx = candidates if len(candidates) <= fit_size else rng.choice(candidates, size=fit_size, replace=False)
    scaler = StandardScaler().fit(embeddings[fit_idx])
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=seed, n_jobs=-1)
    iso.fit(scaler.transform(embeddings[fit_idx]))
    return -iso.decision_function(scaler.transform(embeddings[candidates]))

def sample_candidates_for_labeling(
    embeddings: np.ndarray,
    margins: Optional[np.ndarray],
    entropies: Optional[np.ndarray],
    already_labeled_indices: Set[int],
    n_samples: int = 20,
    uncertainty_method: str = "margin",
    probs: Optional[np.ndarray] = None,
    classes: Optional[List[str]] = None,
    uncertainty_ratio: float = 0.5,
    exploit_ratio: float = 0.15,
    anomaly_ratio: float = 0.15,
    random_ratio: float = 0.1,
    diversity_ratio: float = 0.1,
    seed: Optional[int] = None,
    focus_classes: Optional[List[str]] = None
) -> List[Tuple[int, str]]:
    """
    目視ラベリング対象候補を能動的にサンプリング
    Args:
        embeddings: (N, D) 潜在ベクトル (+ 形態特徴)
        margins / entropies: (N,) RF の不確実性。probs=None (未学習) の場合は使わない
        already_labeled_indices: 判定済み (学習用・評価用とも) のインデックス集合
        probs: (N, C) RF のクラス確率。None なら初期段階として不確実性・exploit 枠を他の枠に回す
        focus_classes: 2 クラス (例 ['galaxy', 'star']) を指定すると、その 2 クラスの境界に集中する。
            確率の上位 2 クラスがこの 2 つであるソースから、2 つの確率の差が小さい順に選ぶ (枠名 'boundary')。
            配分は boundary 80% / random 10% / diversity 10%。
        classes: クラス名 ('bogus' を含めば exploit = 非 bogus 確率)
        *_ratio: 各選択枠の比率
    Returns:
        [(index, selection_reason), ...]  reason は 'uncertainty' / 'exploit' / 'anomaly' / 'random' / 'diversity'
    """
    rng = np.random.default_rng(seed)
    total = len(embeddings)
    unlabeled_mask = valid_embedding_mask(embeddings)
    if already_labeled_indices:
        unlabeled_mask[list(already_labeled_indices)] = False
    unlabeled_indices = np.where(unlabeled_mask)[0]
    if len(unlabeled_indices) <= n_samples:
        return [(int(i), "all_remaining") for i in unlabeled_indices]

    if probs is None:
        # 初期段階: モデルが無いので不確実性・exploit は意味を持たない
        ratios = [("anomaly", 1.0), ("random", 1.0), ("diversity", 1.0)]
    elif focus_classes:
        if classes is None or len(focus_classes) != 2 or any(c not in classes for c in focus_classes):
            raise ValueError(f"focus_classes must be 2 of {classes}: {focus_classes}")
        ratios = [("boundary", 0.8), ("random", 0.1), ("diversity", 0.1)]
    else:
        ratios = [("exploit", exploit_ratio), ("uncertainty", uncertainty_ratio), ("anomaly", anomaly_ratio),
                  ("random", random_ratio), ("diversity", diversity_ratio)]
    plan = [(name, k) for name, k in _allocate(n_samples, ratios) if k > 0]

    selected: List[Tuple[int, str]] = []
    taken = np.zeros(total, dtype=bool)

    def remaining() -> np.ndarray:
        return unlabeled_indices[~taken[unlabeled_indices]]

    for name, k in plan:
        cand = remaining()
        if len(cand) == 0:
            break
        if name == "exploit":
            if classes is not None and "bogus" in classes:
                score = 1.0 - probs[cand, classes.index("bogus")]
            else:
                score = probs[cand].max(axis=1)
            picks = _diverse_top(embeddings, cand, score, k)
        elif name == "boundary":
            ia, ib = classes.index(focus_classes[0]), classes.index(focus_classes[1])
            top2 = np.argsort(-probs[cand], axis=1)[:, :2]
            on_pair = ((top2[:, 0] == ia) & (top2[:, 1] == ib)) | ((top2[:, 0] == ib) & (top2[:, 1] == ia))
            pc = cand[on_pair]
            if len(pc) == 0:
                print(f"[!] No unlabeled sources with top-2 classes {focus_classes}.")
                picks = []
            else:
                score = -np.abs(probs[pc, ia] - probs[pc, ib])
                picks = _diverse_top(embeddings, pc, score, k)
            name = f"boundary:{focus_classes[0]}-{focus_classes[1]}"
        elif name == "uncertainty":
            score = -margins[cand] if uncertainty_method == "margin" else entropies[cand]
            picks = _diverse_top(embeddings, cand, score, k)
        elif name == "anomaly":
            picks = _diverse_top(embeddings, cand, anomaly_scores(embeddings, cand), k)
        elif name == "random":
            picks = rng.choice(cand, size=min(k, len(cand)), replace=False).tolist()
        else:  # diversity: 未知の潜在構造・レア天体の探索
            kmeans = MiniBatchKMeans(n_clusters=k, random_state=int(rng.integers(1 << 31)), batch_size=4096, n_init="auto")
            kmeans.fit(embeddings[cand])
            picks = []
            for center in kmeans.cluster_centers_:
                best = int(cand[int(np.argmin(np.linalg.norm(embeddings[cand] - center, axis=1)))])
                if not taken[best] and best not in picks:
                    picks.append(best)
        for p in picks:
            if not taken[p]:
                taken[p] = True
                selected.append((int(p), name))

    # 枠が埋まらなかった分はランダムで補充
    if len(selected) < n_samples:
        cand = remaining()
        for p in rng.choice(cand, size=min(n_samples - len(selected), len(cand)), replace=False):
            selected.append((int(p), "random"))

    return selected[:n_samples]
