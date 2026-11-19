"""
Group B Classifier (step 2b)
step2 で「中心に母天体が無い、または母天体の S/N が足りない」グループ B のソースを、専用のモデルで 4 クラスに分ける。

  no_host        : Template の突発天体の付近に有意な光が無い                       -> 最終判定 no_host
  real_uncertain : 天体としては本物だが、S/N が悪く母天体の種類を判断できない      -> 最終判定 uncertain
  galaxy         : 大きな銀河の裾野の部分に重なっている (母天体の中心から離れた位置) -> 最終判定 galaxy
  bogus          : 明るい恒星の裾、ゴースト、引き算の失敗などの偽検出                -> 最終判定 bogus

特徴量: step2 の SimCLR 埋め込み (Science + Template, 128) + step1 の SimCLR 埋め込み (Difference, 128)
        + step1 の差分像の形態特徴 (15) + 母天体の形態・文脈の特徴 (host_morphology, Template と Science)
対象: data/host_morph_step2.csv の host_group == 'B' のソース (除外した visit のソースは除く)
ラベル: data/human_labels_step2b.csv (学習用), data/eval_labels_step2b.csv (評価用: グループ B から一様ランダム, ブラインド)
"""

import os
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

from .config import DATA_DIR, MODELS_DIR, FITS_ROOT
from .labeling_viewer import LabelingViewer, load_human_labels, cleanup_gui

CLASSES = ["no_host", "real_uncertain", "galaxy", "bogus"]
FINAL_DECISION = {"no_host": "no_host", "real_uncertain": "uncertain", "galaxy": "galaxy", "bogus": "bogus"}
LABELS_FILE = os.path.join(DATA_DIR, "human_labels_step2b.csv")
EVAL_FILE = os.path.join(DATA_DIR, "eval_labels_step2b.csv")
MODEL_DIR = os.path.join(MODELS_DIR, "step2b")
MODEL_FILE = os.path.join(MODEL_DIR, "random_forest.joblib")
PRED_FILE = os.path.join(DATA_DIR, "predictions_step2b.csv")
HISTORY_FILE = os.path.join(DATA_DIR, "active_learning_history_step2b.csv")

# ---------------------------------------------------------------------------
# データの準備
# ---------------------------------------------------------------------------
def load_group_b() -> Dict:
    """グループ B のソースの一覧と特徴量の行列"""
    from .feature_extractor import load_cached_embeddings
    from .host_morphology import clean_for_rf
    H = pd.read_csv(os.path.join(DATA_DIR, "host_morph_step2.csv"))
    ex_file = os.path.join(DATA_DIR, "excluded_sources.csv")
    if os.path.exists(ex_file):
        H = H[~H["diaSourceId"].isin(pd.read_csv(ex_file)["diaSourceId"])]
    B = H[H["host_group"] == "B"].reset_index(drop=True)
    host_cols = [c for c in H.columns if c.startswith("tmp_") or c.startswith("sci_") or c == "ext_ratio_sci_minus_tmp"]

    c2 = load_cached_embeddings(2)
    c1 = load_cached_embeddings(1)
    i2 = pd.Series(np.arange(len(c2["diaSourceIds"])), index=c2["diaSourceIds"].astype(np.int64))
    i1 = pd.Series(np.arange(len(c1["diaSourceIds"])), index=c1["diaSourceIds"].astype(np.int64))
    sid = B["diaSourceId"].astype(np.int64)
    ok = sid.isin(i2.index) & sid.isin(i1.index)
    if (~ok).any():
        print(f"[!] {int((~ok).sum())} group-B sources are missing from the embedding caches and are skipped.")
    B = B[ok.values].reset_index(drop=True)
    sid = B["diaSourceId"].astype(np.int64)
    r2, r1 = i2.loc[sid].values, i1.loc[sid].values
    X = np.hstack([
        c2["simclr_embeddings"][r2],                       # step2 SimCLR (Science + Template)
        c1["simclr_embeddings"][r1],                       # step1 SimCLR (Difference)
        clean_for_rf(c1["morph_features"][r1]) if c1["morph_features"] is not None else np.zeros((len(B), 0)),
        clean_for_rf(B[host_cols].values.astype(np.float32)),
    ]).astype(np.float32)
    meta = {int(s): {"midpointMjdTai": float(m), "band": str(b), "diaObjectId": int(o)}
            for s, m, b, o in zip(c2["diaSourceIds"][r2], c2["midpointMjdTai"][r2], c2["bands"][r2], c2["diaObjectIds"][r2])}
    return {"df": B, "X": X, "meta": meta, "host_cols": host_cols}

def _context_note(r) -> str:
    """判定画面のタイトルに出す、母天体まわりの数値"""
    return (f"Group B | Tmp S/N at transient {r['tmp_center_snr']:.1f} | brightest in Tmp: S/N {r['tmp_brightest_snr']:.0f} "
            f"at {r['tmp_brightest_dist_fwhm']:.1f} FWHM")

def _viewer() -> LabelingViewer:
    v = LabelingViewer(step=2)
    v.classes = CLASSES
    return v

def _labels(path: str) -> pd.DataFrame:
    df = load_human_labels(path)
    return df[df["class_name"].isin(CLASSES)] if not df.empty else df

# ---------------------------------------------------------------------------
# モデル
# ---------------------------------------------------------------------------
def train_rf(X: np.ndarray, y: np.ndarray, seed: int = 42) -> RandomForestClassifier:
    import warnings
    rf = RandomForestClassifier(n_estimators=500, class_weight="balanced", oob_score=True, n_jobs=-1, random_state=seed)
    with warnings.catch_warnings():
        # ラベルの少ないクラスがあると OOB の推定が無いサンプルが出る、という警告 (学習には影響しない)
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
        rf.fit(X, y)
    return rf

def predict_full(rf, X: np.ndarray) -> np.ndarray:
    raw = rf.predict_proba(X)
    P = np.zeros((len(X), len(CLASSES)))
    for j, k in enumerate(rf.classes_):
        P[:, int(k)] = raw[:, j]
    return P

def _xy(data: Dict, labels: pd.DataFrame):
    idx = pd.Series(np.arange(len(data["df"])), index=data["df"]["diaSourceId"].astype(np.int64))
    lab = labels[labels["diaSourceId"].astype(np.int64).isin(idx.index)]
    return data["X"][idx.loc[lab["diaSourceId"].astype(np.int64)].values], lab["label"].astype(int).values, lab

def evaluate(rf, data: Dict, verbose: bool = True) -> Dict:
    ev = _labels(EVAL_FILE)
    X, y, _ = _xy(data, ev)
    if len(y) == 0:
        print(f"[!] No group-B evaluation labels yet ({EVAL_FILE}). Run --mode b_eval_set.")
        return {}
    pred = predict_full(rf, X).argmax(1)
    m = {"n_eval": int(len(y)), "eval_acc": float(accuracy_score(y, pred)),
         "eval_bacc": float(balanced_accuracy_score(y, pred))}
    if verbose:
        present = sorted(set(y) | set(pred))
        print(f"\n[Group B eval set] N={m['n_eval']} (random sample of group B) "
              f"Accuracy={m['eval_acc']:.3f}  Balanced Accuracy={m['eval_bacc']:.3f}")
        print(pd.DataFrame(confusion_matrix(y, pred, labels=present),
                           index=[f"true:{CLASSES[i]}" for i in present], columns=[f"pred:{CLASSES[i]}" for i in present]))
        print(classification_report(y, pred, labels=present, target_names=[CLASSES[i] for i in present], zero_division=0))
    return m

def _append_history(entry: Dict):
    df = pd.DataFrame([entry])
    if os.path.exists(HISTORY_FILE):
        df = pd.concat([pd.read_csv(HISTORY_FILE), df], ignore_index=True)
    df.to_csv(HISTORY_FILE, index=False)

def retrain(data: Optional[Dict] = None) -> Optional[RandomForestClassifier]:
    data = data or load_group_b()
    tr = _labels(LABELS_FILE)
    ev = _labels(EVAL_FILE)
    if not ev.empty and not tr.empty:
        tr = tr[~tr["diaSourceId"].isin(ev["diaSourceId"])]
    X, y, _ = _xy(data, tr)
    if len(y) < 6 or len(set(y)) < 2:
        print(f"[!] Not enough group-B training labels ({len(y)}; need >= 6 in >= 2 classes).")
        return None
    rf = train_rf(X, y)
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(rf, MODEL_FILE)
    print(f"[+] Group-B RF trained on {len(y)} labels {pd.Series([CLASSES[i] for i in y]).value_counts().to_dict()} "
          f"(OOB {rf.oob_score_:.3f}) -> {MODEL_FILE}")
    m = evaluate(rf, data)
    _append_history({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "total_labeled": len(y),
                     "oob_accuracy": rf.oob_score_, **m})
    return rf

# ---------------------------------------------------------------------------
# ラベル付け
# ---------------------------------------------------------------------------
def _review(data: Dict, rows: pd.DataFrame, labels_file: str, selection: List[str], probs=None,
            interactive: bool = True, header: str = "") -> int:
    v = _viewer()
    v.labels_file = labels_file
    n_new = 0
    for k, (i, why) in enumerate(zip(rows.index, selection)):
        r = data["df"].loc[i]
        sid = int(r["diaSourceId"])
        m = data["meta"][sid]
        rec = {"diaSourceId": sid, "diaObjectId": m["diaObjectId"], "midpointMjdTai": m["midpointMjdTai"], "band": m["band"]}
        print(f"\n[{header} {k + 1}/{len(rows)}]")
        if probs is None:
            ans = v.review_sample(rec, pred_class=None, confidence=None, margin=None, interactive=interactive,
                                  selection=why, title_note=_context_note(r))
        else:
            p = probs[i]
            srt = np.sort(p)[::-1]
            ans = v.review_sample(rec, pred_class=CLASSES[int(p.argmax())], confidence=float(srt[0]),
                                  margin=float(srt[0] - srt[1]), interactive=interactive, selection=why,
                                  title_note=_context_note(r))
        if ans == "QUIT":
            print("[*] Paused. Re-run the same command to continue.")
            break
        if ans not in (None, "SKIP_EMPTY"):
            n_new += 1
    cleanup_gui()
    return n_new

def build_eval_set(n_target: int = 150, interactive: bool = True, seed: int = 20261021) -> None:
    """グループ B から天体が重ならないよう一様ランダムに選び、ブラインドで判定 (中断・再開可)"""
    data = load_group_b()
    df = data["df"]
    ev = _labels(EVAL_FILE)
    tr = _labels(LABELS_FILE)
    done = set(ev["diaSourceId"].astype(np.int64)) if not ev.empty else set()
    used_obj = set(tr["diaObjectId"].astype(np.int64)) if not tr.empty else set()
    used_obj |= set(ev["diaObjectId"].astype(np.int64)) if not ev.empty else set()
    print(f"=== Group-B evaluation set: {len(done)} / {n_target} labeled -> {EVAL_FILE} ===")
    if len(done) >= n_target:
        print("[+] Target reached.")
        return
    order = df.sample(frac=1.0, random_state=seed)
    order = order[~order["diaObjectId"].isin(used_obj)].drop_duplicates("diaObjectId")
    order = order.head(n_target - len(done))
    _review(data, order, EVAL_FILE, ["eval_random_b"] * len(order), interactive=interactive, header="Eval B")
    ev = _labels(EVAL_FILE)
    print(f"\n[+] Group-B evaluation set: {len(ev)} labels: {ev['class_name'].value_counts().to_dict() if len(ev) else {}}")

def active_loop(n_samples: int = 20, interactive: bool = True) -> None:
    """不確実性・多様性などの枠で選んだグループ B のソースを判定し、RF を学び直す"""
    from .active_sampler import sample_candidates_for_labeling
    data = load_group_b()
    df = data["df"]
    tr = _labels(LABELS_FILE)
    ev = _labels(EVAL_FILE)
    excl_obj = set(ev["diaObjectId"].astype(np.int64)) if not ev.empty else set()
    excl = set(np.where(df["diaObjectId"].isin(excl_obj))[0])
    if not tr.empty:
        excl |= set(np.where(df["diaSourceId"].isin(tr["diaSourceId"]))[0])
    probs = margins = entropies = None
    if not os.path.exists(MODEL_FILE) and len(tr) >= 6:
        retrain(data)          # 取り込んだラベルなどがあれば、最初にモデルを作ってから候補を選ぶ
    if os.path.exists(MODEL_FILE) and len(tr) >= 6:
        rf = joblib.load(MODEL_FILE)
        probs = predict_full(rf, data["X"])
        srt = np.sort(probs, axis=1)[:, ::-1]
        margins = srt[:, 0] - srt[:, 1]
        entropies = -(probs * np.log2(probs + 1e-12)).sum(1)
    cand = sample_candidates_for_labeling(embeddings=data["X"], margins=margins, entropies=entropies,
                                          already_labeled_indices=excl, n_samples=n_samples, probs=probs, classes=CLASSES)
    print(f"Selected {len(cand)} group-B candidates: {pd.Series([w for _, w in cand]).value_counts().to_dict()}")
    rows = df.iloc[[i for i, _ in cand]]
    n = _review(data, rows, LABELS_FILE, [w for _, w in cand], probs=probs, interactive=interactive, header="Group B")
    print(f"\n[+] Added {n} group-B labels.")
    retrain(data)

def classify() -> pd.DataFrame:
    """グループ B の全ソースを判定し、data/predictions_step2b.csv に保存"""
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"{MODEL_FILE} not found. Label with --mode b_active_loop first.")
    data = load_group_b()
    rf = joblib.load(MODEL_FILE)
    P = predict_full(rf, data["X"])
    out = data["df"][["diaSourceId", "diaObjectId"]].copy()
    for k, c in enumerate(CLASSES):
        out[f"prob_b_{c}"] = P[:, k]
    out["class_b"] = [CLASSES[i] for i in P.argmax(1)]
    out["decision_b"] = out["class_b"].map(FINAL_DECISION)
    out.to_csv(PRED_FILE, index=False)
    print(f"[+] Group-B predictions for {len(out)} sources -> {PRED_FILE}")
    print(out["class_b"].value_counts().to_string())
    return out

def seed_from_step2() -> None:
    """既存の step2 のラベルのうちグループ B に入るもの (no_host / bogus / galaxy) を、学習用ラベルの出発点として取り込む"""
    data = load_group_b()
    ids = set(data["df"]["diaSourceId"].astype(np.int64))
    old = pd.concat([load_human_labels(os.path.join(DATA_DIR, "human_labels_step2.csv")),
                     load_human_labels(os.path.join(DATA_DIR, "eval_labels_step2.csv"))]).drop_duplicates("diaSourceId")
    old = old[old["diaSourceId"].astype(np.int64).isin(ids) & old["class_name"].isin(["no_host", "bogus", "galaxy"])]
    cur = _labels(LABELS_FILE)
    ev = _labels(EVAL_FILE)
    skip = set(cur["diaSourceId"]) if not cur.empty else set()
    skip |= set(ev["diaSourceId"]) if not ev.empty else set()
    old = old[~old["diaSourceId"].isin(skip)]
    if old.empty:
        print("[*] Nothing to import.")
        return
    new = pd.DataFrame({"diaSourceId": old["diaSourceId"].astype(np.int64), "diaObjectId": old["diaObjectId"].astype(np.int64),
                        "label": old["class_name"].map({c: i for i, c in enumerate(CLASSES)}).astype(int),
                        "class_name": old["class_name"], "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "selection": "seed_from_step2"})
    out = pd.concat([cur, new], ignore_index=True) if not cur.empty else new
    os.makedirs(os.path.dirname(LABELS_FILE) or ".", exist_ok=True)
    out.to_csv(LABELS_FILE, index=False)
    print(f"[+] Imported {len(new)} step2 labels into {LABELS_FILE}: {new['class_name'].value_counts().to_dict()}")
    print("    (old 'galaxy' labels in group B were given under the old definition; review them with the relabel step if needed)")

# ---------------------------------------------------------------------------
# step2 の最終判定 (従来の RF は使わない)
# ---------------------------------------------------------------------------
FINAL_FILE = os.path.join(DATA_DIR, "predictions_step2_final.csv")
HOST_A_TO_DECISION = {"point": "star", "extended": "galaxy", "uncertain": "uncertain"}

def write_final() -> pd.DataFrame:
    """
    step2 の最終判定を data/predictions_step2_final.csv に書く。
      グループ A: 天体単位の母天体の形だけで決める (point -> star, extended -> galaxy, uncertain -> uncertain)
      グループ B: 4 クラスの RF (data/predictions_step2b.csv) で決める
                  (no_host -> no_host, real_uncertain -> uncertain, galaxy -> galaxy, bogus -> bogus)。
                  まだ判定が無いソースは uncertain (decision_detail = 'B:no_model')
      除外した visit のソースは bogus
    """
    from .visit_quality import excluded_source_ids
    H = pd.read_csv(os.path.join(DATA_DIR, "host_morph_step2.csv"))
    src = pd.read_csv(os.path.join(DATA_DIR, "step2_sources.csv"), usecols=["diaSourceId", "midpointMjdTai"])
    keep = ["diaSourceId", "diaObjectId", "band", "host_group", "host_morph_obj", "host_score", "band_scores",
            "tmp_snr", "tmp_size_ratio", "tmp_center_snr", "tmp_brightest_snr", "tmp_brightest_dist_fwhm"]
    F = H[[c for c in keep if c in H.columns]].merge(src, on="diaSourceId", how="left")
    F = F.rename(columns={"host_morph_obj": "host_morph"})
    a = F["host_group"] == "A"
    F["decision"] = "uncertain"
    F["decision_detail"] = ""
    F.loc[a, "decision"] = F.loc[a, "host_morph"].map(HOST_A_TO_DECISION).fillna("uncertain")
    F.loc[a, "decision_detail"] = "A:" + F.loc[a, "host_morph"].astype(str)
    F.loc[~a, "decision_detail"] = "B:no_model"
    if os.path.exists(PRED_FILE):
        pb = pd.read_csv(PRED_FILE)
        F = F.merge(pb.drop(columns=["diaObjectId"], errors="ignore"), on="diaSourceId", how="left")
        b = (~a) & F["class_b"].notna()
        F.loc[b, "decision"] = F.loc[b, "class_b"].map(FINAL_DECISION)
        F.loc[b, "decision_detail"] = "B:" + F.loc[b, "class_b"]
    ex = F["diaSourceId"].isin(excluded_source_ids())
    F.loc[ex, "decision"] = "bogus"
    F.loc[ex, "decision_detail"] = "excluded_visit"
    F.to_csv(FINAL_FILE, index=False)
    print(f"\n[+] Step 2 final decisions (no legacy RF) -> {FINAL_FILE}")
    print(pd.crosstab(F["decision_detail"], F["decision"]).to_string())
    print("Final decision:", F["decision"].value_counts().to_dict())
    if (F["decision_detail"] == "B:no_model").any():
        print("    (group B has no model predictions yet: run --mode b_classify after labeling)")
    return F
