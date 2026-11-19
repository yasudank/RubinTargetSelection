"""
Object-level Aggregation of Step 1 Decisions
classify_all (固定モデル) のソース単位の判定を天体 (diaObjectId) 単位にまとめ、step2 (母天体分類) の入力ソースを選ぶ。

天体の判定 (object_class):
  real      : real (positive_real / negative_real) と判定されたソースが min_real_sources 件以上ある
  uncertain : real は足りないが、real または uncertain と判定されたソースがある
  bogus     : すべてのソースが bogus
real_type: real ソースの種類 (positive / negative / both)。real 以外では空。
"""

import json
import os
import shutil
import time
from typing import Dict, List

import numpy as np
import pandas as pd

REAL = ["positive_real", "negative_real"]

def aggregate_objects(df: pd.DataFrame, min_real_sources: int = 1) -> pd.DataFrame:
    d = df.copy()
    d["real_score"] = 1.0 - d["prob_bogus"]
    d["day_obs"] = np.floor(d["midpointMjdTai"] - 0.5).astype(int)
    for c in REAL + ["uncertain", "bogus"]:
        d[f"is_{c}"] = (d["decision"] == c).astype(int)

    g = d.groupby("diaObjectId", sort=False)
    out = pd.DataFrame({
        "n_sources": g.size(),
        "n_positive_real": g["is_positive_real"].sum(),
        "n_negative_real": g["is_negative_real"].sum(),
        "n_uncertain": g["is_uncertain"].sum(),
        "n_bogus": g["is_bogus"].sum(),
        "n_days": g["day_obs"].nunique(),
        "bands": g["band"].agg(lambda s: "".join(sorted(set(map(str, s))))),
        "mjd_first": g["midpointMjdTai"].min(),
        "mjd_last": g["midpointMjdTai"].max(),
        "max_prob_positive_real": g["prob_positive_real"].max(),
        "max_prob_negative_real": g["prob_negative_real"].max(),
        "max_real_score": g["real_score"].max(),
        "mean_real_score": g["real_score"].mean(),
    })
    # real らしさが最大のソース (天体の代表; step2 の 'best' 選択や画像確認に使う)
    best_idx = g["real_score"].idxmax()
    out["best_diaSourceId"] = d.loc[best_idx.values, "diaSourceId"].values

    n_real = out["n_positive_real"] + out["n_negative_real"]
    out["n_real"] = n_real
    out["real_fraction"] = n_real / out["n_sources"]
    out["object_class"] = np.where(n_real >= min_real_sources, "real",
                                   np.where((n_real + out["n_uncertain"]) > 0, "uncertain", "bogus"))
    pos, neg = out["n_positive_real"] > 0, out["n_negative_real"] > 0
    out["real_type"] = np.where(out["object_class"] != "real", "",
                                np.where(pos & neg, "both", np.where(pos, "positive", "negative")))
    out = out.reset_index()
    if "model_version" in df.columns:
        out["model_version"] = df["model_version"].iloc[0]
    cols = ["diaObjectId", "object_class", "real_type", "n_sources", "n_real", "n_positive_real", "n_negative_real",
            "n_uncertain", "n_bogus", "real_fraction", "n_days", "bands", "mjd_first", "mjd_last",
            "max_prob_positive_real", "max_prob_negative_real", "max_real_score", "mean_real_score", "best_diaSourceId"]
    cols += [c for c in out.columns if c not in cols]
    return out[cols].sort_values(["object_class", "max_real_score"], ascending=[True, False]).reset_index(drop=True)

def select_step2_sources(df_src: pd.DataFrame, df_obj: pd.DataFrame, object_classes: List[str],
                         mode: str = "real_uncertain", include_moving: bool = False) -> pd.DataFrame:
    """
    step2 の入力ソースを選ぶ。
      real_uncertain : 対象天体のソースのうち、step1 の判定が real / uncertain のもの (デフォルト)
      all            : 対象天体の全ソース (bogus と判定された観測も含む)
      best           : 対象天体ごとに real らしさが最大の 1 ソース
    """
    sel_obj = df_obj[df_obj["object_class"].isin(object_classes)]
    if not include_moving and "moving_candidate" in sel_obj.columns:
        sel_obj = sel_obj[~sel_obj["moving_candidate"]]
    obj_class = dict(zip(sel_obj["diaObjectId"], sel_obj["object_class"]))
    if mode == "best":
        src = df_src[df_src["diaSourceId"].isin(sel_obj["best_diaSourceId"])]
    else:
        src = df_src[df_src["diaObjectId"].isin(sel_obj["diaObjectId"])]
        if mode == "real_uncertain":
            src = src[src["decision"].isin(REAL + ["uncertain"])]
        elif mode != "all":
            raise ValueError(f"Unknown step2 source selection mode: {mode}")
    cols = ["diaSourceId", "diaObjectId", "midpointMjdTai", "band", "decision",
            "prob_positive_real", "prob_negative_real", "prob_bogus"]
    out = src[cols].copy()
    out["object_class"] = out["diaObjectId"].map(obj_class)
    return out.reset_index(drop=True)

def write_step2_source_list(df_sel: pd.DataFrame, versioned_path: str, active_path: str, provenance: Dict) -> None:
    """版付きのファイルと、step2 が読む既定のファイルの両方に保存 (既定ファイルが別内容なら退避)"""
    os.makedirs(os.path.dirname(versioned_path) or ".", exist_ok=True)
    df_sel.to_csv(versioned_path, index=False)
    if os.path.exists(active_path):
        old = pd.read_csv(active_path, usecols=["diaSourceId"])["diaSourceId"]
        if set(old.astype(np.int64)) != set(df_sel["diaSourceId"].astype(np.int64)):
            bak = f"{active_path}.bak_{time.strftime('%Y%m%d_%H%M%S')}"
            shutil.move(active_path, bak)
            print(f"[*] Existing step2 source list differs; moved it to {bak}")
            meta_old = os.path.splitext(active_path)[0] + ".json"
            if os.path.exists(meta_old):
                shutil.move(meta_old, bak + ".json")
    shutil.copy2(versioned_path, active_path)
    with open(os.path.splitext(active_path)[0] + ".json", "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=False, default=str)

# ---------------------------------------------------------------------------
# 検出履歴 (diaSource + prvDiaSources) の天体単位集計
# ---------------------------------------------------------------------------
# diaSource には受信したアラートの検出だけが入り、それ以前の検出はアラートに同梱される履歴 (prvDiaSources)
# にしか無い。両者を diaSourceId で重複排除して合わせ、天体ごとの真の検出回数・夜数・期間を求める。
# DB は読み取りのみ。prvDiaSources は数億行あるため結果をローカルにキャッシュし、
# 取り込み済みファイル数や diaSource の行数が変わったときだけ再計算する。

HISTORY_SQL = """
    SELECT "diaObjectId", "diaSourceId", "midpointMjdTai", band, ra, dec FROM "diaSource" WHERE "diaObjectId" IS NOT NULL
    UNION
    SELECT DISTINCT "diaObjectId", "diaSourceId", "midpointMjdTai", band, ra, dec FROM "prvDiaSources" WHERE "diaObjectId" IS NOT NULL
"""

# キャッシュの形式が変わったら再計算させるための版番号 (2: 同一夜内の移動量を追加)
HISTORY_SCHEMA_VERSION = 2
# 同一夜内の 2 検出の時間差がこれより短い場合は、速度計算の分母をこの値で下限クリップする (位置誤差で速度が発散しないように)
MIN_DT_MINUTES = 0.5

def _table_counts(conn) -> Dict[str, int]:
    from sqlalchemy import text
    # キャッシュの鮮度判定用。prvDiaSources (数億行) は数えるのに数分かかるため、
    # 取り込み済みファイル数 (processed_files) と diaSource の行数で代用する (prvDiaSources はファイル取り込み時にしか増えない)
    return {t: int(conn.execute(text(f'SELECT count(*) FROM "{t}"')).scalar()) for t in ["processed_files", "diaSource"]}

def load_detection_history(db_config, cache_file: str, refresh: bool = False) -> pd.DataFrame:
    """天体ごとの検出履歴の集計 (キャッシュ付き)"""
    from sqlalchemy import create_engine, text
    meta_file = os.path.splitext(cache_file)[0] + ".json"
    engine = create_engine(db_config.url)
    with engine.connect() as conn:
        counts = _table_counts(conn)
        if not refresh and os.path.exists(cache_file) and os.path.exists(meta_file):
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("table_counts") == counts and meta.get("schema_version") == HISTORY_SCHEMA_VERSION:
                print(f"[*] Using cached detection history: {cache_file} (created {meta.get('created')})")
                return pd.read_parquet(cache_file)
            print("[*] DB tables changed since the detection history cache was made; recomputing.")
        print("Reading all detections from diaSource + prvDiaSources (read-only; this can take ~10 min)...")
        t0 = time.time()
        conn.execute(text("SET work_mem = '2GB';"))
        det = pd.read_sql_query(text(HISTORY_SQL), conn)
    hist = summarize_detections(det)
    os.makedirs(os.path.dirname(cache_file) or ".", exist_ok=True)
    hist.to_parquet(cache_file, index=False)
    with open(meta_file, "w") as f:
        json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "table_counts": counts,
                   "schema_version": HISTORY_SCHEMA_VERSION,
                   "n_detections": int(len(det)), "n_objects": int(len(hist))}, f, indent=2)
    print(f"[+] Detection history of {len(hist)} objects ({len(det)} unique detections) "
          f"in {time.time() - t0:.0f}s -> {cache_file}")
    return hist

def summarize_detections(det: pd.DataFrame) -> pd.DataFrame:
    """検出 (diaObjectId, diaSourceId, midpointMjdTai, band, ra, dec) から天体ごとの検出履歴の集計を作る"""
    det = det.drop_duplicates("diaSourceId").copy()
    det["night"] = np.floor(det["midpointMjdTai"] - 0.5).astype(int)
    per_night = det.groupby(["diaObjectId", "night"]).size()
    g = det.groupby("diaObjectId")
    hist = pd.DataFrame({
        "n_det_total": g.size(),
        "n_nights_total": per_night.groupby(level=0).size(),
        "max_det_per_night": per_night.groupby(level=0).max(),
        "n_bands_total": g["band"].nunique(),
        "first_det_mjd": g["midpointMjdTai"].min(),
        "last_det_mjd": g["midpointMjdTai"].max(),
    })
    hist["det_duration_days"] = hist["last_det_mjd"] - hist["first_det_mjd"]
    hist = hist.join(intra_night_motion(det))
    return hist.reset_index()

def detection_history_for_objects(conn, object_ids) -> pd.DataFrame:
    """指定した天体だけの検出履歴 (diaSource + prvDiaSources、読み取りのみ)。新しいデータの処理用"""
    from sqlalchemy import text
    ids = sorted({int(o) for o in object_ids})
    parts = []
    for i in range(0, len(ids), 5000):
        chunk = ",".join(str(o) for o in ids[i:i + 5000])
        for table, distinct in [("diaSource", ""), ("prvDiaSources", "DISTINCT ")]:
            parts.append(pd.read_sql_query(text(
                f'SELECT {distinct}"diaObjectId", "diaSourceId", "midpointMjdTai", band, ra, dec '
                f'FROM "{table}" WHERE "diaObjectId" IN ({chunk})'), conn))
    det = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["diaObjectId", "diaSourceId", "midpointMjdTai", "band", "ra", "dec"])
    return summarize_detections(det)

def intra_night_motion(det: pd.DataFrame) -> pd.DataFrame:
    """
    同一夜に 2 回以上検出された天体について、その夜の最初と最後の検出の位置のずれ (秒角) と移動速度 (秒角/分) を求める。
    複数の夜で該当する場合は、ずれが最大の夜の値を天体の値とする。
    """
    d = det.sort_values(["diaObjectId", "night", "midpointMjdTai"])
    g = d.groupby(["diaObjectId", "night"])
    first, last, n = g.first(), g.last(), g.size()
    m = n >= 2
    first, last = first[m], last[m]
    dec_mid = np.radians(0.5 * (first["dec"].values + last["dec"].values))
    dra = (last["ra"].values - first["ra"].values + 540.0) % 360.0 - 180.0   # RA の 0/360 をまたぐ場合に対応
    dra_arcsec = dra * np.cos(dec_mid) * 3600.0
    ddec_arcsec = (last["dec"].values - first["dec"].values) * 3600.0
    dt_min = (last["midpointMjdTai"].values - first["midpointMjdTai"].values) * 1440.0
    per_night = pd.DataFrame({
        "intra_night_sep_arcsec": np.hypot(dra_arcsec, ddec_arcsec),
        "intra_night_dt_min": dt_min,
    }, index=first.index)
    per_night["intra_night_rate_arcsec_per_min"] = per_night["intra_night_sep_arcsec"] / np.maximum(dt_min, MIN_DT_MINUTES)
    best = per_night.reset_index().sort_values("intra_night_sep_arcsec").groupby("diaObjectId").last()
    return best[["intra_night_sep_arcsec", "intra_night_dt_min", "intra_night_rate_arcsec_per_min"]]

def flag_moving(df_obj: pd.DataFrame, min_rate: float, min_sep: float) -> pd.DataFrame:
    """
    移動天体の候補: 1 夜だけで検出され、その夜の 2 検出の間に min_sep 秒角以上、min_rate 秒角/分以上の速さで動いているもの。
    (複数の夜に同じ diaObjectId で検出される天体は、その場に留まっているとみなす)
    """
    out = df_obj.copy()
    out["moving_candidate"] = ((out["n_nights_total"] == 1)
                               & (out["intra_night_sep_arcsec"] >= min_sep)
                               & (out["intra_night_rate_arcsec_per_min"] >= min_rate)).fillna(False).astype(bool)
    return out

def add_detection_history(df_obj: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """
    天体表に検出履歴の列を加える。
    detection_pattern:
      multi_night        : 2 夜以上で検出
      single_night_multi : 1 夜だけだが、その夜に 2 回以上検出
      single             : 検出は 1 回だけ
    """
    out = df_obj.merge(hist, on="diaObjectId", how="left")
    out["detection_pattern"] = np.where(out["n_nights_total"] >= 2, "multi_night",
                                        np.where(out["max_det_per_night"] >= 2, "single_night_multi",
                                                 np.where(out["n_det_total"] == 1, "single", "")))
    return out
