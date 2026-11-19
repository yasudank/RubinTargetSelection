"""
Visit Quality: exclusion list and suspicious-visit report
- data/excluded_visits.csv に載せた visit のソースを、天体単位の集計 (aggregate_objects) 以降で除外する
  (step1 の判定は bogus 扱いにする)。DB や step1 のキャッシュは変更しない。
- 怪しい visit の自動検出 (報告のみ。除外はしない):
  主な基準: positive_real のソースで、Template の中の最も明るい点の位置 (中心からの方向と距離) がそろっている
      (2026011700646 では、すべての「突発天体」から同じ方向・同じ距離に明るい恒星があった)。
      positive_real が min_pos_checked 件以上あるすべての visit について調べる (max_checked を指定すると件数の多い上位だけ)。
      件数が少なくても起きることがある (2026011700639 は positive_real 78 件で、そろい具合 0.94)。
  参考情報: positive_real の件数がバンドの中央値より極端に多い visit。
      step1 は「同じ天体・同じ夜・同じバンドは最新の 1 件」だけを使うので、連続撮像の最後の visit に件数が集まる。
      件数だけでは異常とは言えないので、報告するだけで「怪しい」には数えない。
"""

import os
import time
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import DATA_DIR

EXCLUDED_VISITS_FILE = os.path.join(DATA_DIR, "excluded_visits.csv")
EXCLUDED_SOURCES_FILE = os.path.join(DATA_DIR, "excluded_sources.csv")
VISIT_REPORT_FILE = os.path.join(DATA_DIR, "visit_quality_report.csv")

def load_excluded_visits() -> pd.DataFrame:
    if not os.path.exists(EXCLUDED_VISITS_FILE):
        return pd.DataFrame(columns=["visit", "band", "reason", "added"])
    return pd.read_csv(EXCLUDED_VISITS_FILE)

def excluded_source_ids(db_config=None, refresh: bool = False) -> np.ndarray:
    """除外する visit のソース (diaSourceId)。DB から読み、一覧が変わったときだけ作り直す"""
    ex = load_excluded_visits()
    if ex.empty:
        return np.array([], dtype=np.int64)
    visits = sorted(ex["visit"].astype(np.int64).tolist())
    if os.path.exists(EXCLUDED_SOURCES_FILE) and not refresh:
        cached = pd.read_csv(EXCLUDED_SOURCES_FILE)
        if sorted(cached["visit"].unique().astype(np.int64).tolist()) == visits:
            return cached["diaSourceId"].astype(np.int64).values
    from sqlalchemy import create_engine, text
    from .config import DBConfig
    engine = create_engine((db_config or DBConfig()).url)
    with engine.connect() as conn:
        d = pd.read_sql_query(text('SELECT "diaSourceId", visit FROM "diaSource" WHERE visit IN ('
                                   + ",".join(str(v) for v in visits) + ")"), conn)
    d.to_csv(EXCLUDED_SOURCES_FILE, index=False)
    print(f"[*] {len(d)} sources from {len(visits)} excluded visit(s) -> {EXCLUDED_SOURCES_FILE}")
    return d["diaSourceId"].astype(np.int64).values

def _visit_band(visit: int) -> str:
    from sqlalchemy import create_engine, text
    from .config import DBConfig
    with create_engine(DBConfig().url).connect() as conn:
        b = conn.execute(text(f'SELECT band FROM "diaSource" WHERE visit={int(visit)} LIMIT 1')).scalar()
    return b or ""

def add_excluded_visit(visit: int, band: str, reason: str) -> None:
    band = band or _visit_band(visit)
    ex = load_excluded_visits()
    if int(visit) in set(ex["visit"].astype(np.int64)):
        print(f"[*] visit {visit} is already excluded.")
        return
    ex = pd.concat([ex, pd.DataFrame([{"visit": int(visit), "band": band, "reason": reason,
                                       "added": time.strftime("%Y-%m-%d")}])], ignore_index=True)
    os.makedirs(os.path.dirname(EXCLUDED_VISITS_FILE) or ".", exist_ok=True)
    ex.to_csv(EXCLUDED_VISITS_FILE, index=False)
    print(f"[+] Added visit {visit} to {EXCLUDED_VISITS_FILE}")

def _offset_consistency(rows: pd.DataFrame, fits_root: str, n_sample: int = 60, seed: int = 0):
    """Template の最も明るい点の、中心からのずれのベクトルがそろっている割合 (中央値のベクトルから 2 px 以内)"""
    from .dataset import get_cutout_path
    from .labeling_viewer import read_stamp_fits
    r = rows.sample(min(n_sample, len(rows)), random_state=seed)
    vec = []
    for x in r.itertuples():
        t = read_stamp_fits(get_cutout_path(fits_root, x.diaObjectId, x.midpointMjdTai, x.band, x.diaSourceId, "template"))
        if not np.any(t):
            continue
        py, px = np.unravel_index(np.argmax(t), t.shape)
        d = (px - 14.5, py - 14.5)
        if np.hypot(*d) > 3:          # 中心 (突発天体の位置) 以外に明るい点があるものだけ
            vec.append(d)
    if len(vec) < 10:
        return np.nan, len(vec), (np.nan, np.nan)
    v = np.array(vec)
    med = np.median(v, axis=0)
    frac = float(np.mean(np.hypot(*(v - med).T) <= 2.0))
    return frac, len(v), (float(med[0]), float(med[1]))

def report_suspicious_visits(pred_step1_csv: str, fits_root: str = "cutouts", db_config=None,
                             count_factor: float = 10.0, min_pos: int = 300, max_checked: Optional[int] = None,
                             min_pos_checked: int = 10,
                             consistency_min: float = 0.5) -> pd.DataFrame:
    """怪しい visit を報告する (除外はしない)。結果は data/visit_quality_report.csv"""
    from sqlalchemy import create_engine, text
    from .config import DBConfig
    engine = create_engine((db_config or DBConfig()).url)
    with engine.connect() as conn:
        src = pd.read_sql_query(text('SELECT "diaSourceId", "diaObjectId", "midpointMjdTai", band, visit FROM "diaSource"'), conn)
    s1 = pd.read_csv(pred_step1_csv, usecols=["diaSourceId", "decision"])
    d = src.merge(s1, on="diaSourceId", how="left")
    v = d.groupby(["visit", "band"]).agg(n_all=("diaSourceId", "size"), n_cat=("decision", "count"),
                                         n_pos=("decision", lambda s: int((s == "positive_real").sum()))).reset_index()
    med = v[v["n_cat"] > 0].groupby("band")["n_pos"].median()
    v["pos_median_band"] = v["band"].map(med)
    v["pos_ratio"] = v["n_pos"] / v["pos_median_band"].clip(lower=1)
    ex = set(load_excluded_visits()["visit"].astype(np.int64))
    v["excluded"] = v["visit"].isin(ex)
    # 件数で候補を絞り、上位だけ Template のずれの向きを調べる
    cand = v[v["n_pos"] >= min_pos_checked].sort_values("n_pos", ascending=False)
    if max_checked:
        cand = cand.head(max_checked)
    print(f"Checking template bright-neighbour offsets for {len(cand)} visits (n_pos >= {min_pos_checked})...")
    cons = []
    for r in cand.itertuples():
        rows = d[(d["visit"] == r.visit) & (d["decision"] == "positive_real")]
        frac, n, vec = _offset_consistency(rows, fits_root, n_sample=40) if len(rows) >= 10 else (np.nan, 0, (np.nan, np.nan))
        cons.append((r.visit, frac, n, vec[0], vec[1]))
    c = pd.DataFrame(cons, columns=["visit", "offset_consistency", "n_offset_checked", "offset_dx_px", "offset_dy_px"])
    v = v.merge(c, on="visit", how="left")
    v["flag_count"] = (v["n_pos"] >= min_pos) & (v["pos_ratio"] >= count_factor)
    v["flag_offset"] = v["offset_consistency"] >= consistency_min
    v["suspicious"] = v["flag_offset"] & ~v["excluded"]
    v = v.sort_values(["suspicious", "pos_ratio"], ascending=[False, False])
    v.to_csv(VISIT_REPORT_FILE, index=False)
    cols = ["visit", "band", "n_all", "n_cat", "n_pos", "pos_ratio", "offset_consistency", "offset_dx_px", "offset_dy_px",
            "flag_count", "flag_offset", "excluded"]
    print(f"\n=== Visit quality report ({len(v)} visits) -> {VISIT_REPORT_FILE} ===")
    print(f"Criterion: template bright-neighbour offset consistency >= {consistency_min} "
          f"(checked for {len(cand)} visits with >= {min_pos_checked} positive_real)")
    print("Median positive_real per visit by band:", med.round(1).to_dict())
    exr = v[v["excluded"]]
    if len(exr):
        print("\nExcluded visits (data/excluded_visits.csv):")
        print(exr[cols].round(2).to_string(index=False))
    sus = v[v["suspicious"]]
    if len(sus):
        print(f"\n[!] {len(sus)} suspicious visit(s) (NOT excluded; review them, and add to the exclusion list if needed):")
        print(sus[cols].round(2).to_string(index=False))
    else:
        print("\nNo other suspicious visits.")
    many = v[v["flag_count"] & ~v["excluded"] & ~v["suspicious"]]
    print(f"\nFor reference: {len(many)} visit(s) with many positive_real (>= {min_pos} and >= {count_factor:g} x band median) "
          f"but no offset pattern. Usually the last visit of a consecutive block (step1 keeps the latest per object/night/band) "
          f"or nights with many asteroids. Top 10:")
    print(many.sort_values("n_pos", ascending=False).head(10)[cols].round(2).to_string(index=False))
    return v
