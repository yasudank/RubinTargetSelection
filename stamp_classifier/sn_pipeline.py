"""
SN Candidate Pipeline (inference on new data)
固定したモデル一式 (bundle) を使い、PostgreSQL のアラート DB の指定期間のデータから超新星候補を選ぶ。

  freeze: 現在のモデルと閾値を models/sn_pipeline/<version>/ に固定する
  run   : DB 名と観測夜 (dayObs) の範囲を受け取り、step1 → 天体集計 → step2 → 候補選択 → 天体ごとの CSV / PDF を出力する

DB は読み取りだけ。カットアウトの FITS は、取り込み (avro_to_fits_pg.py) で作られたものを使う。
"""

import hashlib
import json
import os
import shutil
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .config import DBConfig, StepConfig, MODELS_DIR, DATA_DIR

BUNDLE_ROOT = os.path.join(MODELS_DIR, "sn_pipeline")
S1_CLASSES = ["positive_real", "negative_real", "bogus"]

def _md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

# ---------------------------------------------------------------------------
# モデルの固定
# ---------------------------------------------------------------------------
def freeze_bundle(version: str, step1_version: str = "v20261019", moving_min_sep: float = 1.0,
                  moving_min_rate: float = 0.3) -> str:
    out = os.path.join(BUNDLE_ROOT, version)
    if os.path.exists(out):
        raise FileExistsError(f"Bundle '{version}' already exists: {out}")
    s1_meta_path = os.path.join(MODELS_DIR, "step1", f"rf_{step1_version}.json")
    with open(s1_meta_path) as f:
        s1_meta = json.load(f)
    s1_enc = os.path.join(MODELS_DIR, "step1", "best_simclr_encoder.pt")
    if _md5(s1_enc) != s1_meta.get("encoder_md5"):
        raise RuntimeError(f"The step1 encoder changed since {step1_version} was frozen; its RF would get wrong features.")
    # RF は保存済みの埋め込みで学習されている。今のエンコーダでその埋め込みが再現できるかを確かめる
    check_encoder_reproduces_cache(1, s1_enc, ["difference"])
    check_encoder_reproduces_cache(2, os.path.join(MODELS_DIR, "step2", "best_simclr_encoder.pt"), ["science", "template"])
    from .host_morphology import HOST_FEATURE_VERSION, HOST_MORPH_FEATURE_NAMES
    with open(os.path.join(DATA_DIR, "host_morph_thresholds_step2.json")) as f:
        host_thr = json.load(f)
    if host_thr.get("feature_version", 1) != HOST_FEATURE_VERSION:
        raise RuntimeError("Host-shape thresholds were derived for another feature version. Run --step 2 --mode host_morph first.")
    files = {
        "step1_encoder.pt": s1_enc,
        "step1_rf.joblib": os.path.join(MODELS_DIR, "step1", f"rf_{step1_version}.joblib"),
        "step2_encoder.pt": os.path.join(MODELS_DIR, "step2", "best_simclr_encoder.pt"),
        "step2b_rf.joblib": os.path.join(MODELS_DIR, "step2b", "random_forest.joblib"),
        "host_thresholds.json": os.path.join(DATA_DIR, "host_morph_thresholds_step2.json"),
    }
    os.makedirs(out)
    for name, src in files.items():
        shutil.copy2(src, os.path.join(out, name))
    rf_b = joblib.load(os.path.join(out, "step2b_rf.joblib"))
    manifest = {
        "version": version,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step1": {"model_version": step1_version, "classes": S1_CLASSES, "thresholds": s1_meta["thresholds"],
                  "eval": s1_meta.get("eval_set")},
        "moving": {"min_sep_arcsec": moving_min_sep, "min_rate_arcsec_per_min": moving_min_rate},
        "host": {"feature_version": HOST_FEATURE_VERSION, "feature_names": HOST_MORPH_FEATURE_NAMES},
        "group_b": {"classes": ["no_host", "real_uncertain", "galaxy", "bogus"], "n_features": int(rf_b.n_features_in_)},
        "visit_check": {"consistency_min": 0.5, "min_offset_px": 5.0, "min_pos": 10},
        "md5": {name: _md5(os.path.join(out, name)) for name in files},
        "sources": files,
    }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[+] Froze SN pipeline bundle '{version}' -> {out}")
    return out

def check_encoder_reproduces_cache(step: int, encoder_path: str, image_types: List[str], n: int = 30,
                                   tol: float = 0.02) -> None:
    """保存済みの埋め込み (RF の学習に使ったもの) が、このエンコーダで再現できるか (ランダムな n ソースで比べる)"""
    from .feature_extractor import load_cached_embeddings
    c = load_cached_embeddings(step)
    rng = np.random.default_rng(0)
    i = rng.choice(len(c["diaSourceIds"]), size=min(n, len(c["diaSourceIds"])), replace=False)
    df = pd.DataFrame({"diaSourceId": c["diaSourceIds"][i], "diaObjectId": c["diaObjectIds"][i],
                       "midpointMjdTai": c["midpointMjdTai"][i], "band": c["bands"][i]})
    e, _ = encode(df, image_types, encoder_path, "cutouts", workers=4)
    diff = float(np.abs(e - c["simclr_embeddings"][i]).max())
    if diff > tol:
        raise RuntimeError(f"Step {step}: the encoder {encoder_path} does not reproduce the cached embeddings used to "
                           f"train the RF (max |diff| = {diff:.3f}). Re-extract the embeddings and retrain the RF first.")
    print(f"[*] Step {step} encoder reproduces the cached embeddings (max |diff| = {diff:.4f}).")

def load_bundle(version: str) -> Dict:
    d = os.path.join(BUNDLE_ROOT, version)
    with open(os.path.join(d, "manifest.json")) as f:
        m = json.load(f)
    for name, h in m["md5"].items():
        if _md5(os.path.join(d, name)) != h:
            raise RuntimeError(f"{name} in bundle '{version}' was modified (md5 mismatch).")
    with open(os.path.join(d, "host_thresholds.json")) as f:
        m["host_thresholds"] = json.load(f)
    m["dir"] = d
    return m

# ---------------------------------------------------------------------------
# 特徴量
# ---------------------------------------------------------------------------
def _load_encoder(path: str, in_channels: int, device):
    import torch
    from .models import StampEncoder
    enc = StampEncoder(in_channels=in_channels, latent_dim=128)
    ck = torch.load(path, map_location=device, weights_only=False)
    state = ck.get("encoder_state_dict", ck.get("model_state_dict", {}))
    own = enc.state_dict()
    enc.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items() if k.replace("encoder.", "") in own},
                        strict=False)
    return enc.to(device).eval()

def encode(df: pd.DataFrame, image_types: List[str], encoder_path: str, fits_root: str,
           batch_size: int = 256, workers: int = 8):
    """カットアウトを学習時と同じ前処理で読み、SimCLR の埋め込み (N, 128) と画像 (N, C, 30, 30) を返す"""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from .dataset import PostageStampDataset
    cfg = StepConfig(step=0, name="inference", classes=[], image_types=image_types, in_channels=len(image_types),
                     checkpoint_dir="", embeddings_file="", labels_file="")
    ds = PostageStampDataset(df_records=df, step_config=cfg, fits_root=fits_root, mode="single", cache_in_ram=False)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = _load_encoder(encoder_path, len(image_types), device)
    embs, imgs = [], []
    with torch.no_grad():
        for x, _ in tqdm(dl, desc=f"Encoding {'+'.join(image_types)}"):
            imgs.append(x.numpy().astype(np.float32))
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                embs.append(enc(x.to(device)).float().cpu().numpy())
    if not embs:
        return np.zeros((0, 128), np.float32), np.zeros((0, len(image_types), 30, 30), np.float32)
    return np.concatenate(embs), np.concatenate(imgs)

# ---------------------------------------------------------------------------
# DB
# ---------------------------------------------------------------------------
SRC_COLS = '"diaSourceId", "diaObjectId", "midpointMjdTai", band, ra, dec, visit, "scienceFlux", "scienceFluxErr", "psfFlux", "psfFluxErr"'

def query_sources(conn, d0: int, d1: int) -> pd.DataFrame:
    # 太陽系天体に結び付いたソースは diaObjectId = 0 (ssObjectId 側に ID がある) で毎晩 1-2 割含まれる。
    # 天体として集計すると全視野の寄せ集めになるので、ここで除く。
    from sqlalchemy import text
    return pd.read_sql_query(text(
        f'SELECT {SRC_COLS} FROM "diaSource" WHERE "diaObjectId" > 0 '
        f'AND floor("midpointMjdTai" - 0.5) BETWEEN {int(d0)} AND {int(d1)}'), conn)

def latest_per_night_band(df: pd.DataFrame) -> pd.DataFrame:
    """学習時と同じ: 同じ天体・同じ夜・同じバンドの検出は最新の 1 件だけ"""
    d = df.copy()
    d["dayObs"] = np.floor(d["midpointMjdTai"] - 0.5).astype(int)
    return d.sort_values("midpointMjdTai").groupby(["diaObjectId", "dayObs", "band"]).tail(1).reset_index(drop=True)

def best_snr_per_night_band(df: pd.DataFrame) -> pd.DataFrame:
    """同じ天体・同じ夜・同じバンドの検出から、S/N (|psfFlux| / psfFluxErr) が最大の 1 件を選ぶ。
    負の検出 (暗くなったもの) も絶対値で比べる。S/N が同じ (または求まらない) ときは新しい方。"""
    d = df.copy()
    d["dayObs"] = np.floor(d["midpointMjdTai"] - 0.5).astype(int)
    d["_snr"] = (d["psfFlux"].abs() / d["psfFluxErr"]).fillna(-1.0)
    d = d.sort_values(["_snr", "midpointMjdTai"])
    return d.groupby(["diaObjectId", "dayObs", "band"]).tail(1).drop(columns="_snr").reset_index(drop=True)

PER_NIGHT_BAND = {"snr": best_snr_per_night_band, "latest": latest_per_night_band}

# 1 夜だけで検出された天体の判定に使う: その夜、天体の位置を覆った visit の数
FOOTPRINT_MARGIN_DEG = 0.03   # detector の範囲 (その detector の source の RA/Dec の範囲) に足す余裕

def count_covering_visits(conn, objs: pd.DataFrame, margin_deg: float = FOOTPRINT_MARGIN_DEG) -> pd.Series:
    """
    objs (diaObjectId, ra, dec, night) の各天体について、night (dayObs) の visit のうち、
    天体の位置を覆ったものの数を返す (index = diaObjectId)。
    visit の範囲は detector ごとの source の RA/Dec の範囲 (margin_deg だけ広げる) の和で近似する。
    アラートは領域の一部しか出ていない visit もあるので、夜の visit 数そのものではなくこの方法で数える。
    """
    from sqlalchemy import text
    if objs.empty:
        return pd.Series(dtype=int)
    nights = sorted({int(n) for n in objs["night"]})
    fp = pd.read_sql_query(text(
        'SELECT floor("midpointMjdTai" - 0.5)::int AS night, visit, detector, '
        'min(ra) AS ra0, max(ra) AS ra1, min(dec) AS dec0, max(dec) AS dec1 FROM "diaSource" '
        f'WHERE floor("midpointMjdTai" - 0.5) IN ({",".join(map(str, nights))}) GROUP BY 1, 2, 3'), conn)
    out = {}
    for night, o in objs.groupby("night"):
        f = fp[fp["night"] == night]
        for r in o.itertuples():
            m_ra = margin_deg / max(np.cos(np.radians(r.dec)), 0.1)
            inside = ((f["ra0"] - m_ra <= r.ra) & (r.ra <= f["ra1"] + m_ra) &
                      (f["dec0"] - margin_deg <= r.dec) & (r.dec <= f["dec1"] + margin_deg))
            out[r.diaObjectId] = int(f.loc[inside, "visit"].nunique())
    return pd.Series(out, dtype=int)

# ---------------------------------------------------------------------------
# visit の確認
# ---------------------------------------------------------------------------
def check_visits(src: pd.DataFrame, fits_root: str, cfg: Dict) -> pd.DataFrame:
    """positive_real の多い visit で、Template の明るい点のずれの方向と距離がそろっているかを調べる"""
    from .visit_quality import _offset_consistency
    rows = []
    for v, g in src[src["s1_decision"] == "positive_real"].groupby("visit"):
        if len(g) < cfg["min_pos"]:
            continue
        frac, n, vec = _offset_consistency(g, fits_root, n_sample=40)
        rows.append({"visit": int(v), "band": g["band"].iloc[0], "n_positive_real": len(g),
                     "offset_consistency": frac, "n_checked": n, "offset_dx_px": vec[0], "offset_dy_px": vec[1]})
    r = pd.DataFrame(rows, columns=["visit", "band", "n_positive_real", "offset_consistency", "n_checked",
                                    "offset_dx_px", "offset_dy_px"])
    r["suspicious"] = (r["offset_consistency"] >= cfg["consistency_min"]) & \
                      (np.hypot(r["offset_dx_px"], r["offset_dy_px"]) >= cfg["min_offset_px"])
    return r

# ---------------------------------------------------------------------------
# 実行
# ---------------------------------------------------------------------------
def run(db: str, start: str, end: str, bundle_version: str, out_dir: str, fits_root: str = "cutouts",
        host: str = "localhost", port: int = 5432, user: str = "lsst", password: str = "rubin",
        include_uncertain: bool = True, keep_suspicious_visits: bool = False, per_object_csv: bool = False,
        workers: int = 8, min_nights: int = 1, min_nights_no_host: int = 2,
        include_negative_galaxy: bool = True, per_night_band: str = "snr",
        pfs_semester: str = "s26b", pfs_old_list: str = None) -> pd.DataFrame:
    from astropy.time import Time
    from sqlalchemy import create_engine
    from .model_registry import decide
    from .morphology import extract_morphological_features_batch
    from .object_aggregation import (aggregate_objects, add_detection_history, flag_moving, select_step2_sources,
                                     detection_history_for_objects)
    from .host_morphology import (host_features_batch, object_host_morph, clean_for_rf, HOST_MORPH_FEATURE_NAMES)
    from .group_b import CLASSES as B_CLASSES, FINAL_DECISION as B_FINAL, HOST_A_TO_DECISION
    from .visit_quality import load_excluded_visits

    t0 = time.time()
    B = load_bundle(bundle_version)
    d0, d1 = int(np.floor(Time(start).mjd)), int(np.floor(Time(end).mjd))
    os.makedirs(out_dir, exist_ok=True)
    info = {"db": db, "start": start, "end": end, "dayObs_range": [d0, d1], "bundle": bundle_version,
            "fits_root": fits_root, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    engine = create_engine(DBConfig(host=host, port=port, database=db, user=user, password=password).url)

    # 1. ソース
    with engine.connect() as conn:
        raw = query_sources(conn, d0, d1)
    print(f"[1] {len(raw)} diaSources in dayObs {d0}-{d1} ({start} .. {end}) from database '{db}'")
    if raw.empty:
        print("No data in the range.")
        return pd.DataFrame()
    src = PER_NIGHT_BAND[per_night_band](raw)
    print(f"    {len(src)} sources after keeping the {'highest-S/N' if per_night_band == 'snr' else 'latest'}"
          f" per object / night / band")
    info["per_night_band"] = per_night_band

    # 2. step1
    emb1, img1 = encode(src, ["difference"], os.path.join(B["dir"], "step1_encoder.pt"), fits_root, workers=workers)
    morph1 = extract_morphological_features_batch(img1)
    has_cut = np.abs(img1).reshape(len(img1), -1).sum(1) > 0
    rf1 = joblib.load(os.path.join(B["dir"], "step1_rf.joblib"))
    raw_p = rf1.predict_proba(np.hstack([emb1, morph1]))
    P1 = np.zeros((len(src), 3))
    for j, k in enumerate(rf1.classes_):
        P1[:, int(k)] = raw_p[:, j]
    src["s1_prob_positive_real"], src["s1_prob_negative_real"], src["s1_prob_bogus"] = P1[:, 0], P1[:, 1], P1[:, 2]
    src["s1_decision"] = decide(P1, S1_CLASSES, B["step1"]["thresholds"])
    src.loc[~has_cut, "s1_decision"] = "no_cutout"
    print(f"[2] step1: {src['s1_decision'].value_counts().to_dict()}")

    # 3. visit
    vchk = check_visits(src, fits_root, B["visit_check"])
    vchk.to_csv(os.path.join(out_dir, "visit_check.csv"), index=False)
    known = set(load_excluded_visits()["visit"].astype(np.int64))
    sus = set(vchk.loc[vchk["suspicious"], "visit"].astype(np.int64))
    bad = known | (set() if keep_suspicious_visits else sus)
    src["bad_visit"] = src["visit"].isin(bad)
    src.loc[src["bad_visit"], "s1_decision"] = "bogus"
    info["suspicious_visits"] = sorted(int(v) for v in sus)
    info["excluded_visits_applied"] = sorted(int(v) for v in bad & set(src["visit"].astype(np.int64)))
    print(f"[3] visits: {len(vchk)} checked; suspicious {sorted(sus) or 'none'}"
          f"{' (kept)' if keep_suspicious_visits else ''}; excluded in range: {info['excluded_visits_applied'] or 'none'}")

    # 4. 天体単位の集計 (検出履歴は DB 全体 + prvDiaSources)
    agg_in = src.rename(columns={"s1_decision": "decision", "s1_prob_positive_real": "prob_positive_real",
                                 "s1_prob_negative_real": "prob_negative_real", "s1_prob_bogus": "prob_bogus"})
    agg_in = agg_in[agg_in["decision"] != "no_cutout"]
    obj = aggregate_objects(agg_in)
    with engine.connect() as conn:
        hist = detection_history_for_objects(conn, obj["diaObjectId"].values)
    obj = flag_moving(add_detection_history(obj, hist), B["moving"]["min_rate_arcsec_per_min"], B["moving"]["min_sep_arcsec"])
    s2 = select_step2_sources(agg_in, obj, ["real", "uncertain"], mode="real_uncertain", include_moving=False)
    s2 = s2.merge(src[["diaSourceId", "ra", "dec"]], on="diaSourceId", how="left")
    print(f"[4] objects: {obj['object_class'].value_counts().to_dict()}, moving {int(obj['moving_candidate'].sum())}; "
          f"step2 input {len(s2)} sources of {s2['diaObjectId'].nunique()} objects")

    # 5. step2: 母天体の形 (グループ A) と 4 クラス RF (グループ B)
    F = s2[["diaSourceId", "diaObjectId", "midpointMjdTai", "band"]].copy()
    F["decision"] = pd.Series(dtype=object)
    F["decision_detail"] = pd.Series(dtype=object)
    if len(F):
        H = host_features_batch(F.to_dict("records"), fits_root)
        Hdf = pd.DataFrame(H, columns=HOST_MORPH_FEATURE_NAMES)
        F = pd.concat([F.reset_index(drop=True), Hdf], axis=1)
        O = object_host_morph(F, B["host_thresholds"])
        F = F.merge(O[["diaObjectId", "host_score", "host_morph_obj", "band_scores"]], on="diaObjectId", how="left")
        F["host_group"] = np.where(F["host_morph_obj"].isin(["point", "extended", "uncertain"]), "A", "B")
        F["decision"] = "uncertain"
        F["decision_detail"] = ""
        a = F["host_group"] == "A"
        F.loc[a, "decision"] = F.loc[a, "host_morph_obj"].map(HOST_A_TO_DECISION)
        F.loc[a, "decision_detail"] = "A:" + F.loc[a, "host_morph_obj"]
        bi = np.where(~a)[0]
        F["class_b"] = None
        if len(bi):
            Fb = F.iloc[bi]
            emb2, _ = encode(Fb.merge(s2[["diaSourceId", "ra", "dec"]], on="diaSourceId", how="left"),
                             ["science", "template"], os.path.join(B["dir"], "step2_encoder.pt"), fits_root, workers=workers)
            pos = pd.Series(np.arange(len(src)), index=src["diaSourceId"].values).loc[Fb["diaSourceId"].values].values
            Xb = np.hstack([emb2, emb1[pos], clean_for_rf(morph1[pos]), clean_for_rf(Fb[HOST_MORPH_FEATURE_NAMES].values)])
            rfb = joblib.load(os.path.join(B["dir"], "step2b_rf.joblib"))
            raw_b = rfb.predict_proba(Xb.astype(np.float32))
            Pb = np.zeros((len(Xb), len(B_CLASSES)))
            for j, k in enumerate(rfb.classes_):
                Pb[:, int(k)] = raw_b[:, j]
            cls = np.array(B_CLASSES)[Pb.argmax(1)]
            F.loc[F.index[bi], "class_b"] = cls
            F.loc[F.index[bi], "prob_b"] = Pb.max(1)
            F.loc[F.index[bi], "decision"] = [B_FINAL[c] for c in cls]
            F.loc[F.index[bi], "decision_detail"] = ["B:" + c for c in cls]
        print(f"[5] step2: {F['decision_detail'].value_counts().to_dict()}")

    # 6. 候補
    s2_classes = ["galaxy", "no_host"] + (["uncertain"] if include_uncertain else [])
    full = src.merge(F.drop(columns=["midpointMjdTai", "band", "diaObjectId"], errors="ignore"), on="diaSourceId", how="left")
    full = full.merge(obj[["diaObjectId", "object_class", "n_nights_total", "n_det_total", "max_det_per_night",
                           "first_det_mjd", "last_det_mjd", "moving_candidate"]], on="diaObjectId", how="left")
    full.to_csv(os.path.join(out_dir, "all_sources.csv"), index=False)
    is_pos = (full["s1_decision"] == "positive_real") & full["decision"].isin(s2_classes)
    is_neg_gal = (full["s1_decision"] == "negative_real") & (full["decision"] == "galaxy") if include_negative_galaxy else False
    hit = full[is_pos | is_neg_gal]
    if min_nights is not None and min_nights > 1:
        hit = hit[hit["n_nights_total"] >= min_nights]
    if min_nights_no_host is not None and min_nights_no_host > 1:
        hit = hit[~((hit["decision"] == "no_host") & (hit["n_nights_total"] < min_nights_no_host))]
    # 1 夜だけの天体: その夜に位置を覆った visit が 2 つ以上あれば、その夜に 2 回以上の検出を求める
    # (visit が 1 つだけの夜は確かめようがないので残す)。残ったものの優先度は 2 (pfs_target.compute_priority)。
    single = hit[hit["n_nights_total"] < 2].drop_duplicates("diaObjectId")
    n_single_dropped = 0
    n_cov = pd.Series(dtype=int)
    if len(single):
        objs = pd.DataFrame({"diaObjectId": single["diaObjectId"].values,
                             "ra": single["ra"].values, "dec": single["dec"].values,
                             "night": np.floor(single["first_det_mjd"].values - 0.5).astype(int)})
        with engine.connect() as conn:
            n_cov = count_covering_visits(conn, objs)
        s = single.set_index("diaObjectId")
        drop = s.index[(n_cov.reindex(s.index).fillna(0) >= 2) & (s["max_det_per_night"].fillna(1) < 2)]
        n_single_dropped = len(drop)
        hit = hit[~hit["diaObjectId"].isin(drop)]
    cand_ids = hit["diaObjectId"].unique()
    cond_desc = [f"step1 positive_real & step2 in {s2_classes}"]
    if include_negative_galaxy:
        cond_desc[0] += " + (step1 negative_real & step2 galaxy)"
    if min_nights is not None and min_nights > 1:
        cond_desc.append(f"n_nights_total >= {min_nights}")
    if min_nights_no_host is not None and min_nights_no_host > 1:
        cond_desc.append(f"no_host: n_nights_total >= {min_nights_no_host}")
    cond_desc.append("single night: >= 2 detections if >= 2 visits cover it")
    print(f"[6] candidates: {len(cand_ids)} objects ({', '.join(cond_desc)}); "
          f"single-night: {len(single)} checked, {n_single_dropped} dropped")
    info["single_night_checked"] = int(len(single))
    info["single_night_dropped"] = int(n_single_dropped)

    # 7. 出力: 候補の一覧 (CSV) と、それに対応する 1 つの PDF
    # 順序: (positive_real, galaxy) -> (positive_real, no_host) -> (negative_real, galaxy) -> (positive_real, uncertain)
    order_map = {
        ("positive_real", "galaxy"): 0,
        ("positive_real", "no_host"): 1,
        ("negative_real", "galaxy"): 2,
        ("positive_real", "uncertain"): 3,
    }
    combo = list(zip(hit["s1_decision"], hit["decision"]))
    hit = hit.assign(_order=[order_map.get(c, 99) for c in combo],
                     _s1_prob=np.where(hit["s1_decision"] == "negative_real",
                                       hit["s1_prob_negative_real"],
                                       hit["s1_prob_positive_real"]))
    rep = hit.sort_values(["_order", "midpointMjdTai"], ascending=[True, False]) \
             .groupby("diaObjectId").head(1)
    rep = rep.sort_values(["_order", "_s1_prob"], ascending=[True, False]).set_index("diaObjectId")
    rows = []
    det_in = raw[raw["diaObjectId"].isin(cand_ids)].sort_values("midpointMjdTai")
    if per_object_csv:
        os.makedirs(os.path.join(out_dir, "objects"), exist_ok=True)
    for oid, r in rep.iterrows():
        d = det_in[det_in["diaObjectId"] == oid]
        latest = d.groupby("band").tail(1)
        row = {"diaObjectId": oid, "ra": float(d["ra"].mean()), "dec": float(d["dec"].mean()),
               "step2_decision": r["decision"], "decision_detail": r["decision_detail"],
               "host_score": r.get("host_score"), "band_scores": r.get("band_scores"), "prob_b": r.get("prob_b"),
               "s1_decision": r.get("s1_decision"),
               "s1_prob_positive_real": r["s1_prob_positive_real"],
               "s1_prob_negative_real": r.get("s1_prob_negative_real"),
               "s1_prob_bogus": r["s1_prob_bogus"],
               "rep_diaSourceId": int(r["diaSourceId"]), "rep_band": r["band"], "rep_mjd": r["midpointMjdTai"],
               "n_det_in_range": len(d), "n_nights_total": r.get("n_nights_total"),
               "max_det_per_night": r.get("max_det_per_night"),
               "n_visits_night": n_cov.get(oid, np.nan) if r.get("n_nights_total", 2) < 2 else np.nan,
               "first_det_mjd": r.get("first_det_mjd"), "last_det_mjd": r.get("last_det_mjd")}
        for b in BAND_ORDER:
            x = latest[latest["band"] == b]
            row[f"scienceFlux_{b}"] = float(x["scienceFlux"].iloc[0]) if len(x) else np.nan
            row[f"scienceFluxErr_{b}"] = float(x["scienceFluxErr"].iloc[0]) if len(x) else np.nan
            row[f"mjd_{b}"] = float(x["midpointMjdTai"].iloc[0]) if len(x) else np.nan
        rows.append(row)
        if per_object_csv:
            latest.assign(date_utc=Time(latest["midpointMjdTai"].values, format="mjd", scale="tai").utc.iso) \
                  .to_csv(os.path.join(out_dir, "objects", f"{oid}.csv"), index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(out_dir, "candidates.csv"), index=False)
    from .pfs_target import create_pfs_targets_df
    pfs_df = create_pfs_targets_df(summary, semester=pfs_semester, exptime=5400, resolution="L",
                                   reference_arm="r", old_pfs_list=pfs_old_list)
    pfs_df.to_csv(os.path.join(out_dir, "rubin_targets.csv"), index=False, na_rep="")
    pdf_path = os.path.join(out_dir, "candidates.pdf")
    if len(summary):
        extra_note = ""
        if min_nights is not None and min_nights > 1:
            extra_note += f", nights>={min_nights}"
        if min_nights_no_host is not None and min_nights_no_host > 1:
            extra_note += f", no_host nights>={min_nights_no_host}"
        candidates_pdf(pdf_path, summary, fits_root,
                       title=f"SN candidates  {start} .. {end}  (db {db}, bundle {bundle_version}{extra_note})")
    info.update({"n_sources": int(len(src)), "n_candidates": int(len(summary)),
                 "candidate_decisions": summary["decision_detail"].value_counts().to_dict() if len(summary) else {},
                 "include_uncertain": include_uncertain, "min_nights": min_nights,
                 "min_nights_no_host": min_nights_no_host,
                 "include_negative_galaxy": include_negative_galaxy,
                 "pfs_semester": pfs_semester,
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "elapsed_sec": round(time.time() - t0)})
    with open(os.path.join(out_dir, "run_info.json"), "w") as f:
        json.dump(info, f, indent=2, ensure_ascii=False, default=str)
    print(f"[7] {len(summary)} candidates -> {out_dir}/candidates.csv, candidates.pdf, and rubin_targets.csv ({info['elapsed_sec']} s)")
    return summary

# ---------------------------------------------------------------------------
# 候補の PDF (candidates.csv と同じ順に、1 ページ rows x cols 天体)
# ---------------------------------------------------------------------------
BAND_ORDER = ["u", "g", "r", "i", "z", "y"]

def candidates_pdf(path: str, summary: pd.DataFrame, fits_root: str, title: str, rows: int = 8, cols: int = 10) -> None:
    import sys
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from astropy.time import Time
    from tqdm import tqdm
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from export_candidates_pdf import triplet

    per = rows * cols
    n_pages = (len(summary) + per - 1) // per
    with PdfPages(path) as pdf:
        for page in range(n_pages):
            chunk = summary.iloc[page * per:(page + 1) * per]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.0), squeeze=False)
            fig.suptitle(f"{title} | Page {page + 1}/{n_pages}\nLayout: Sci | Tmp | Dif  (order = candidates.csv)",
                         fontsize=14, fontweight="bold")
            flat = axes.flatten()
            for i, r in enumerate(tqdm(list(chunk.itertuples()), desc=f"PDF page {page + 1}/{n_pages}", leave=False)):
                ax = flat[i]
                ax.imshow(triplet(r.diaObjectId, r.rep_mjd, r.rep_band, r.rep_diaSourceId, fits_root),
                          cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                date = Time(r.rep_mjd, format="mjd", scale="tai").utc.iso[5:10]
                if str(r.decision_detail).startswith("A:"):
                    line3 = f"host score {r.host_score:.2f}"
                else:
                    line3 = f"B: P={r.prob_b:.2f}" if pd.notna(r.prob_b) else "B"
                s1_dec = getattr(r, "s1_decision", "positive_real")
                if s1_dec == "negative_real":
                    s1_lbl = f"-real:{getattr(r, 's1_prob_negative_real', 0.0):.2f}"
                else:
                    s1_lbl = f"real:{r.s1_prob_positive_real:.2f}"
                ax.set_title(f"{r.diaObjectId} | {r.rep_band} | {date} | {r.decision_detail}\n"
                             f"{s1_lbl} bogus:{r.s1_prob_bogus:.2f}\n{line3}", fontsize=7)
                ax.axis("off")
            for j in range(len(chunk), len(flat)):
                flat[j].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)
