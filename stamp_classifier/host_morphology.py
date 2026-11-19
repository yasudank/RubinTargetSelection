"""
Host Morphology Features for Step 2 (point-like vs extended)
Science / Template カットアウトの母天体が、点源 (PSF と同じ形) か広がっているかを、
カットアウトに同梱された PSF 画像 (PSFIMAGE) と比較して測る。観測日ごとのシーイングの違いは、
各画像自身の PSF と比べることで打ち消す。

各画像 (tmp_ / sci_) について:
  psf_fwhm_arcsec   : PSF の FWHM (2 次モーメントから)
  snr               : 開口内の S/N (UNCERT を使用)
  ext_ratio         : (開口フラックス / PSF フラックス) を PSF 自身の同じ比で割ったもの。点源で ~1、広がると > 1
                      (SDSS の psfMag - cModelMag に相当する考え方)
  size_ratio        : ガウス重み付き 2 次モーメントの大きさ T = Ixx + Iyy の、天体 / PSF の比。点源で ~1
  ellipticity       : 天体の楕円率 (ガウス重み付きモーメント)
  conc_ratio        : 集中度 F(r < 1 FWHM) / F(r < 2.5 FWHM) を PSF の同じ比で割ったもの。点源で ~1、広がると < 1
  psf_chi2          : PSF だけのモデル (振幅のみ) の、r < 2.5 FWHM での換算 χ² (log10)
  offset_fwhm       : 母天体の重心とカットアウト中心 (= 突発天体の位置) の距離 (PSF FWHM 単位)
最後に tmp と sci の ext_ratio の差 (sci - tmp) を加える (突発天体の光の寄与の目安)。
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy import ndimage

PIXEL_SCALE = 0.2   # arcsec / pixel (LSST)
SIZE = 30

_PER_IMAGE = ["psf_fwhm_arcsec", "snr", "ext_ratio", "size_ratio", "ellipticity", "conc_ratio", "psf_chi2", "offset_fwhm",
              "central_host", "center_snr", "brightest_snr", "brightest_dist_fwhm"]
# 特徴量の版 (キャッシュのファイル名に入れる)。2: 中心の母天体の判定 (極大の条件) と文脈の特徴量を追加
HOST_FEATURE_VERSION = 2
# 「中心に母天体がある」: PSF で平滑化した Template の、中心から CENTRAL_RADIUS_FWHM 以内の最大点が、その内側で極大になっていること
CENTRAL_RADIUS_FWHM = 1.5
HOST_MORPH_FEATURE_NAMES = [f"tmp_{n}" for n in _PER_IMAGE] + [f"sci_{n}" for n in _PER_IMAGE] + ["ext_ratio_sci_minus_tmp"]
N_HOST = len(HOST_MORPH_FEATURE_NAMES)

_yy, _xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)

def _center_crop(a: np.ndarray, size: int = SIZE) -> np.ndarray:
    H, W = a.shape
    cy, cx = H // 2, W // 2
    h = size // 2
    out = np.zeros((size, size), dtype=float)
    y1, y2, x1, x2 = max(0, cy - h), min(H, cy + h), max(0, cx - h), min(W, cx + h)
    sub = a[y1:y2, x1:x2]
    sy, sx = (size - sub.shape[0]) // 2, (size - sub.shape[1]) // 2
    out[sy:sy + sub.shape[0], sx:sx + sub.shape[1]] = sub
    return out

def _read(path: str) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """(image, sigma, psf) を返す。PSF は和 1 に規格化し、ピークをカットアウト中心に合わせる"""
    try:
        with fits.open(path) as h:
            img = np.nan_to_num(np.asarray(h[0].data, dtype=float))
            unc = np.asarray(h["UNCERT"].data, dtype=float) if "UNCERT" in h and h["UNCERT"].data is not None \
                else (np.asarray(h[1].data, dtype=float) if len(h) > 1 and h[1].data is not None else None)
            psf = np.asarray(h["PSFIMAGE"].data, dtype=float) if "PSFIMAGE" in h and h["PSFIMAGE"].data is not None \
                else (np.asarray(h[2].data, dtype=float) if len(h) > 2 and h[2].data is not None else None)
    except Exception:
        return None
    if img.ndim != 2 or psf is None or psf.ndim != 2:
        return None
    img = _center_crop(img)
    if unc is None or unc.ndim != 2:
        sig = np.full_like(img, 1.4826 * np.median(np.abs(img - np.median(img))) + 1e-12)
    else:
        # UNCERT が分散の場合と標準偏差の場合がある。中央値をノイズの実測と比べて判定する
        unc = _center_crop(np.nan_to_num(unc, nan=np.inf, posinf=np.inf))
        mad = 1.4826 * np.median(np.abs(img - np.median(img)))
        med_u = np.median(unc[np.isfinite(unc)]) if np.isfinite(unc).any() else 0
        sig = np.sqrt(unc) if (med_u > 0 and abs(np.log(np.sqrt(med_u) / max(mad, 1e-12))) < abs(np.log(med_u / max(mad, 1e-12)))) else unc
        sig = np.where(np.isfinite(sig) & (sig > 0), sig, np.inf)
    psf = np.nan_to_num(psf)
    pk = np.unravel_index(np.argmax(psf), psf.shape)
    psf = ndimage.shift(psf, (psf.shape[0] // 2 - pk[0], psf.shape[1] // 2 - pk[1]), order=1, mode="constant")
    psf = _center_crop(psf)
    s = psf.sum()
    if s <= 0:
        return None
    return img, sig, psf / s

def _weighted_moments(img, cx, cy, sw):
    w = np.exp(-0.5 * ((_xx - cx) ** 2 + (_yy - cy) ** 2) / sw ** 2)
    f = img * w
    tot = f.sum()
    if tot <= 0:
        return np.nan, np.nan, np.nan
    ixx = (f * (_xx - cx) ** 2).sum() / tot
    iyy = (f * (_yy - cy) ** 2).sum() / tot
    ixy = (f * (_xx - cx) * (_yy - cy)).sum() / tot
    return ixx, iyy, ixy

def _measure(img, sig, psf) -> Dict[str, float]:
    out = {n: np.nan for n in _PER_IMAGE}
    bkg = np.median(img[((_xx - 14.5) ** 2 + (_yy - 14.5) ** 2) > 12 ** 2])
    img = img - bkg
    # PSF の大きさ
    pxx, pyy, _ = _weighted_moments(psf, 15.0, 15.0, 3.0)
    p_sigma = np.sqrt(max((pxx + pyy) / 2, 0.25))
    fwhm = 2.355 * p_sigma                                   # pixel
    out["psf_fwhm_arcsec"] = fwhm * PIXEL_SCALE
    out["central_host"] = 0.0
    # PSF を使ったマッチドフィルタの S/N マップ (各ピクセルに点源があるとしたときの S/N)
    k = psf[15 - 7:15 + 8, 15 - 7:15 + 8]
    var = np.where(np.isfinite(sig), sig ** 2, 1e30)
    num = ndimage.convolve(img, k[::-1, ::-1], mode="constant")
    den_map = np.sqrt(ndimage.convolve(var, (k ** 2)[::-1, ::-1], mode="constant"))
    snr_map = np.where(den_map > 0, num / den_map, 0.0)
    rr = np.hypot(_xx - 14.5, _yy - 14.5)
    inner = (_xx >= 2) & (_xx <= SIZE - 3) & (_yy >= 2) & (_yy <= SIZE - 3)
    out["center_snr"] = float(np.max(snr_map[rr <= fwhm]))
    by, bx = np.unravel_index(np.argmax(np.where(inner, snr_map, -np.inf)), snr_map.shape)
    out["brightest_snr"] = float(snr_map[by, bx])
    out["brightest_dist_fwhm"] = float(rr[by, bx] / fwhm)
    # 母天体の位置: 中心から CENTRAL_RADIUS_FWHM 以内の (平滑化した) 最大点。
    # それが範囲の内側で極大になっていなければ (隣の天体の裾などが範囲の端にかかっているだけなら)、中心に母天体なしとする
    sm = ndimage.gaussian_filter(img, p_sigma)
    r_c = rr <= max(CENTRAL_RADIUS_FWHM * fwhm, 3.0)
    if not np.any(sm[r_c] > 0):
        return out
    py, px = np.unravel_index(np.argmax(np.where(r_c, sm, -np.inf)), sm.shape)
    nb = sm[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2]
    if sm[py, px] < nb.max():
        return out
    out["central_host"] = 1.0
    win = ((_xx - px) ** 2 + (_yy - py) ** 2) <= (1.5 * p_sigma) ** 2
    wv = np.clip(img, 0, None) * win
    if wv.sum() <= 0:
        return out
    cx, cy = (wv * _xx).sum() / wv.sum(), (wv * _yy).sum() / wv.sum()
    out["offset_fwhm"] = np.hypot(cx - 14.5, cy - 14.5) / fwhm
    # PSF を母天体の位置に移す
    psf_c = ndimage.shift(psf, (cy - 15.0, cx - 15.0), order=1, mode="constant")
    r2 = (_xx - cx) ** 2 + (_yy - cy) ** 2
    ap = r2 <= (2.5 * fwhm) ** 2
    core = r2 <= fwhm ** 2
    w = np.where(np.isfinite(sig), 1.0 / sig ** 2, 0.0) * ap
    # PSF フラックス (振幅の最小二乗) と開口フラックス
    den = (psf_c ** 2 * w).sum()
    if den <= 0:
        return out
    a_psf = (img * psf_c * w).sum() / den
    f_ap = img[ap].sum()
    var_ap = np.sum(np.where(ap & np.isfinite(sig), sig ** 2, 0.0))
    out["snr"] = f_ap / np.sqrt(var_ap) if var_ap > 0 else np.nan
    if a_psf <= 0 or f_ap <= 0:
        return out
    c_psf = psf_c[ap].sum()                                   # PSF 自身の開口内の割合 (PSF フラックス 1 に対して)
    out["ext_ratio"] = (f_ap / a_psf) / c_psf
    out["conc_ratio"] = (img[core].sum() / f_ap) / (psf_c[core].sum() / c_psf)
    resid = (img - a_psf * psf_c) ** 2 * w
    n = ap.sum() - 1
    out["psf_chi2"] = np.log10(max(resid.sum() / max(n, 1), 1e-6))
    sw = 1.5 * p_sigma
    oxx, oyy, oxy = _weighted_moments(img, cx, cy, sw)
    qxx, qyy, _ = _weighted_moments(psf_c, cx, cy, sw)
    if np.isfinite(oxx) and (qxx + qyy) > 0:
        out["size_ratio"] = (oxx + oyy) / (qxx + qyy)
        T = oxx + oyy
        out["ellipticity"] = np.hypot(oxx - oyy, 2 * oxy) / T if T > 0 else np.nan
    return out

def host_features_from_paths(sci_path: str, tmp_path: str) -> np.ndarray:
    vals = {}
    for prefix, path in [("tmp_", tmp_path), ("sci_", sci_path)]:
        r = _read(path)
        m = _measure(*r) if r is not None else {n: np.nan for n in _PER_IMAGE}
        vals.update({prefix + k: v for k, v in m.items()})
    vals["ext_ratio_sci_minus_tmp"] = vals["sci_ext_ratio"] - vals["tmp_ext_ratio"]
    return np.array([vals[n] for n in HOST_MORPH_FEATURE_NAMES], dtype=np.float32)

def host_features_batch(records: List[Dict], fits_root: str, workers: int = 16, desc: str = "Host morphology") -> np.ndarray:
    """records: diaSourceId / diaObjectId / midpointMjdTai / band を持つ辞書のリスト"""
    from tqdm import tqdm
    from .dataset import get_cutout_path

    def one(rec):
        p = lambda t: get_cutout_path(fits_root, rec["diaObjectId"], rec["midpointMjdTai"], rec["band"], rec["diaSourceId"], t)
        try:
            return host_features_from_paths(p("science"), p("template"))
        except Exception:
            return np.full(N_HOST, np.nan, dtype=np.float32)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(tqdm(ex.map(one, records), total=len(records), desc=desc))
    return np.vstack(rows).astype(np.float32)

def clean_for_rf(feats: np.ndarray) -> np.ndarray:
    """RF 用に NaN / inf を番兵値 (-999) に置き換える"""
    return np.where(np.isfinite(feats), feats, -999.0).astype(np.float32)

def host_cache_path(source_ids) -> str:
    """step2 の入力ソース集合ごとのキャッシュファイル名 (ソース一覧のハッシュ付き)"""
    import hashlib
    from .config import DATA_DIR
    h = hashlib.md5(np.asarray(sorted(np.asarray(source_ids, dtype=np.int64)), dtype=np.int64).tobytes()).hexdigest()[:10]
    return os.path.join(DATA_DIR, "cache", f"host_morph_step2_v{HOST_FEATURE_VERSION}_N{len(source_ids)}_{h}.npz")

def compute_host_morph_cache(df_sources, fits_root: str, refresh: bool = False) -> Dict[str, np.ndarray]:
    """
    df_sources: diaSourceId / diaObjectId / midpointMjdTai / band を持つ DataFrame。
    全ソースの母天体の形態特徴を計算してキャッシュする (既にあれば読み込む)。
    """
    path = host_cache_path(df_sources["diaSourceId"].values)
    if os.path.exists(path) and not refresh:
        d = np.load(path, allow_pickle=False)
        print(f"[*] Using cached host morphology: {path}")
        return {"diaSourceIds": d["diaSourceIds"], "features": d["features"], "names": list(d["names"])}
    if not refresh:
        # 同じ版のキャッシュに、求めるソースがすべて含まれていれば、その部分を使う (step2 の入力が絞られた場合など)
        import glob
        want = df_sources["diaSourceId"].values.astype(np.int64)
        for cand in sorted(glob.glob(os.path.join(os.path.dirname(path), f"host_morph_step2_v{HOST_FEATURE_VERSION}_N*.npz"))):
            d = np.load(cand, allow_pickle=False)
            idx = {int(s): i for i, s in enumerate(d["diaSourceIds"])}
            if all(int(s) in idx for s in want):
                rows = [idx[int(s)] for s in want]
                print(f"[*] Using subset of cached host morphology: {cand}")
                return {"diaSourceIds": want, "features": d["features"][rows], "names": list(d["names"])}
    recs = df_sources[["diaSourceId", "diaObjectId", "midpointMjdTai", "band"]].to_dict("records")
    feats = host_features_batch(recs, fits_root)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, diaSourceIds=df_sources["diaSourceId"].values.astype(np.int64), features=feats,
                        names=np.array(HOST_MORPH_FEATURE_NAMES))
    print(f"[+] Host morphology of {len(feats)} sources -> {path}")
    return {"diaSourceIds": df_sources["diaSourceId"].values.astype(np.int64), "features": feats,
            "names": HOST_MORPH_FEATURE_NAMES}

# ---------------------------------------------------------------------------
# 点源 / 広がりの判定 (方針 A): 基準の恒星の分布から S/N ごとの閾値を決める
# ---------------------------------------------------------------------------
# 基準の恒星: SIMBAD で恒星と分類されたもの、または Gaia DR3 の視差の S/N > 5 のもの
STAR_OTYPES = {"*", "EB*", "RR*", "HB*", "LM*", "WD*", "BS*", "LP*", "V*", "Ce*", "dS*", "Ro*", "PM*", "HV*", "SB*",
               "RG*", "s*b", "C*", "Mi*", "Pe*", "RS*", "BY*", "gD*", "sg*", "Er*", "Ir*", "Or*", "El*", "a2*", "WR*",
               "Be*", "pr*", "**", "TT*", "Y*O", "HS*", "LP?", "RR?", "EB?"}
SNR_EDGES = [10, 20, 50, 100, 200, 500, float("inf")]
MIN_SNR = 10.0            # Template の S/N がこれ未満なら判定しない (low_snr)
MIN_STARS_PER_BIN = 200   # 恒星がこれより少ない S/N の区間は、隣の区間とまとめて閾値を決める

def reference_star_mask(simbad_otype, plx, e_plx) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = np.asarray(plx, dtype=float) / np.asarray(e_plx, dtype=float)
    return np.isin(np.asarray(simbad_otype, dtype=object), list(STAR_OTYPES)) | (snr > 5)

def derive_thresholds(tmp_snr: np.ndarray, tmp_size_ratio: np.ndarray, is_star: np.ndarray,
                      lo_pct: float = 95.0, hi_pct: float = 99.0) -> Dict:
    """
    基準の恒星の Template の大きさの比 (PSF に対する比) の分布から、S/N の区間ごとに
      point の上限 = lo_pct パーセンタイル, extended の下限 = hi_pct パーセンタイル
    を決める。恒星の少ない区間は、恒星が MIN_STARS_PER_BIN 以上になるまで隣の区間を加える。
    """
    ok = is_star & np.isfinite(tmp_size_ratio) & np.isfinite(tmp_snr)
    bins = []
    for k in range(len(SNR_EDGES) - 1):
        lo_e, hi_e = SNR_EDGES[k], SNR_EDGES[k + 1]
        a, b = k, k + 1
        sel = ok & (tmp_snr >= SNR_EDGES[a]) & (tmp_snr < SNR_EDGES[b])
        while sel.sum() < MIN_STARS_PER_BIN and (a > 0 or b < len(SNR_EDGES) - 1):
            if b < len(SNR_EDGES) - 1:
                b += 1
            else:
                a -= 1
            sel = ok & (tmp_snr >= SNR_EDGES[a]) & (tmp_snr < SNR_EDGES[b])
        s = tmp_size_ratio[sel]
        bins.append({"snr_min": lo_e, "snr_max": hi_e, "n_stars": int(sel.sum()),
                     "point_max": float(np.percentile(s, lo_pct)), "extended_min": float(np.percentile(s, hi_pct))})
    return {"feature": "tmp_size_ratio", "lo_pct": lo_pct, "hi_pct": hi_pct, "min_snr": MIN_SNR,
            "reference": "SIMBAD stellar otypes or Gaia DR3 parallax S/N > 5", "bins": bins}

def classify_host(tmp_snr: np.ndarray, tmp_size_ratio: np.ndarray, thr: Dict) -> np.ndarray:
    """'point' / 'extended' / 'uncertain' (2 つの閾値の間) / 'low_snr' (S/N 不足または測定不能)"""
    out = np.full(len(tmp_snr), "low_snr", dtype=object)
    for b in thr["bins"]:
        m = (tmp_snr >= max(b["snr_min"], thr["min_snr"])) & (tmp_snr < b["snr_max"]) & np.isfinite(tmp_size_ratio)
        out[m & (tmp_size_ratio <= b["point_max"])] = "point"
        out[m & (tmp_size_ratio > b["extended_min"])] = "extended"
        out[m & (tmp_size_ratio > b["point_max"]) & (tmp_size_ratio <= b["extended_min"])] = "uncertain"
    return out

# step2 の最終判定: RF が no_host / bogus と判定したものはそのまま。それ以外は Template の形で決める
HOST_TO_CLASS = {"point": "star", "extended": "galaxy"}

def combine_with_rf(rf_decision: np.ndarray, host_morph: np.ndarray) -> np.ndarray:
    rf_decision = np.asarray(rf_decision, dtype=object)
    out = np.array([HOST_TO_CLASS.get(h, "uncertain") for h in host_morph], dtype=object)
    keep = np.isin(rf_decision, ["no_host", "bogus"])
    out[keep] = rf_decision[keep]
    return out

def source_extendedness_score(tmp_snr: np.ndarray, tmp_size_ratio: np.ndarray, thr: Dict) -> np.ndarray:
    """
    観測ごとの大きさの比を、その S/N での閾値で規格化した値:
      0 = 点源の上限 (恒星の 95 パーセンタイル), 1 = 広がりの下限 (恒星の 99 パーセンタイル)。
    S/N が MIN_SNR 未満、または測定できないものは NaN。
    """
    s = np.full(len(tmp_snr), np.nan)
    for b in thr["bins"]:
        m = (tmp_snr >= max(b["snr_min"], thr["min_snr"])) & (tmp_snr < b["snr_max"]) & np.isfinite(tmp_size_ratio)
        s[m] = (tmp_size_ratio[m] - b["point_max"]) / (b["extended_min"] - b["point_max"])
    return s

def object_host_morph(df, thr: Dict):
    """
    天体単位の判定。df: diaObjectId / band / tmp_snr / tmp_size_ratio を持つ (観測ごと)。
    1) 観測ごとの規格化した値 (source_extendedness_score) のバンドごとの中央値と、S/N の中央値をとる
       (Template はバンドごとに同じコアドなので、観測数ではなく再サンプリングのばらつきを抑えるため)
    2) バンドの値を、そのバンドの S/N の中央値で重み付けして平均する (host_score)
    3) host_score >= 1 なら extended, <= 0 なら point, その間は uncertain。使えるバンドが無ければ low_snr
    返り値: diaObjectId ごとの DataFrame (host_score, host_morph_obj, n_bands_used, band_scores)
    """
    import pandas as pd
    d = df[["diaObjectId", "band", "tmp_snr", "tmp_size_ratio"]].copy()
    d["score"] = source_extendedness_score(d["tmp_snr"].values, d["tmp_size_ratio"].values, thr)
    d = d[np.isfinite(d["score"])]
    pb = d.groupby(["diaObjectId", "band"]).agg(score=("score", "median"), snr=("tmp_snr", "median")).reset_index()
    pb["w"] = pb["snr"]
    pb["ws"] = pb["w"] * pb["score"]
    g = pb.groupby("diaObjectId")
    out = pd.DataFrame({"host_score": g["ws"].sum() / g["w"].sum(), "n_bands_used": g.size()})
    out["band_scores"] = g.apply(lambda x: ";".join(f"{b}:{s:.2f}" for b, s in zip(x["band"], x["score"])),
                                 include_groups=False)
    out["host_morph_obj"] = np.where(out["host_score"] >= 1, "extended",
                                     np.where(out["host_score"] <= 0, "point", "uncertain"))
    out = out.reset_index()
    allobj = pd.DataFrame({"diaObjectId": df["diaObjectId"].unique()})
    out = allobj.merge(out, on="diaObjectId", how="left")
    out["host_morph_obj"] = out["host_morph_obj"].fillna("low_snr")
    out["n_bands_used"] = out["n_bands_used"].fillna(0).astype(int)
    return out
