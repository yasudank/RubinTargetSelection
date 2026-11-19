"""
Subaru/PFS Target Uploader 向けのターゲットリスト作成およびリスト突合モジュール。
User Guide: https://pfs-etc.naoj.hawaii.edu/uploader/doc/inputs.html

必須・標準カラム:
  ob_code       : rubin_{semester}_XXXX (XXXX は 4 桁以上の通し番号)
  obj_id        : diaObjectId (64-bit int)
  ra            : 赤経 (deg)
  dec           : 赤緯 (deg)
  exptime       : 露光時間 (秒, 既定 5400)
  priority      : 優先度 (0, 1, 2)
                  - 0: (positive_real, galaxy), (positive_real, no_host)
                  - 1: (negative_real, galaxy)
                  - 2: (positive_real, uncertain)
  resolution    : "L" (Low resolution)
  reference_arm : "r" (Red arm)
  flux カラム   : g_hsc, r2_hsc, i2_hsc, z_hsc, y_hsc
                  情報のあるバンドごとに最新 diaSource の scienceFlux (nJy) を設定
"""

import os
import re
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd

HSC_BAND_MAP = {
    "g": "g_hsc",
    "r": "r2_hsc",
    "i": "i2_hsc",
    "z": "z_hsc",
    "y": "y_hsc",
}

def compute_priority(s1_decision: str, s2_decision: str, n_nights_total=None) -> int:
    """
    (s1_decision, step2_decision) から PFS の priority (0, 1, 2) を計算する。
    0: (positive_real, galaxy), (positive_real, no_host)
    1: (negative_real, galaxy)
    2: (positive_real, uncertain)
    その他: 2 (フォールバック)
    n_nights_total (検出のある夜の数) が 2 未満の天体は判定によらず 2
    (2 つ以上の別の夜に検出されたものだけを 0, 1 にする)。値がない (NaN/None) ときは従来どおり。
    """
    try:
        if n_nights_total is not None and not pd.isna(n_nights_total) and int(n_nights_total) < 2:
            return 2
    except (TypeError, ValueError):
        pass
    if s1_decision == "positive_real" and s2_decision in ("galaxy", "no_host"):
        return 0
    elif s1_decision == "negative_real" and s2_decision == "galaxy":
        return 1
    elif s1_decision == "positive_real" and s2_decision == "uncertain":
        return 2
    return 2

def parse_max_seq_from_codes(codes: List[str], semester: str) -> int:
    """既存の ob_code リストから、rubin_{semester}_XXXX の最大通し番号を取得する"""
    pattern = re.compile(rf"^rubin_{re.escape(semester)}_(\d+)$")
    seqs = []
    for c in codes:
        if pd.isna(c):
            continue
        m = pattern.match(str(c).strip())
        if m:
            try:
                seqs.append(int(m.group(1)))
            except ValueError:
                pass
    return max(seqs) if seqs else 0

def create_pfs_targets_df(
    candidates_df: pd.DataFrame,
    semester: str = "s26b",
    exptime: int = 5400,
    resolution: str = "L",
    reference_arm: str = "r",
    old_pfs_list: Optional[Union[str, pd.DataFrame]] = None,
    all_bands: bool = False,
    include_errors: bool = False,
) -> pd.DataFrame:
    """
    candidates.csv の DataFrame から PFS Target Uploader 形式の DataFrame を生成する。

    Parameters
    ----------
    candidates_df : pd.DataFrame
        candidates.csv の内容
    semester : str
        学期コード (既定: 's26b')
    exptime : int
        露光時間 (既定: 5400 秒)
    resolution : str
        解像度モード (既定: 'L')
    reference_arm : str
        リファレンスアーム (既定: 'r')
    old_pfs_list : str or pd.DataFrame, optional
        過去の PFS ターゲットリスト。指定された場合、既存の obj_id は同じ ob_code を維持し、
        新規天体には最大通し番号以降の新しい番号を割り当てる。
    all_bands : bool
        True の場合、データがないバンドも含め全 5 バンド (g_hsc, r2_hsc, i2_hsc, z_hsc, y_hsc)
        の列を出力する。既定 (False) は候補中に測定値が存在するバンドのみ列を作成する。
    include_errors : bool
        True の場合、対応する *_error 列も出力する (scienceFluxErr から取得)。
    """
    if len(candidates_df) == 0:
        base_cols = ["ob_code", "obj_id", "ra", "dec", "exptime", "priority", "resolution", "reference_arm"]
        return pd.DataFrame(columns=base_cols)

    df = candidates_df.copy()

    # 1. 過去リストの読み込みと ob_code マッピング
    existing_map: Dict[int, str] = {}
    current_max_seq = 0
    if old_pfs_list is not None:
        if isinstance(old_pfs_list, str):
            if os.path.exists(old_pfs_list):
                old_df = pd.read_csv(old_pfs_list)
            else:
                old_df = pd.DataFrame()
        else:
            old_df = old_pfs_list.copy()

        if len(old_df) and "obj_id" in old_df.columns and "ob_code" in old_df.columns:
            for _, r in old_df.iterrows():
                try:
                    oid = int(r["obj_id"])
                    existing_map[oid] = str(r["ob_code"]).strip()
                except (ValueError, TypeError):
                    continue
            current_max_seq = parse_max_seq_from_codes(old_df["ob_code"].tolist(), semester)

    # 2. ob_code の割り当て
    ob_codes = []
    next_seq = current_max_seq
    for _, r in df.iterrows():
        oid = int(r["diaObjectId"])
        if oid in existing_map:
            ob_codes.append(existing_map[oid])
        else:
            next_seq += 1
            ob_codes.append(f"rubin_{semester}_{next_seq:04d}")

    # 3. 基本属性
    obj_ids = [int(oid) for oid in df["diaObjectId"]]
    ras = df["ra"].astype(float).values
    decs = df["dec"].astype(float).values

    priorities = [
        compute_priority(r.get("s1_decision", "positive_real"), r.get("step2_decision", "galaxy"),
                         r.get("n_nights_total"))
        for _, r in df.iterrows()
    ]

    out_dict = {
        "ob_code": ob_codes,
        "obj_id": obj_ids,
        "ra": ras,
        "dec": decs,
        "exptime": [int(exptime)] * len(df),
        "priority": priorities,
        "resolution": [str(resolution)] * len(df),
        "reference_arm": [str(reference_arm)] * len(df),
    }

    # 4. flux カラムの構築 (g_hsc, r2_hsc, i2_hsc, z_hsc, y_hsc)
    for b, col_name in HSC_BAND_MAP.items():
        src_col = f"scienceFlux_{b}"
        has_data = src_col in df.columns and df[src_col].notna().any()
        if all_bands or has_data:
            if src_col in df.columns:
                out_dict[col_name] = df[src_col].values
            else:
                out_dict[col_name] = [np.nan] * len(df)

            if include_errors:
                err_src = f"scienceFluxErr_{b}"
                err_col = f"{col_name}_error"
                if err_src in df.columns:
                    out_dict[err_col] = df[err_src].values
                else:
                    out_dict[err_col] = [np.nan] * len(df)

    pfs_df = pd.DataFrame(out_dict)
    return pfs_df

def reconcile_pfs_target_lists(
    old_pfs_path: str,
    new_pfs_path: str,
    out_path: str,
    semester: str = "s26b",
    mode: str = "update_new",
    exptime: Optional[int] = None,
    resolution: Optional[str] = None,
    reference_arm: Optional[str] = None,
    make_pdf: bool = False,
    fits_root: str = "cutouts",
) -> pd.DataFrame:
    """
    古い PFS ターゲットリストと新しいリストを入力し、古いリストの番号が変わらないように
    通し番号を調整してユニークにする。

    Parameters
    ----------
    old_pfs_path : str
        古い PFS ターゲットリスト CSV のパス
    new_pfs_path : str
        新しい PFS ターゲットリスト CSV のパス
    out_path : str
        調整後の CSV の出力先
    semester : str
        学期コード (既定: 's26b')
    mode : str
        'update_new' : 新しいリストの天体のみを出力 (重複天体は古い番号を維持、新規天体は新しい番号)
        'merge'      : 古いリストと新しいリストを統合して出力 (重複天体は新しいデータで更新)
    exptime : int, optional
        露光時間 (秒) を上書き設定する場合に指定
    resolution : str, optional
        解像度モード (L/M) を上書き設定する場合に指定
    reference_arm : str, optional
        リファレンスアーム (r/b/n/m) を上書き設定する場合に指定
    make_pdf : bool
        True の場合、出力 CSV に基づく 3 連画像 PDF (out_path の .pdf) も同時に作成する
    fits_root : str
        カットアウト FITS 画像のルートディレクトリ (既定: 'cutouts')
    """
    old_df = pd.read_csv(old_pfs_path)
    new_df = pd.read_csv(new_pfs_path)

    old_map: Dict[int, str] = {}
    for _, r in old_df.iterrows():
        try:
            oid = int(r["obj_id"])
            old_map[oid] = str(r["ob_code"]).strip()
        except (ValueError, TypeError):
            continue

    max_seq = parse_max_seq_from_codes(old_df["ob_code"].tolist(), semester)

    # new_df の ob_code を調整
    reconciled_codes = []
    next_seq = max_seq
    for _, r in new_df.iterrows():
        oid = int(r["obj_id"])
        if oid in old_map:
            reconciled_codes.append(old_map[oid])
        else:
            next_seq += 1
            reconciled_codes.append(f"rubin_{semester}_{next_seq:04d}")

    new_df["ob_code"] = reconciled_codes

    if mode == "merge":
        # old_df にしか存在しない天体を追加
        new_oids = set(new_df["obj_id"].astype(int))
        old_only = old_df[~old_df["obj_id"].astype(int).isin(new_oids)].copy()
        combined = pd.concat([new_df, old_only], ignore_index=True)
        # ob_code の通し番号順にソート
        pattern = re.compile(rf"^rubin_{re.escape(semester)}_(\d+)$")
        combined["_seq"] = combined["ob_code"].map(lambda c: int(pattern.match(str(c)).group(1)) if pattern.match(str(c)) else 999999)
        combined = combined.sort_values("_seq").drop(columns=["_seq"]).reset_index(drop=True)
        res = combined
    else:
        res = new_df

    if exptime is not None:
        res["exptime"] = int(exptime)
    if resolution is not None:
        res["resolution"] = str(resolution)
    if reference_arm is not None:
        res["reference_arm"] = str(reference_arm)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    res.to_csv(out_path, index=False, na_rep="")

    if make_pdf:
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        export_pfs_targets_pdf(
            targets_csv_or_df=res,
            out_pdf=pdf_path,
            candidates_csvs=[
                os.path.join(os.path.dirname(old_pfs_path), "candidates.csv"),
                os.path.join(os.path.dirname(new_pfs_path), "candidates.csv"),
            ],
            fits_root=fits_root,
        )

    return res


def export_pfs_targets_pdf(
    targets_csv_or_df: Union[str, pd.DataFrame],
    out_pdf: Optional[str] = None,
    candidates_csvs: Optional[List[str]] = None,
    fits_root: str = "cutouts",
    title: Optional[str] = None,
    rows: int = 8,
    cols: int = 10,
    dpi: int = 120,
) -> str:
    """
    Subaru/PFS ターゲットリスト (rubin_targets.csv) を元に、
    ob_code 順に並べた 3 連画像 (Science | Template | Difference) の PDF を作成する。
    """
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from astropy.time import Time
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from export_candidates_pdf import triplet

    if isinstance(targets_csv_or_df, str):
        targets_df = pd.read_csv(targets_csv_or_df)
        base_dir = os.path.dirname(os.path.abspath(targets_csv_or_df))
        if out_pdf is None:
            out_pdf = os.path.splitext(targets_csv_or_df)[0] + ".pdf"
    else:
        targets_df = targets_csv_or_df.copy()
        base_dir = "."
        if out_pdf is None:
            out_pdf = "rubin_targets.pdf"

    # candidates.csv の探索・収集
    cand_files: List[str] = []
    if candidates_csvs:
        for c in candidates_csvs:
            if c and os.path.exists(c):
                cand_files.append(c)

    # 追加でローカルや reports ディレクトリ内の candidates.csv も検索
    extra_cands = []
    if os.path.exists(os.path.join(base_dir, "candidates.csv")):
        extra_cands.append(os.path.join(base_dir, "candidates.csv"))
    extra_cands.extend(sorted(glob.glob("reports/**/candidates.csv", recursive=True)))

    seen = {os.path.abspath(c) for c in cand_files}
    for c in extra_cands:
        abs_c = os.path.abspath(c)
        if abs_c not in seen and os.path.exists(c):
            seen.add(abs_c)
            cand_files.append(c)

    if cand_files:
        dfs = []
        for f in cand_files:
            try:
                dfs.append(pd.read_csv(f))
            except Exception:
                pass
        if dfs:
            all_cands = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["diaObjectId"], keep="last")
            merged = targets_df.merge(all_cands, left_on="obj_id", right_on="diaObjectId", how="left")
        else:
            merged = targets_df.copy()
    else:
        merged = targets_df.copy()

    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)

    if title is None:
        title = f"Rubin PFS Targets ({len(merged)} objects in ob_code order)"

    per = rows * cols
    n_pages = max(1, (len(merged) + per - 1) // per)

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            chunk = merged.iloc[page * per:(page + 1) * per]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.1), squeeze=False)
            fig.suptitle(f"{title} | Page {page + 1}/{n_pages}\nLayout: Sci | Tmp | Dif  (order = ob_code)",
                         fontsize=13, fontweight="bold")
            flat = axes.flatten()
            for i, r in enumerate(chunk.itertuples()):
                ax = flat[i]
                oid = getattr(r, "obj_id", getattr(r, "diaObjectId", None))
                ob = getattr(r, "ob_code", "N/A")
                prio = getattr(r, "priority", "-")
                mjd = getattr(r, "rep_mjd", None)
                band = getattr(r, "rep_band", "")
                sid = getattr(r, "rep_diaSourceId", None)

                if pd.notna(mjd) and pd.notna(sid) and oid:
                    img = triplet(int(oid), float(mjd), str(band), int(sid), fits_root)
                    date = Time(float(mjd), format="mjd", scale="tai").utc.iso[5:10]
                else:
                    img = np.zeros((30, 90), dtype=np.uint8)
                    date = "-"

                ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

                # 3 つの各パネル (Sci, Tmp, Dif) の中心 (14.5, 14.5) にレティクルを描画
                for cx in [14.5, 44.5, 74.5]:
                    ax.add_patch(plt.Circle((cx, 14.5), 3.5, color="cyan", fill=False, linewidth=0.8, linestyle="--"))

                dec_detail = getattr(r, "decision_detail", "")
                if pd.isna(dec_detail) or not dec_detail:
                    dec_detail = getattr(r, "decision", "")
                s1_dec = getattr(r, "s1_decision", "positive_real")
                p_neg = getattr(r, "s1_prob_negative_real", 0.0)
                p_pos = getattr(r, "s1_prob_positive_real", 0.0)
                p_bog = getattr(r, "s1_prob_bogus", 0.0)
                if s1_dec == "negative_real":
                    s1_lbl = f"-real:{p_neg:.2f}"
                else:
                    s1_lbl = f"real:{p_pos:.2f}"

                ax.set_title(f"{ob} (P{prio}) | {band} | {date}\n"
                             f"oid:{oid}\n"
                             f"{s1_lbl} bog:{p_bog:.2f} | {dec_detail}", fontsize=6.2)
                ax.axis("off")
            for j in range(len(chunk), len(flat)):
                flat[j].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)

    return out_pdf


