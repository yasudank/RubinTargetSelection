"""
step1 / step2 の判定で選んだ天体の Science | Template | Difference 画像を並べた PDF を作成する。

選択条件 (既定): step1 の判定が positive_real、かつ同じソースの step2 の判定が galaxy または no_host で、
観測夜 (dayObs = floor(MJD - 0.5)) が指定期間内のもの。
--include_uncertain を付けると、step2 の判定が uncertain でも確率が最大のクラスが galaxy / no_host のソースも含める。
--include_host_morph uncertain,low_snr を付けると (step2 の CSV が host_morph の出力 *_hm.csv のとき)、
母天体の形の判定が uncertain / low_snr のために最終判定が uncertain になったソースも含める (超新星候補の完全性を上げる)。
天体ごとに 1 ソースを表示する (確かな判定のソースを優先し、その中の最新)。
並び順は、閾値を超えた判定 (galaxy → no_host)、RF の uncertain (galaxy 寄り → no_host 寄り)、
母天体の形が uncertain、low_snr の順。それぞれの中では P(galaxy) などの確率の降順。

例 (母天体の形の判定を使い、完全性を優先):
  bin/python export_candidates_pdf.py --step2_csv data/predictions_step2_s2_v1_hm.csv \
      --include_host_morph uncertain,low_snr --output_pdf reports/sn_candidates_hm_202605.pdf

例:
  bin/python export_candidates_pdf.py --start 2026-05-01 --end 2026-05-31 \
      --output_pdf reports/real_pos_galaxy_nohost_202605.pdf
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

from stamp_classifier.dataset import get_cutout_path

# 表示用のスケーリングは fink の従来の PDF と同じものを使う
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fink", "scripts"))
from survey_utils import apply_arcsinh_v2, apply_arcsinh_v2_diff  # noqa: E402

STAMP = 30

def read_fits(path):
    try:
        with fits.open(path) as h:
            data = h[0].data if h[0].data is not None else h[1].data
        if data is None or data.ndim != 2:
            return None
        return data.astype(np.float64)
    except Exception:
        return None

def triplet(oid, mjd, band, sid, fits_root):
    """Science | Template | Difference を横に並べた uint8 画像 (欠損は黒)"""
    panels = []
    for itype, scale in [("science", apply_arcsinh_v2), ("template", apply_arcsinh_v2),
                         ("difference", apply_arcsinh_v2_diff)]:
        d = read_fits(get_cutout_path(fits_root, oid, mjd, band, sid, itype))
        if d is None:
            img = np.zeros((STAMP, STAMP), dtype=np.uint8)
        else:
            scaled = scale(d)
            if scaled.shape == (STAMP, STAMP):
                img = scaled
            else:
                # 30x30 以外の画像は中心 (cy, cx) を基準に中心クロップ / パディング
                H, W = scaled.shape
                cy, cx = H // 2, W // 2
                half = STAMP // 2
                src_y0 = max(0, cy - half)
                src_y1 = min(H, cy + (STAMP - half))
                src_x0 = max(0, cx - half)
                src_x1 = min(W, cx + (STAMP - half))
                ch = src_y1 - src_y0
                cw = src_x1 - src_x0
                dst_y0 = (STAMP - ch) // 2
                dst_x0 = (STAMP - cw) // 2
                out = np.zeros((STAMP, STAMP), dtype=np.uint8)
                out[dst_y0:dst_y0 + ch, dst_x0:dst_x0 + cw] = scaled[src_y0:src_y1, src_x0:src_x1]
                img = out
        panels.append(img)
    return np.concatenate(panels, axis=1)

def mjd_of(date_str):
    return int(np.floor(Time(date_str, scale="utc").mjd))

def select_final(args, df):
    """
    最終判定ファイル (data/predictions_step2_final.csv) からの選択。
      step2 の判定が --s2_classes (既定 galaxy,no_host) のもの。
      --include_uncertain を付けると、判定が uncertain のもの (グループ A で形が判定できない / グループ B の real_uncertain /
      グループ B でまだモデルの判定が無いもの) も含める。
    天体ごとに 1 ソース (確かな判定を優先し、その中で最新)。
    並び順: 判定 (galaxy → no_host) → uncertain。それぞれ グループ A → B、その中で スコア (A: host_score, B: その判定の確率) の降順。
    """
    s2_classes = [c.strip() for c in args.s2_classes.split(",")]
    d0, d1 = mjd_of(args.start), mjd_of(args.end)
    base = (df["dayObs"] >= d0) & (df["dayObs"] <= d1) & (df["s1_decision"] == args.s1_class)
    decided = df["decision"].isin(s2_classes)
    unc = (df["decision"] == "uncertain") if args.include_uncertain else pd.Series(False, index=df.index)
    sel = df[base & (decided | unc)].copy()
    sel["tier"] = np.where(sel["decision"].isin(s2_classes), 0, 1)
    sel["is_uncertain"] = sel["tier"] > 0
    sel["s2_label"] = sel["decision_detail"]
    sel = sel.sort_values(["tier", "midpointMjdTai"], ascending=[False, True]).groupby("diaObjectId").tail(1)
    pb = np.full(len(sel), np.nan)
    if "class_b" in sel.columns:
        for c in ["no_host", "real_uncertain", "galaxy", "bogus"]:
            m = (sel["class_b"] == c).values
            if f"prob_b_{c}" in sel.columns:
                pb[m] = sel.loc[m, f"prob_b_{c}"].values
    sel["s2_score"] = np.where(sel["host_group"] == "A", sel["host_score"], pb)
    cls_order = {c: i for i, c in enumerate(s2_classes)}
    sel["class_order"] = sel["tier"] * 100 + sel["decision"].map(cls_order).fillna(len(s2_classes)) * 10 \
        + (sel["host_group"] == "B").astype(int)
    sel = sel.sort_values(["class_order", "s2_score"], ascending=[True, False]).reset_index(drop=True)
    sel["date_utc"] = Time(sel["midpointMjdTai"].values, format="mjd", scale="tai").utc.iso
    sel["date_utc"] = sel["date_utc"].str[:16]
    return sel, d0, d1

def select(args):
    s1 = pd.read_csv(args.step1_csv, usecols=["diaSourceId", "decision", "prob_positive_real",
                                              "prob_negative_real", "prob_bogus"])
    s1 = s1.rename(columns={"decision": "s1_decision", "prob_positive_real": "s1_prob_positive_real",
                            "prob_negative_real": "s1_prob_negative_real", "prob_bogus": "s1_prob_bogus"})
    s2 = pd.read_csv(args.step2_csv)
    df = s2.merge(s1, on="diaSourceId", how="inner")
    # 除外する visit のソースは選ばない (data/excluded_sources.csv; stamp_classifier の visit_quality が作成)
    ex_file = os.path.join("data", "excluded_sources.csv")
    if os.path.exists(ex_file):
        ex = pd.read_csv(ex_file, usecols=["diaSourceId"])["diaSourceId"]
        n0 = len(df)
        df = df[~df["diaSourceId"].isin(ex)]
        if len(df) < n0:
            print(f"[*] Dropped {n0 - len(df)} sources from excluded visits ({ex_file}).")
    df["dayObs"] = np.floor(df["midpointMjdTai"] - 0.5).astype(int)
    if "decision_detail" in df.columns:
        return select_final(args, df)
    d0, d1 = mjd_of(args.start), mjd_of(args.end)
    s2_classes = [c.strip() for c in args.s2_classes.split(",")]
    base = (df["dayObs"] >= d0) & (df["dayObs"] <= d1) & (df["s1_decision"] == args.s1_class)
    host_incl = [c.strip() for c in args.include_host_morph.split(",")] if args.include_host_morph else []
    if host_incl and "host_morph" not in df.columns:
        raise ValueError("--include_host_morph needs a step 2 CSV with a host_morph column "
                         "(data/predictions_step2_<version>_hm.csv from --mode host_morph).")
    has_hm = "host_morph" in df.columns
    is_unc = df["decision"] == "uncertain"
    # 母天体の形のために uncertain になったもの (host_morph の出力では、uncertain はすべてこれ)
    by_host = is_unc & df["host_morph"].isin(["uncertain", "low_snr"]) if has_hm else pd.Series(False, index=df.index)
    decided = df["decision"].isin(s2_classes)
    rf_unc = (is_unc & ~by_host & df["predicted_class"].isin(s2_classes)) if args.include_uncertain \
        else pd.Series(False, index=df.index)
    host_unc = by_host & df["host_morph"].isin(host_incl) if host_incl else pd.Series(False, index=df.index)
    sel = df[base & (decided | rf_unc | host_unc)].copy()
    n = len(s2_classes)
    host_rank = {"uncertain": 0, "low_snr": 1}
    # tier: 0 = 閾値を超えた判定, 1 = RF の uncertain, 2 = 母天体の形が uncertain / low_snr
    sel["tier"] = np.select([sel["decision"].isin(s2_classes), sel["decision"] == "uncertain"], [0, 1], 0)
    if has_hm:
        sel.loc[(sel["decision"] == "uncertain") & sel["host_morph"].isin(host_incl), "tier"] = 2
    sel["is_uncertain"] = sel["tier"] > 0
    sel["s2_class"] = np.where(sel["tier"] == 0, sel["decision"],
                               np.where(sel["tier"] == 1, sel["predicted_class"], "galaxy"))
    sel["s2_label"] = np.select(
        [sel["tier"] == 0, sel["tier"] == 1],
        [sel["decision"], "uncertain(" + sel["predicted_class"].astype(str) + ")"],
        "host:" + (sel["host_morph"].astype(str) if has_hm else ""))
    # 天体ごとに 1 ソース: 確かな判定を優先し、その中で最新
    sel = sel.sort_values(["tier", "midpointMjdTai"], ascending=[False, True]).groupby("diaObjectId").tail(1)
    sel["s2_prob_selected"] = [r[f"prob_{r['s2_class']}"] for _, r in sel.iterrows()]
    sub = np.where(sel["tier"] == 2, sel["host_morph"].map(host_rank).fillna(0) if has_hm else 0,
                   sel["s2_class"].map({c: i for i, c in enumerate(s2_classes)}).fillna(n))
    sel["class_order"] = sel["tier"] * (n + 2) + sub
    sel = sel.sort_values(["class_order", "s2_prob_selected"], ascending=[True, False]).reset_index(drop=True)
    sel["date_utc"] = Time(sel["midpointMjdTai"].values, format="mjd", scale="tai").utc.iso
    sel["date_utc"] = sel["date_utc"].str[:16]
    return sel, d0, d1

def export_pdf(sel, out_pdf, fits_root, title, rows=8, cols=10):
    per_page = rows * cols
    n_pages = max(1, (len(sel) + per_page - 1) // per_page)
    os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            chunk = sel.iloc[page * per_page:(page + 1) * per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.0), squeeze=False)
            fig.suptitle(f"{title} | Page {page + 1}/{n_pages}\nLayout: Sci | Tmp | Dif",
                         fontsize=14, fontweight="bold")
            flat = axes.flatten()
            for i, r in enumerate(tqdm(list(chunk.itertuples()), desc=f"Page {page + 1}/{n_pages}", leave=False)):
                ax = flat[i]
                ax.imshow(triplet(r.diaObjectId, r.midpointMjdTai, r.band, r.diaSourceId, fits_root),
                          cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                final = hasattr(r, "decision_detail")
                unc_mark = ""
                if final:
                    unc_mark = " | " + str(r.decision_detail)
                elif r.tier == 1:
                    unc_mark = " | S2:unc"
                elif r.tier == 2:
                    unc_mark = " | H:" + ("lowSN" if r.host_morph == "low_snr" else "unc")
                if final:
                    if r.host_group == "A":
                        line3 = f"host score {r.host_score:.2f} | Tmp S/N {r.tmp_snr:.0f}"
                    else:
                        line3 = (f"B: P={r.s2_score:.2f} | Tmp S/N@SN {r.tmp_center_snr:.1f}" if np.isfinite(r.s2_score)
                                 else f"B: no model | Tmp S/N@SN {r.tmp_center_snr:.1f}")
                    ax.set_title(f"{r.diaObjectId} | {r.band} | {r.date_utc[5:10]}{unc_mark}\n"
                                 f"real:{r.s1_prob_positive_real:.2f} bogus:{r.s1_prob_bogus:.2f}\n{line3}", fontsize=7)
                    ax.axis("off")
                    continue
                ax.set_title(f"{r.diaObjectId} | {r.band} | {r.date_utc[5:10]}{unc_mark}\n"
                             f"real:{r.s1_prob_positive_real:.2f} bogus:{r.s1_prob_bogus:.2f}\n"
                             f"gal:{r.prob_galaxy:.2f} nh:{r.prob_no_host:.2f} star:{r.prob_star:.2f}",
                             fontsize=7)
                ax.axis("off")
            for j in range(len(chunk), len(flat)):
                flat[j].axis("off")
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig, dpi=120)
            plt.close(fig)

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--step1_csv", default="data/predictions_step1_v20261019.csv")
    p.add_argument("--step2_csv", default="data/predictions_step2_final.csv",
                   help="step2 の判定。既定は最終判定 (host_morph / b_classify が作成)")
    p.add_argument("--s1_class", default="positive_real")
    p.add_argument("--s2_classes", default="galaxy,no_host", help="表示順にカンマ区切り")
    p.add_argument("--start", default="2026-05-01", help="観測夜 (dayObs) の開始日 (この日を含む)")
    p.add_argument("--end", default="2026-05-31", help="観測夜 (dayObs) の終了日 (この日を含む)")
    p.add_argument("--include_uncertain", action="store_true",
                   help="最終判定ファイルでは、判定が uncertain のものも含める。従来の RF の出力では、"
                        "uncertain でも確率が最大のクラスが --s2_classes に含まれるソースを含める")
    p.add_argument("--include_host_morph", type=str, default="",
                   help="host_morph の出力 (*_hm.csv) を使うとき、母天体の形の判定がこれらのために uncertain になった"
                        "ソースも含める。例: uncertain,low_snr")
    p.add_argument("--fits_root", default="cutouts")
    p.add_argument("--output_pdf", default="reports/real_pos_galaxy_nohost_202605.pdf")
    p.add_argument("--rows", type=int, default=8)
    p.add_argument("--cols", type=int, default=10)
    args = p.parse_args()

    sel, d0, d1 = select(args)
    if sel.empty:
        print("No objects match the selection.")
        return
    out_csv = os.path.splitext(args.output_pdf)[0] + ".csv"
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    cols = ["diaObjectId", "diaSourceId", "band", "midpointMjdTai", "date_utc", "dayObs", "s1_decision",
            "s1_prob_positive_real", "s1_prob_negative_real", "s1_prob_bogus", "decision", "s2_label",
            "prob_galaxy", "prob_no_host", "prob_star", "prob_bogus", "host_group", "host_morph", "host_score",
            "band_scores", "tmp_snr", "tmp_size_ratio", "tmp_center_snr", "class_b", "prob_b_no_host",
            "prob_b_real_uncertain", "prob_b_galaxy", "prob_b_bogus"]
    sel[[c for c in cols if c in sel.columns]].rename(
        columns={"decision": "s2_decision", "prob_galaxy": "s2_prob_galaxy", "prob_no_host": "s2_prob_no_host",
                 "prob_star": "s2_prob_star", "prob_bogus": "s2_prob_bogus"}).to_csv(out_csv, index=False)
    print(f"{len(sel)} objects (dayObs {d0}-{d1}): {sel['s2_label'].value_counts().to_dict()} -> {out_csv}")
    unc_txt = " (incl. uncertain)" if args.include_uncertain else ""
    if args.include_host_morph:
        unc_txt += f" (incl. host shape: {args.include_host_morph})"
    title = (f"S1:{args.s1_class} & S2:{'/'.join(args.s2_classes.split(','))}{unc_txt} | "
             f"dayObs {args.start} - {args.end}")
    export_pdf(sel, args.output_pdf, args.fits_root, title, rows=args.rows, cols=args.cols)
    print(f"Done. PDF saved as {args.output_pdf}")

if __name__ == "__main__":
    main()
