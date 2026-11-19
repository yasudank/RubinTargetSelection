#!/usr/bin/env python3
"""
Subaru/PFS design ファイル (pfsDesign-0x*.fits) に基づき、
視野 (FOV / design) ごとに割り当てられた超新星候補天体の
3 連画像 (Science | Template | Difference) の PDF を作成する。

全視野まとめ総合 PDF (all_pfs_designs_cutouts.pdf) では：
  - 第 1 ページ: 各視野および未割り当て天体の候補天体数、priority ごとの天体数のまとめ表
  - 第 2 ページ以降: 視野名順 (OPT_FOV_1, OPT_FOV_2, ...) の 3 連画像
  - 最終ページ: ファイバーが割り当てられなかった候補天体の 3 連画像

使用例:
  # 既定のレポートディレクトリと design ディレクトリで実行
  ./bin/python export_pfs_design_pdf.py

  # 特定のレポートディレクトリや出力先を指定して実行
  ./bin/python export_pfs_design_pdf.py \\
      --report_dir reports/sn_xmm_lss_alert_2026-10-29_2026-10-31 \\
      --design_dir design \\
      --out_dir reports/sn_xmm_lss_alert_2026-10-29_2026-10-31/design_cutouts

  # 特定の design ファイルのみ処理
  ./bin/python export_pfs_design_pdf.py --design_files design/pfsDesign-0x140069c0d16bb4e5.fits
"""

import argparse
import glob
import os
import re
import sys
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.time import Time
from tqdm import tqdm

# triplet 関数をインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_candidates_pdf import triplet


def natural_sort_key(s: str):
    """文字列中の数値を数値として扱い、自然順ソート (OPT_FOV_1, OPT_FOV_2, ...) を行うキー"""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", str(s))]


def parse_pfs_design_fits(fits_path: str) -> Tuple[dict, pd.DataFrame]:
    """
    pfsDesign FITS ファイルからヘッダーメタデータと Rubin ターゲットの割り当て一覧を取得する。
    """
    with fits.open(fits_path) as hdul:
        h0 = hdul[0].header
        dsgn_nam = str(h0.get("DSGN_NAM", "UNKNOWN")).strip()
        dsgn_id = h0.get("W_PFDSGN")
        if isinstance(dsgn_id, int):
            dsgn_hex = f"0x{dsgn_id:016x}"
        else:
            m = re.search(r"0x[0-9a-fA-F]+", os.path.basename(fits_path))
            dsgn_hex = m.group(0) if m else str(dsgn_id)

        meta = {
            "path": fits_path,
            "filename": os.path.basename(fits_path),
            "design_name": dsgn_nam,
            "design_id_hex": dsgn_hex,
            "design_id": dsgn_id,
            "ra": float(h0.get("RA", 0.0)),
            "dec": float(h0.get("DEC", 0.0)),
            "posang": float(h0.get("POSANG", 0.0)),
        }

        d = hdul[1].data
        ob_codes = [str(c).strip() for c in d["obCode"]]
        mask = [c.startswith("rubin_") for c in ob_codes]

        if not any(mask):
            return meta, pd.DataFrame()

        sub = d[mask]
        # FITS のビッグエンディアン配列をリトルエンディアン (ネイティブ) に変換して DataFrame 化
        df_design = pd.DataFrame({
            "fiberId": np.array(sub["fiberId"], dtype=np.int32),
            "cobraId": np.array(sub["cobraId"], dtype=np.int32),
            "pfs_ra": np.array(sub["ra"], dtype=np.float64),
            "pfs_dec": np.array(sub["dec"], dtype=np.float64),
            "targetType": np.array(sub["targetType"], dtype=np.int32),
            "ob_code": [ob_codes[i] for i, m in enumerate(mask) if m],
            "obj_id": np.array(sub["objId"], dtype=np.int64),
        })

    return meta, df_design


def load_candidate_catalogs(candidates_csv: str, rubin_targets_csv: str) -> pd.DataFrame:
    """
    candidates.csv と rubin_targets.csv を読み込み、obj_id (diaObjectId) で結合する。
    """
    if not os.path.exists(candidates_csv):
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")
    if not os.path.exists(rubin_targets_csv):
        raise FileNotFoundError(f"Rubin targets CSV not found: {rubin_targets_csv}")

    df_cand = pd.read_csv(candidates_csv)
    df_targ = pd.read_csv(rubin_targets_csv)

    merged = df_targ.merge(
        df_cand,
        left_on="obj_id",
        right_on="diaObjectId",
        suffixes=("", "_cand")
    )
    return merged


def render_field_pdf(
    meta: dict,
    targets: pd.DataFrame,
    out_pdf: str,
    fits_root: str,
    rows: int = 6,
    cols: int = 8,
    dpi: int = 120,
) -> None:
    """
    1 つの PFS 視野 (design) または未割り当て天体群に含まれるターゲットの 3 連画像 PDF を描画・保存する。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)

    n_targets = len(targets)
    per_page = rows * cols
    n_pages = max(1, (n_targets + per_page - 1) // per_page)

    with PdfPages(out_pdf) as pdf:
        for page in range(n_pages):
            chunk = targets.iloc[page * per_page:(page + 1) * per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), squeeze=False)

            if meta.get("is_unassigned"):
                suptitle = (
                    f"Unassigned Rubin Candidates (No Fiber Allocated) | "
                    f"{n_targets} Candidates | Page {page + 1}/{n_pages}\n"
                    f"Cutout Layout: Science | Template | Difference"
                )
            else:
                page_sub = f" | Page {page + 1}/{n_pages}" if n_pages > 1 else ""
                suptitle = (
                    f"Subaru/PFS Field: {meta['design_name']} ({meta['design_id_hex']}) | "
                    f"Center: RA={meta['ra']:.4f}°, Dec={meta['dec']:.4f}° | "
                    f"{n_targets} Rubin Targets{page_sub}\n"
                    f"Cutout Layout: Science | Template | Difference"
                )
            fig.suptitle(suptitle, fontsize=12, fontweight="bold")
            flat = axes.flatten()

            for i, r in enumerate(chunk.itertuples()):
                ax = flat[i]
                img = triplet(r.diaObjectId, r.rep_mjd, r.rep_band, r.rep_diaSourceId, fits_root)
                ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

                try:
                    date_str = Time(r.rep_mjd, format="mjd", scale="tai").utc.iso[5:10]
                except Exception:
                    date_str = "N/A"

                if str(r.decision_detail).startswith("A:"):
                    line3 = f"host score {getattr(r, 'host_score', float('nan')):.2f}"
                else:
                    pb = getattr(r, "prob_b", float("nan"))
                    line3 = f"B: P={pb:.2f}" if pd.notna(pb) else "B"

                s1_dec = getattr(r, "s1_decision", "positive_real")
                neg_p = getattr(r, "s1_prob_negative_real", 0.0)
                pos_p = getattr(r, "s1_prob_positive_real", 0.0)
                bog_p = getattr(r, "s1_prob_bogus", 0.0)

                if s1_dec == "negative_real":
                    s1_lbl = f"-real:{neg_p:.2f}"
                else:
                    s1_lbl = f"real:{pos_p:.2f}"

                pri = getattr(r, "priority", "N/A")
                fib_id = getattr(r, "fiberId", None)
                if pd.notna(fib_id) and int(fib_id) > 0:
                    fib_lbl = f"fib:{int(fib_id)}"
                else:
                    fib_lbl = "unassigned"

                t1 = f"{r.ob_code} | {fib_lbl} | P:{pri} | {r.rep_band} | {date_str}"
                t2 = f"{r.diaObjectId} | {r.decision_detail}"
                t3 = f"{s1_lbl} bog:{bog_p:.2f} | {line3}"

                ax.set_title(f"{t1}\n{t2}\n{t3}", fontsize=6.5)
                ax.axis("off")

            for j in range(len(chunk), len(flat)):
                flat[j].axis("off")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig, dpi=dpi)
            plt.close(fig)


def render_summary_page(
    fields_data: List[Tuple[dict, pd.DataFrame]],
    unassigned_df: Optional[pd.DataFrame],
    report_name: str,
    total_pages: int,
    rows: int = 6,
    cols: int = 8,
) -> plt.Figure:
    """
    総合 PDF の第 1 ページとなる、視野別・未割り当て・優先度別の候補天体数サマリー表を生成する。
    """
    fig = plt.figure(figsize=(cols * 2.5, rows * 2.2))
    ax = fig.add_subplot(111)
    ax.axis("off")

    all_assigned_objs = [oid for _, df in fields_data for oid in df["obj_id"]]
    n_unique_assigned = len(set(all_assigned_objs))
    n_total_allocations = len(all_assigned_objs)
    n_unassigned = len(unassigned_df) if unassigned_df is not None else 0
    total_candidates_count = n_unique_assigned + n_unassigned

    title = "Subaru/PFS Observation Designs: Supernova Candidate Target Summary"
    subtitle = (
        f"Candidates Dataset: {report_name} | "
        f"Total Candidates: {total_candidates_count} ({n_unique_assigned} assigned across fields, {n_unassigned} unassigned)"
    )
    fig.text(0.5, 0.92, title, ha="center", va="top", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.88, subtitle, ha="center", va="top", fontsize=13, color="#333333")

    headers = [
        "Field Name",
        "Design ID",
        "Center RA (°)",
        "Center Dec (°)",
        "Targets (N)",
        "Priority 0",
        "Priority 1",
        "Priority 2",
        "Cutout Page",
    ]
    # Design ID (18文字) に十分な幅を確保
    col_widths = [0.12, 0.20, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08, 0.10]

    table_rows = []
    current_page = 2
    per_page = rows * cols

    tot_targets = 0
    tot_p0 = 0
    tot_p1 = 0
    tot_p2 = 0

    for meta, targets in fields_data:
        n_t = len(targets)
        p0 = int((targets["priority"] == 0).sum()) if "priority" in targets.columns else 0
        p1 = int((targets["priority"] == 1).sum()) if "priority" in targets.columns else 0
        p2 = int((targets["priority"] == 2).sum()) if "priority" in targets.columns else 0

        tot_targets += n_t
        tot_p0 += p0
        tot_p1 += p1
        tot_p2 += p2

        n_p = max(1, (n_t + per_page - 1) // per_page)
        page_str = f"Page {current_page}" if n_p == 1 else f"Pages {current_page}–{current_page + n_p - 1}"
        current_page += n_p

        ra_val = meta["ra"]
        dec_val = meta["dec"]
        table_rows.append([
            meta["design_name"],
            meta["design_id_hex"],
            f"{ra_val:.4f}",
            f"{dec_val:.4f}",
            str(n_t),
            str(p0),
            str(p1),
            str(p2),
            page_str,
        ])

    # 未割り当て行
    up0, up1, up2 = 0, 0, 0
    if n_unassigned > 0 and unassigned_df is not None:
        up0 = int((unassigned_df["priority"] == 0).sum()) if "priority" in unassigned_df.columns else 0
        up1 = int((unassigned_df["priority"] == 1).sum()) if "priority" in unassigned_df.columns else 0
        up2 = int((unassigned_df["priority"] == 2).sum()) if "priority" in unassigned_df.columns else 0

        n_up = max(1, (n_unassigned + per_page - 1) // per_page)
        unassigned_page_str = f"Page {current_page}" if n_up == 1 else f"Pages {current_page}–{current_page + n_up - 1}"
        current_page += n_up

        table_rows.append([
            "Unassigned",
            "—",
            "—",
            "—",
            str(n_unassigned),
            str(up0),
            str(up1),
            str(up2),
            unassigned_page_str,
        ])

    # 合計行
    all_p0 = tot_p0 + up0
    all_p1 = tot_p1 + up1
    all_p2 = tot_p2 + up2
    page_range_all = f"Pages 2–{total_pages}"
    table_rows.append([
        "Total Candidates",
        "—",
        "—",
        "—",
        str(total_candidates_count),
        str(all_p0),
        str(all_p1),
        str(all_p2),
        page_range_all,
    ])

    tab = ax.table(
        cellText=table_rows,
        colLabels=headers,
        colWidths=col_widths,
        loc="center",
        cellLoc="center",
        bbox=[0.06, 0.33, 0.88, 0.50],
    )
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)

    for (row, col), cell in tab.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold", fontsize=11.5)
        elif row == len(table_rows):
            # Total row
            cell.set_facecolor("#eaeded")
            cell.set_text_props(fontweight="bold", fontsize=11)
        elif n_unassigned > 0 and row == len(table_rows) - 1:
            # Unassigned row
            cell.set_facecolor("#fef9e7")
            cell.set_text_props(fontweight="bold", fontsize=10.5)
        else:
            if row % 2 == 1:
                cell.set_facecolor("#f8f9fa")
            else:
                cell.set_facecolor("#ffffff")

    note_text = (
        "Priority Definition:\n"
        "  • Priority 0: High-confidence candidates in galaxies or hostless (positive_real & galaxy, positive_real & no_host)\n"
        "  • Priority 1: Negative difference detections with host galaxies (negative_real & galaxy)\n"
        "  • Priority 2: Uncertain host associations (positive_real & uncertain)\n\n"
        f"Notes:\n"
        f"  • {n_unique_assigned} unique candidates assigned across {len(fields_data)} PFS fields "
        f"({n_total_allocations} total allocations; {n_total_allocations - n_unique_assigned} candidate(s) allocated in overlapping fields).\n"
        f"  • {n_unassigned} candidates unassigned (outside field footprints or unallocated fibers), shown at the end.\n"
        "  • Cutout Layout on subsequent pages: Science | Template | Difference (30×30 arcsec stamps)\n"
        "  • Individual cutout labels: [Line 1] ob_code | fiberId / unassigned | Priority | Band | Date (UTC)\n"
        "                              [Line 2] diaObjectId | Decision Detail\n"
        "                              [Line 3] Step 1 real/bogus score | Step 2 host score or class probability"
    )
    fig.text(
        0.06,
        0.28,
        note_text,
        ha="left",
        va="top",
        fontsize=10.5,
        linespacing=1.5,
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6f7", edgecolor="#bdc3c7"),
    )
    fig.text(
        0.5,
        0.03,
        f"Page 1 / {total_pages} (Summary Table)",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="#555555",
    )
    return fig


def render_combined_pdf(
    fields_data: List[Tuple[dict, pd.DataFrame]],
    unassigned_df: Optional[pd.DataFrame],
    out_pdf: str,
    fits_root: str,
    report_name: str = "",
    rows: int = 6,
    cols: int = 8,
    dpi: int = 120,
) -> None:
    """
    全視野および未割り当て天体を 1 つにまとめた総合 PDF を作成する。
    第 1 ページにサマリー表、第 2 ページ以降に視野名順の 3 連画像、最後に未割り当て天体の 3 連画像を配置する。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_pdf)), exist_ok=True)
    per_page = rows * cols

    # 視野名で自然順ソート (OPT_FOV_1, OPT_FOV_2, ..., OPT_FOV_7)
    fields_data_sorted = sorted(fields_data, key=lambda x: natural_sort_key(x[0]["design_name"]))

    # 総ページ数の計算: サマリー表 1 ページ + 各視野のカットアウトページ数 + 未割り当てページ数
    cutout_page_counts = [max(1, (len(t) + per_page - 1) // per_page) for _, t in fields_data_sorted if len(t) > 0]
    unassigned_page_count = max(1, (len(unassigned_df) + per_page - 1) // per_page) if (unassigned_df is not None and len(unassigned_df) > 0) else 0
    total_pages = 1 + sum(cutout_page_counts) + unassigned_page_count

    with PdfPages(out_pdf) as pdf:
        # [第 1 ページ] サマリー表
        fig_summary = render_summary_page(
            fields_data=fields_data_sorted,
            unassigned_df=unassigned_df,
            report_name=report_name,
            total_pages=total_pages,
            rows=rows,
            cols=cols,
        )
        pdf.savefig(fig_summary, dpi=dpi)
        plt.close(fig_summary)

        # [第 2 ページ以降] 視野ごとのカットアウト
        current_cutout_page = 2
        for meta, targets in fields_data_sorted:
            n_targets = len(targets)
            if n_targets == 0:
                continue
            n_pages = max(1, (n_targets + per_page - 1) // per_page)

            for page in range(n_pages):
                chunk = targets.iloc[page * per_page:(page + 1) * per_page]
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), squeeze=False)

                page_sub = f" (Page {page + 1}/{n_pages})" if n_pages > 1 else ""
                suptitle = (
                    f"Subaru/PFS Field: {meta['design_name']} ({meta['design_id_hex']}) | "
                    f"Center: RA={meta['ra']:.4f}°, Dec={meta['dec']:.4f}° | "
                    f"{n_targets} Rubin Targets{page_sub}\n"
                    f"Combined PDF Page {current_cutout_page}/{total_pages} | "
                    f"Cutout Layout: Science | Template | Difference"
                )
                fig.suptitle(suptitle, fontsize=12, fontweight="bold")
                flat = axes.flatten()

                for i, r in enumerate(chunk.itertuples()):
                    ax = flat[i]
                    img = triplet(r.diaObjectId, r.rep_mjd, r.rep_band, r.rep_diaSourceId, fits_root)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

                    try:
                        date_str = Time(r.rep_mjd, format="mjd", scale="tai").utc.iso[5:10]
                    except Exception:
                        date_str = "N/A"

                    if str(r.decision_detail).startswith("A:"):
                        line3 = f"host score {getattr(r, 'host_score', float('nan')):.2f}"
                    else:
                        pb = getattr(r, "prob_b", float("nan"))
                        line3 = f"B: P={pb:.2f}" if pd.notna(pb) else "B"

                    s1_dec = getattr(r, "s1_decision", "positive_real")
                    neg_p = getattr(r, "s1_prob_negative_real", 0.0)
                    pos_p = getattr(r, "s1_prob_positive_real", 0.0)
                    bog_p = getattr(r, "s1_prob_bogus", 0.0)

                    if s1_dec == "negative_real":
                        s1_lbl = f"-real:{neg_p:.2f}"
                    else:
                        s1_lbl = f"real:{pos_p:.2f}"

                    pri = getattr(r, "priority", "N/A")
                    t1 = f"{r.ob_code} | fib:{r.fiberId} | P:{pri} | {r.rep_band} | {date_str}"
                    t2 = f"{r.diaObjectId} | {r.decision_detail}"
                    t3 = f"{s1_lbl} bog:{bog_p:.2f} | {line3}"

                    ax.set_title(f"{t1}\n{t2}\n{t3}", fontsize=6.5)
                    ax.axis("off")

                for j in range(len(chunk), len(flat)):
                    flat[j].axis("off")

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                current_cutout_page += 1

        # [最終ページ] 未割り当て天体のカットアウト
        if unassigned_df is not None and not unassigned_df.empty:
            n_targets = len(unassigned_df)
            n_pages = max(1, (n_targets + per_page - 1) // per_page)

            for page in range(n_pages):
                chunk = unassigned_df.iloc[page * per_page:(page + 1) * per_page]
                fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.2), squeeze=False)

                page_sub = f" (Page {page + 1}/{n_pages})" if n_pages > 1 else ""
                suptitle = (
                    f"Unassigned Rubin Candidates (No Fiber Allocated) | "
                    f"{n_targets} Candidates{page_sub}\n"
                    f"Combined PDF Page {current_cutout_page}/{total_pages} | "
                    f"Cutout Layout: Science | Template | Difference"
                )
                fig.suptitle(suptitle, fontsize=12, fontweight="bold")
                flat = axes.flatten()

                for i, r in enumerate(chunk.itertuples()):
                    ax = flat[i]
                    img = triplet(r.diaObjectId, r.rep_mjd, r.rep_band, r.rep_diaSourceId, fits_root)
                    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")

                    try:
                        date_str = Time(r.rep_mjd, format="mjd", scale="tai").utc.iso[5:10]
                    except Exception:
                        date_str = "N/A"

                    if str(r.decision_detail).startswith("A:"):
                        line3 = f"host score {getattr(r, 'host_score', float('nan')):.2f}"
                    else:
                        pb = getattr(r, "prob_b", float("nan"))
                        line3 = f"B: P={pb:.2f}" if pd.notna(pb) else "B"

                    s1_dec = getattr(r, "s1_decision", "positive_real")
                    neg_p = getattr(r, "s1_prob_negative_real", 0.0)
                    pos_p = getattr(r, "s1_prob_positive_real", 0.0)
                    bog_p = getattr(r, "s1_prob_bogus", 0.0)

                    if s1_dec == "negative_real":
                        s1_lbl = f"-real:{neg_p:.2f}"
                    else:
                        s1_lbl = f"real:{pos_p:.2f}"

                    pri = getattr(r, "priority", "N/A")
                    t1 = f"{r.ob_code} | unassigned | P:{pri} | {r.rep_band} | {date_str}"
                    t2 = f"{r.diaObjectId} | {r.decision_detail}"
                    t3 = f"{s1_lbl} bog:{bog_p:.2f} | {line3}"

                    ax.set_title(f"{t1}\n{t2}\n{t3}", fontsize=6.5)
                    ax.axis("off")

                for j in range(len(chunk), len(flat)):
                    flat[j].axis("off")

                plt.tight_layout(rect=[0, 0, 1, 0.96])
                pdf.savefig(fig, dpi=dpi)
                plt.close(fig)
                current_cutout_page += 1


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--report_dir",
        default="reports/sn_xmm_lss_alert_2026-10-29_2026-10-31",
        help="候補天体レポートディレクトリ (既定: reports/sn_xmm_lss_alert_2026-10-29_2026-10-31)"
    )
    p.add_argument(
        "--candidates_csv",
        default=None,
        help="candidates.csv のパス (省略時は <report_dir>/candidates.csv)"
    )
    p.add_argument(
        "--rubin_targets_csv",
        default=None,
        help="rubin_targets.csv のパス (省略時は <report_dir>/rubin_targets.csv)"
    )
    p.add_argument(
        "--design_dir",
        default="design",
        help="pfsDesign*.fits ファイルが置かれているディレクトリ (既定: design)"
    )
    p.add_argument(
        "--design_files",
        nargs="*",
        default=None,
        help="個別に指定する pfsDesign FITS ファイルのリスト"
    )
    p.add_argument(
        "--fits_root",
        default="cutouts",
        help="カットアウト FITS 画像のルートディレクトリ (既定: cutouts)"
    )
    p.add_argument(
        "--out_dir",
        default=None,
        help="PDF 出力先ディレクトリ (既定: <report_dir>/design_cutouts)"
    )
    p.add_argument(
        "--order",
        choices=["ob_code", "fiber_id", "priority"],
        default="ob_code",
        help="PDF 内の天体の並び順 (既定: ob_code; candidates.csv の priority 順に準拠)"
    )
    p.add_argument(
        "--rows",
        type=int,
        default=6,
        help="1 ページの行数 (既定: 6)"
    )
    p.add_argument(
        "--cols",
        type=int,
        default=8,
        help="1 ページの列数 (既定: 8)"
    )
    p.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="PDF レンダリング DPI (既定: 120)"
    )
    p.add_argument(
        "--combined_pdf",
        action="store_true",
        default=True,
        help="全視野をまとめた総合 PDF も同時に作成する (既定: True)"
    )
    p.add_argument(
        "--no_combined_pdf",
        dest="combined_pdf",
        action="store_false",
        help="全視野まとめ PDF の作成を無効化する"
    )

    args = p.parse_args()

    cand_csv = args.candidates_csv or os.path.join(args.report_dir, "candidates.csv")
    targ_csv = args.rubin_targets_csv or os.path.join(args.report_dir, "rubin_targets.csv")
    out_dir = args.out_dir or os.path.join(args.report_dir, "design_cutouts")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[*] Loading candidate catalogs:")
    print(f"    - Candidates: {cand_csv}")
    print(f"    - Targets:    {targ_csv}")
    candidates = load_candidate_catalogs(cand_csv, targ_csv)
    print(f"    -> Loaded {len(candidates)} candidates.")

    # design ファイルの収集
    if args.design_files:
        fits_files = sorted(args.design_files)
    else:
        fits_files = sorted(glob.glob(os.path.join(args.design_dir, "pfsDesign*.fits")))

    if not fits_files:
        print(f"[!] No pfsDesign*.fits files found in {args.design_dir}")
        return

    print(f"[*] Found {len(fits_files)} pfsDesign file(s).")

    all_fields_data = []
    summary_rows = []
    assigned_obj_ids = set()

    for fpath in tqdm(fits_files, desc="Processing PFS Designs"):
        meta, df_design = parse_pfs_design_fits(fpath)
        if df_design.empty:
            print(f"    [Skip] {meta['filename']}: No Rubin targets assigned.")
            continue

        assigned_obj_ids.update(df_design["obj_id"])

        # ターゲット情報のマージ (obj_id を主キーとして結合)
        # design ファイル内の ob_code を正として優先
        cand_sub = candidates.drop(columns=["ob_code"]) if "ob_code" in candidates.columns else candidates
        m = df_design.merge(
            cand_sub,
            on="obj_id",
            suffixes=("", "_cand")
        )

        # 並び順のソート
        if args.order == "ob_code":
            m = m.sort_values("ob_code").reset_index(drop=True)
        elif args.order == "fiber_id":
            m = m.sort_values("fiberId").reset_index(drop=True)
        elif args.order == "priority":
            m = m.sort_values(["priority", "ob_code"]).reset_index(drop=True)

        # 出力ファイル名: 例 OPT_FOV_1_pfsDesign-0x412ffc3962398b04.pdf
        fov_tag = meta["design_name"].replace(" ", "_")
        pdf_name = f"{fov_tag}_{os.path.splitext(meta['filename'])[0]}.pdf"
        out_pdf = os.path.join(out_dir, pdf_name)

        # 個別視野 PDF 描画
        render_field_pdf(
            meta=meta,
            targets=m,
            out_pdf=out_pdf,
            fits_root=args.fits_root,
            rows=args.rows,
            cols=args.cols,
            dpi=args.dpi,
        )

        # 各視野の CSV も保存
        csv_name = f"{fov_tag}_{os.path.splitext(meta['filename'])[0]}_targets.csv"
        out_csv = os.path.join(out_dir, csv_name)
        m.to_csv(out_csv, index=False)

        all_fields_data.append((meta, m))
        summary_rows.append({
            "design_name": meta["design_name"],
            "design_id_hex": meta["design_id_hex"],
            "center_ra": meta["ra"],
            "center_dec": meta["dec"],
            "n_targets": len(m),
            "priority_0": sum(m["priority"] == 0),
            "priority_1": sum(m["priority"] == 1),
            "priority_2": sum(m["priority"] == 2),
            "pdf_file": pdf_name,
            "csv_file": csv_name,
        })

    # 未割り当て天体の抽出
    unassigned_df = candidates[~candidates["obj_id"].isin(assigned_obj_ids)].copy()
    if args.order == "ob_code":
        unassigned_df = unassigned_df.sort_values("ob_code").reset_index(drop=True)
    elif args.order == "priority":
        unassigned_df = unassigned_df.sort_values(["priority", "ob_code"]).reset_index(drop=True)
    elif args.order == "fiber_id":
        unassigned_df = unassigned_df.sort_values("ob_code").reset_index(drop=True)

    print(f"[*] Candidates allocation status:")
    print(f"    - Total Candidates:    {len(candidates)}")
    print(f"    - Assigned (Unique):   {len(assigned_obj_ids)}")
    print(f"    - Unassigned:          {len(unassigned_df)}")

    if not unassigned_df.empty:
        # 未割り当て天体の個別 PDF と CSV も保存
        unassigned_meta = {
            "design_name": "Unassigned Candidates",
            "design_id_hex": "No Fiber",
            "ra": 0.0,
            "dec": 0.0,
            "is_unassigned": True,
        }
        unassigned_pdf = os.path.join(out_dir, "unassigned_targets.pdf")
        unassigned_csv = os.path.join(out_dir, "unassigned_targets.csv")
        render_field_pdf(
            meta=unassigned_meta,
            targets=unassigned_df,
            out_pdf=unassigned_pdf,
            fits_root=args.fits_root,
            rows=args.rows,
            cols=args.cols,
            dpi=args.dpi,
        )
        unassigned_df.to_csv(unassigned_csv, index=False)

        summary_rows.append({
            "design_name": "Unassigned",
            "design_id_hex": "—",
            "center_ra": np.nan,
            "center_dec": np.nan,
            "n_targets": len(unassigned_df),
            "priority_0": sum(unassigned_df["priority"] == 0),
            "priority_1": sum(unassigned_df["priority"] == 1),
            "priority_2": sum(unassigned_df["priority"] == 2),
            "pdf_file": "unassigned_targets.pdf",
            "csv_file": "unassigned_targets.csv",
        })

    # 全視野まとめ PDF の作成 (第 1 ページ: サマリー表、第 2〜8 ページ: 視野名順、最終ページ: 未割り当て)
    if args.combined_pdf and all_fields_data:
        comb_pdf_name = "all_pfs_designs_cutouts.pdf"
        comb_pdf_path = os.path.join(out_dir, comb_pdf_name)
        report_tag = os.path.basename(os.path.normpath(args.report_dir))
        print(f"[*] Rendering combined PDF: {comb_pdf_path}")
        render_combined_pdf(
            fields_data=all_fields_data,
            unassigned_df=unassigned_df,
            out_pdf=comb_pdf_path,
            fits_root=args.fits_root,
            report_name=report_tag,
            rows=args.rows,
            cols=args.cols,
            dpi=args.dpi,
        )

    # サマリー CSV の作成 (視野名で自然順ソート、Unassigned は最後)
    summary_df = pd.DataFrame(summary_rows)
    # Unassigned 以外をソート
    is_unassigned = summary_df["design_name"] == "Unassigned"
    df_fov = summary_df[~is_unassigned].copy()
    df_fov["_sort_key"] = df_fov["design_name"].map(natural_sort_key)
    df_fov = df_fov.sort_values("_sort_key").drop(columns=["_sort_key"])
    if any(is_unassigned):
        summary_df = pd.concat([df_fov, summary_df[is_unassigned]], ignore_index=True)
    else:
        summary_df = df_fov.reset_index(drop=True)

    sum_csv_path = os.path.join(out_dir, "pfs_designs_summary.csv")
    summary_df.to_csv(sum_csv_path, index=False)

    print("\n" + "=" * 80)
    print("Subaru/PFS Design Cutout PDFs Generated Successfully")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print(f"Output directory: {out_dir}")
    if args.combined_pdf:
        print(f"Combined PDF:     {os.path.join(out_dir, 'all_pfs_designs_cutouts.pdf')}")
    print(f"Summary CSV:      {sum_csv_path}")


if __name__ == "__main__":
    main()
