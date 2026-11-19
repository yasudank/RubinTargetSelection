#!/usr/bin/env python
"""
Subaru/PFS ターゲットリスト作成・突合・3連画像PDF生成ツール

1) candidates.csv から PFS ターゲットリストを作成する:
   bin/python make_pfs_targets.py --candidates reports/sn_xmm_lss_alert_2026-10-29_2026-10-31/candidates.csv

   過去のリストがある場合 (既存天体の ob_code 番号を維持し、新規天体に続き番号を振る):
   bin/python make_pfs_targets.py --candidates reports/.../candidates.csv --old_list reports/.../rubin_targets.csv

2) 2 つの PFS ターゲットリストを入力して、古いリストの番号が変わらないように通し番号を調整する:
   bin/python make_pfs_targets.py reconcile --old old_pfs.csv --new new_pfs.csv --out reconciled_pfs.csv
   (オプション --mode merge を付けると、古いリストと新しいリストを統合したファイルを出力)
   (オプション --exptime 2880 で露光時間を上書き設定)
   (オプション --make_pdf を付けると、統合リストの 3 連画像 PDF も同時に生成)

3) rubin_targets.csv を元に 3 連画像 (Science | Template | Difference) の PDF を作成する:
   bin/python make_pfs_targets.py pdf --targets reports/sn_xmm_lss_merged/rubin_targets.csv
"""

import argparse
import glob
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stamp_classifier.pfs_target import (
    create_pfs_targets_df,
    reconcile_pfs_target_lists,
    export_pfs_targets_pdf,
)

def find_default_candidates_csv() -> str:
    """reports/sn_*/candidates.csv の中で最新のものを探す"""
    cand = sorted(glob.glob("reports/sn_*/candidates.csv"), key=os.path.getmtime)
    return cand[-1] if cand else ""

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")

    # 1. candidates -> pfs (既定サブコマンド、または sub なし引数)
    c_p = sub.add_parser("create", help="candidates.csv から PFS ターゲットリストを作成")
    for parser in [p, c_p]:
        parser.add_argument("--candidates", default=None, help="入力 candidates.csv のパス (省略時は最新の reports/sn_*/candidates.csv)")
        parser.add_argument("--old_list", default=None, help="過去の PFS ターゲットリスト (既存天体の ob_code を維持し、新規天体に続き番号を付与)")
        parser.add_argument("--semester", default="s26b", help="学期コード (既定: s26b)")
        parser.add_argument("--exptime", type=int, default=5400, help="露光時間 (秒, 既定: 5400)")
        parser.add_argument("--resolution", default="L", help="解像度モード (既定: L)")
        parser.add_argument("--reference_arm", default="r", help="リファレンスアーム (既定: r)")
        parser.add_argument("--all_bands", action="store_true", help="測定値がないバンドも含め g, r2, i2, z, y 全 5 バンドの列を出力")
        parser.add_argument("--include_errors", action="store_true", help="*_error 列も出力する")
        parser.add_argument("--out", default=None, help="出力 CSV パス (既定: candidates.csv と同じディレクトリの rubin_targets.csv)")
        parser.add_argument("--make_pdf", "--pdf", action="store_true", help="ターゲットの 3 連画像 PDF も同時に作成する")
        parser.add_argument("--fits_root", default="cutouts", help="カットアウト FITS 画像のルートディレクトリ (既定: cutouts)")

    # 2. reconcile サブコマンド
    r_p = sub.add_parser("reconcile", help="2つの PFS ターゲットリストを入力し通し番号を調整")
    r_p.add_argument("--old", required=True, help="古い PFS ターゲットリスト CSV")
    r_p.add_argument("--new", required=True, help="新しい PFS ターゲットリスト CSV")
    r_p.add_argument("--out", required=True, help="調整後の出力 CSV パス")
    r_p.add_argument("--semester", default="s26b", help="学期コード (既定: s26b)")
    r_p.add_argument("--mode", choices=["update_new", "merge"], default="update_new",
                     help="update_new: 新規リストのみ (既存番号維持+続き番号)、merge: 旧新リストを統合")
    r_p.add_argument("--exptime", type=int, default=None, help="露光時間 (秒) を上書き設定")
    r_p.add_argument("--resolution", default=None, help="解像度モード (L/M) を上書き設定")
    r_p.add_argument("--reference_arm", default=None, help="リファレンスアーム (r/b/n/m) を上書き設定")
    r_p.add_argument("--make_pdf", "--pdf", action="store_true", help="統合後のターゲットの 3 連画像 PDF も同時に作成する")
    r_p.add_argument("--fits_root", default="cutouts", help="カットアウト FITS 画像のルートディレクトリ (既定: cutouts)")

    # 3. pdf サブコマンド
    pdf_p = sub.add_parser("pdf", help="rubin_targets.csv から 3 連画像の PDF を作成")
    pdf_p.add_argument("--targets", required=True, help="入力 rubin_targets.csv のパス")
    pdf_p.add_argument("--candidates", nargs="*", default=None, help="候補情報 candidates.csv のパス (省略時は reports/sn_* から自動検出)")
    pdf_p.add_argument("--out", default=None, help="出力 PDF パス (既定: targets と同じディレクトリの .pdf)")
    pdf_p.add_argument("--fits_root", default="cutouts", help="カットアウト FITS 画像のルートディレクトリ (既定: cutouts)")
    pdf_p.add_argument("--rows", type=int, default=8, help="1 ページの行数 (既定: 8)")
    pdf_p.add_argument("--cols", type=int, default=10, help="1 ページの列数 (既定: 10)")
    pdf_p.add_argument("--title", default=None, help="PDF タイトル (省略時は自動設定)")

    args = p.parse_args()

    if args.cmd == "reconcile":
        out_df = reconcile_pfs_target_lists(
            args.old,
            args.new,
            args.out,
            semester=args.semester,
            mode=args.mode,
            exptime=args.exptime,
            resolution=args.resolution,
            reference_arm=args.reference_arm,
            make_pdf=getattr(args, "make_pdf", False),
            fits_root=args.fits_root,
        )
        print(f"Reconciled {len(out_df)} targets (mode: {args.mode}) -> {args.out}")
        if getattr(args, "make_pdf", False):
            pdf_path = os.path.splitext(args.out)[0] + ".pdf"
            print(f"Generated triplet PDF -> {pdf_path}")
        return

    if args.cmd == "pdf":
        pdf_path = export_pfs_targets_pdf(
            targets_csv_or_df=args.targets,
            out_pdf=args.out,
            candidates_csvs=args.candidates,
            fits_root=args.fits_root,
            title=args.title,
            rows=args.rows,
            cols=args.cols,
        )
        print(f"Generated triplet PDF -> {pdf_path}")
        return

    # 通常の create 処理
    cand_path = args.candidates or find_default_candidates_csv()
    if not cand_path or not os.path.exists(cand_path):
        p.error("candidates.csv が見つかりません。--candidates でパスを指定してください。")

    out_path = args.out or os.path.join(os.path.dirname(cand_path), "rubin_targets.csv")
    df_cand = pd.read_csv(cand_path)
    print(f"Reading {len(df_cand)} candidates from {cand_path} ...")

    pfs_df = create_pfs_targets_df(
        candidates_df=df_cand,
        semester=args.semester,
        exptime=args.exptime,
        resolution=args.resolution,
        reference_arm=args.reference_arm,
        old_pfs_list=args.old_list,
        all_bands=args.all_bands,
        include_errors=args.include_errors,
    )

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    pfs_df.to_csv(out_path, index=False, na_rep="")
    print(f"Saved {len(pfs_df)} PFS targets -> {out_path}")
    print(f"Columns: {list(pfs_df.columns)}")
    print(f"Priority breakdown:\n{pfs_df['priority'].value_counts().sort_index().to_dict()}")

    if getattr(args, "make_pdf", False):
        pdf_path = os.path.splitext(out_path)[0] + ".pdf"
        export_pfs_targets_pdf(
            targets_csv_or_df=pfs_df,
            out_pdf=pdf_path,
            candidates_csvs=[cand_path],
            fits_root=args.fits_root,
        )
        print(f"Generated triplet PDF -> {pdf_path}")

if __name__ == "__main__":
    main()

