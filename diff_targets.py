#!/usr/bin/env python
"""
2 つの rubin_targets.csv を比べ、増えた候補・落ちた候補の CSV と 3 連画像 PDF を作る。
PDF は rubin_targets.pdf と同じ形式 (make_pfs_targets.py pdf と同じ export_pfs_targets_pdf)。

   bin/python diff_targets.py --old reports/sn_xmm_lss_alert_2026-10-29_2026-11-05 \
                              --new reports/sn_xmm_lss_alert_2026-10-30_2026-11-06

--old / --new にはレポートディレクトリ (rubin_targets.csv と candidates.csv を含む) を渡す。
出力 (--new のディレクトリ):
   diff_added.csv   / diff_added.pdf     新しく入った候補
   diff_removed.csv / diff_removed.pdf   落ちた候補 (画像の情報は --old の candidates.csv から)
該当が 0 件なら PDF は作らない (CSV はヘッダのみ)。
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stamp_classifier.pfs_target import export_pfs_targets_pdf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True, help="前回のレポートディレクトリ")
    ap.add_argument("--new", required=True, help="今回のレポートディレクトリ (出力先)")
    ap.add_argument("--fits_root", default="cutouts")
    a = ap.parse_args()

    old = pd.read_csv(os.path.join(a.old, "rubin_targets.csv"), dtype={"ob_code": str})
    new = pd.read_csv(os.path.join(a.new, "rubin_targets.csv"), dtype={"ob_code": str})
    old_name, new_name = os.path.basename(os.path.normpath(a.old)), os.path.basename(os.path.normpath(a.new))

    parts = {
        "added": (new[~new["obj_id"].isin(old["obj_id"])], os.path.join(a.new, "candidates.csv"),
                  f"Added since {old_name}"),
        "removed": (old[~old["obj_id"].isin(new["obj_id"])], os.path.join(a.old, "candidates.csv"),
                    f"Removed since {old_name}"),
    }
    print(f"diff {old_name} ({len(old)}) -> {new_name} ({len(new)})")
    for key, (df, cand, label) in parts.items():
        df = df.sort_values("ob_code")
        csv = os.path.join(a.new, f"diff_{key}.csv")
        df.to_csv(csv, index=False, na_rep="")
        pr = df["priority"].value_counts().sort_index().to_dict()
        print(f"  {key}: {len(df)} objects {pr} -> {csv}")
        if len(df):
            pdf = export_pfs_targets_pdf(
                targets_csv_or_df=df,
                out_pdf=os.path.splitext(csv)[0] + ".pdf",
                candidates_csvs=[cand],
                fits_root=a.fits_root,
                title=f"{label} ({len(df)} objects)",
            )
            print(f"    -> {pdf}")


if __name__ == "__main__":
    main()
