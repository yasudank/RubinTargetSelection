"""
新しいアラートデータから、固定したモデル一式で超新星候補を選ぶ。

1) モデル一式を固定する (モデルを学び直したときに 1 回):
   bin/python find_sn_candidates.py freeze --version snp_v1

2) 新しいデータに適用する:
   bin/python find_sn_candidates.py run --db cosmos_alert --start 2026-07-01 --end 2026-07-31 --bundle snp_v1

   --start / --end は観測夜 (dayObs = floor(MJD - 0.5)、チリの夜の開始日) で、両端を含む。
   --min_nights (または --min_nights_total): 候補に残す最小観測夜数 (既定 1; 2 以上で単一夜の移動天体等を除外)
   --min_nights_no_host: no_host 候補にのみ適用する最小観測夜数 (既定 2; 母銀河ありは 1 夜でも残す)
   1 夜だけで検出された天体は、その夜に位置を覆った visit が 2 つ以上あれば、その夜に 2 回以上の検出がないと除く
   (visit が 1 つだけの夜は残す)。残った 1 夜だけの天体の priority は 2、0/1 は 2 夜以上で検出された天体だけ。
   candidates.csv の max_det_per_night / n_visits_night にその判定の値を出す。
   出力 (--out、既定 reports/sn_<db>_<start>_<end>/):
     candidates.csv            候補の一覧 (1 天体 1 行; 各バンドの期間内で最新の scienceFlux / Err / MJD を含む)
     rubin_targets.csv         Subaru/PFS Target Uploader 形式のターゲットリスト
     candidates.pdf            candidates.csv と同じ順に、各天体の代表の検出の Sci | Tmp | Dif を 1 ページ 80 天体で並べる
     objects/<diaObjectId>.csv (--per_object_csv のときだけ) 天体ごとの、各バンドの期間内で最新の検出
     all_sources.csv           期間内の全ソースの step1 / step2 の判定
     visit_check.csv           visit の異常チェックの結果
     run_info.json             実行条件と件数
   DB は読み取りだけ。カットアウトは --fits_root (既定 cutouts) の下にある前提 (avro_to_fits_pg.py で作成)。
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze", help="現在のモデルと閾値を models/sn_pipeline/<version>/ に固定する")
    f.add_argument("--version", required=True)
    f.add_argument("--step1_version", default="v20261019")
    f.add_argument("--moving_min_sep", type=float, default=1.0)
    f.add_argument("--moving_min_rate", type=float, default=0.3)
    r = sub.add_parser("run", help="新しいデータから候補を選ぶ")
    r.add_argument("--db", required=True, help="PostgreSQL のデータベース名")
    r.add_argument("--start", required=True, help="観測夜の開始日 YYYY-MM-DD (含む)")
    r.add_argument("--end", required=True, help="観測夜の終了日 YYYY-MM-DD (含む)")
    r.add_argument("--bundle", required=True, help="freeze で固定したモデル一式の名前")
    r.add_argument("--out", default=None)
    r.add_argument("--fits_root", default="cutouts")
    r.add_argument("--host", default="localhost")
    r.add_argument("--port", type=int, default=5432)
    r.add_argument("--user", default="lsst")
    r.add_argument("--password", default=os.environ.get("PGPASSWORD", "rubin"))
    r.add_argument("--strict", action="store_true", help="step2 が uncertain のものは候補に含めない (既定は含める)")
    r.add_argument("--keep_suspicious_visits", action="store_true",
                   help="異常と判定された visit のソースも候補に残す (既定は除く; visit_check.csv に記録)")
    r.add_argument("--per_object_csv", action="store_true", help="天体ごとの CSV (objects/<diaObjectId>.csv) も出力する")
    r.add_argument("--workers", type=int, default=8)
    r.add_argument("--min_nights", "--min_nights_total", dest="min_nights", type=int, default=1,
                   help="候補に残す最小の総観測夜数 (n_nights_total >= N, 既定 1; 2 以上で単一夜の移動天体・ノイズを除外)")
    r.add_argument("--min_nights_no_host", type=int, default=2,
                   help="no_host 候補にのみ適用する最小観測夜数 (既定 2; 2 以上で単一夜の hostless 移動天体等を除外、母銀河ありは 1 夜でも残す)")
    r.add_argument("--no_negative_galaxy", action="store_true",
                   help="step1 が negative_real かつ step2 が galaxy の天体を候補から除外する (既定は候補に含める)")
    r.add_argument("--per_night_band", choices=["snr", "latest"], default="snr",
                   help="同じ天体・夜・バンドに複数の検出があるとき分類に使う 1 件: snr = S/N 最大 (既定), latest = 最新")
    r.add_argument("--semester", "--pfs_semester", dest="pfs_semester", default="s26b",
                   help="PFS ターゲットの ob_code 用の学期名 (既定: s26b; rubin_{semester}_XXXX)")
    r.add_argument("--old_target_list", "--old_pfs_list", "--pfs_old_list", dest="pfs_old_list", default=None,
                   help="既存のターゲットリスト rubin_targets.csv (既存天体の ob_code 番号を維持し、新規天体に続き番号を振る)")
    a = p.parse_args()

    from stamp_classifier import sn_pipeline
    if a.cmd == "freeze":
        sn_pipeline.freeze_bundle(a.version, a.step1_version, a.moving_min_sep, a.moving_min_rate)
    else:
        out = a.out or os.path.join("reports", f"sn_{a.db}_{a.start}_{a.end}")
        sn_pipeline.run(a.db, a.start, a.end, a.bundle, out, fits_root=a.fits_root, host=a.host, port=a.port,
                        user=a.user, password=a.password, include_uncertain=not a.strict,
                        keep_suspicious_visits=a.keep_suspicious_visits, per_object_csv=a.per_object_csv,
                        workers=a.workers, min_nights=a.min_nights, min_nights_no_host=a.min_nights_no_host,
                        include_negative_galaxy=not a.no_negative_galaxy,
                        pfs_semester=a.pfs_semester, pfs_old_list=a.pfs_old_list,
                        per_night_band=a.per_night_band)

if __name__ == "__main__":
    main()
