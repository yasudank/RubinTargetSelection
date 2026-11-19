#!/bin/bash
# PFS 観測準備を pgx から 1 コマンドで実行する。
#   1. ssh で galuda の prepare_observation.sh を実行 (視野最適化 -> Netflow -> スケジュール -> OPE)
#   2. pgx でファインディングチャート (export_pfs_design_pdf.py) を作る
#
#   ./prepare_observation.sh [galuda 側のオプション ...]
#
# 例:
#   ./prepare_observation.sh                                  # 既定 (S26B-047_YYYYMMDD に出力)
#   ./prepare_observation.sh --obstime 2026-11-10T10:00:00Z --nights "2026-11-09 2026-11-10 2026-11-11"
#   ./prepare_observation.sh --pointings xmm_lss/xmm_lss_centers_s26b_2pat.ecsv   # 視野は固定
# オプションの一覧: ssh galuda /mnt/ugnas/work/PFS/galuda/NetflowByUser/prepare_observation.sh --help
#
# 入力は /mnt/ugnas/work/LSST/PFS/S26B-047/targets_all.csv (daily_targets.sh が毎日更新)。
# 候補リストは reports/sn_xmm_lss_alert_latest (daily_targets.sh の最新結果)。
set -euo pipefail

FINK=/mnt/ugnas/work/LSST/fink2
NETFLOW=/mnt/ugnas/work/PFS/galuda/NetflowByUser
GALUDA=${GALUDA:-galuda}
# cron 用の鍵 (パスフレーズなし、galuda 側で prepare_observation.sh だけに制限) があれば使う
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_galuda_cron}
SSH_OPTS=(-o BatchMode=yes -o ServerAliveInterval=60)
[ -f "$SSH_KEY" ] && SSH_OPTS+=(-i "$SSH_KEY" -o IdentitiesOnly=yes)
cd "$FINK"

REPORT=$(readlink -f reports/sn_xmm_lss_alert_latest)
[ -d "$REPORT" ] || { echo "候補リストがありません: reports/sn_xmm_lss_alert_latest" >&2; exit 2; }

mkdir -p logs/prepare
LOG=logs/prepare/$(date +%Y%m%d_%H%M%S).log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') prepare_observation (report: $REPORT) ==="

# 引数を 1 つずつクォートして渡す (--nights "D1 D2" のような空白入りを保つ)
# 引数なしで printf ' %q' を使うと " ''" (空引数) になり ssh_gate に拒否されるので分ける
REMOTE_ARGS=""
if [ $# -gt 0 ]; then REMOTE_ARGS=$(printf ' %q' "$@"); fi
ssh "${SSH_OPTS[@]}" "$GALUDA" \
    "cd $NETFLOW && ./prepare_observation.sh$REMOTE_ARGS" | tee logs/prepare/.last_remote
RUN_DIR=$(sed -n 's/^=== .* done: //p' logs/prepare/.last_remote | sed 's/ ===$//' | tail -1)
[ -n "$RUN_DIR" ] && [ -d "$RUN_DIR/design" ] || { echo "galuda 側が完了していません" >&2; exit 1; }

echo; echo "=== $(date '+%F %T') export_pfs_design_pdf.py ==="
./bin/python export_pfs_design_pdf.py --report_dir "$REPORT" \
    --design_dir "$RUN_DIR/design" --out_dir "$RUN_DIR/design_cutouts"

echo; echo "=== $(date '+%F %T') done ==="
echo "run dir:  $RUN_DIR"
echo "OPE:      $(ls "$RUN_DIR"/ope/*_scheduled.ope)"
echo "charts:   $RUN_DIR/design_cutouts"
