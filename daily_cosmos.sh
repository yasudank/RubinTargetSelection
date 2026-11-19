#!/bin/bash
# COSMOS 領域の Rubin アラートを Fink から取得し、cosmos_alert データベースに登録する (毎日の処理)。
# daily_targets.sh の 1-3 (取得・受信・DB 取り込み) だけを COSMOS 用の設定で行う。候補抽出はしない。
#
#   ./daily_cosmos.sh [DATE]        DATE = Fink の転送日 (YYYY-MM-DD, 省略時は前日)
#
# Fink の日付は UTC 日付で、DATE に入るのは観測夜 dayObs = DATE-1 のデータ。
#   1. datatransfer_cosmos.yml の日付を DATE にして Fink ポータルにジョブを投入する (fink_submit.py)
#      ポータルの推定アラート数が 0、または領域内のアラートが 0 件なら、ここで終了する
#   2. finkctl transfer で ftransfer_cosmos_DATE/ に受信する (Lag が 0 になるまで繰り返す)
#   3. avro_to_fits_pg.py で cosmos_alert に取り込む (カットアウトは cutouts/ の下)
#
# 同じ DATE で再実行すると、受信済み (ftransfer_cosmos_DATE/.transfer_done がある) なら 1-2 は飛ばす。
# 設定は下の変数か、同名の環境変数で変えられる。ログは logs/cosmos/DATE.log。
set -euo pipefail

BASE=/mnt/ugnas/work/LSST/fink2
cd "$BASE"

DATE=${1:-$(date -d yesterday +%F)}
date -d "$DATE" >/dev/null 2>&1 || { echo "bad date: $DATE" >&2; exit 2; }
DATE=$(date -d "$DATE" +%F)

YML_TEMPLATE=${YML_TEMPLATE:-datatransfer_cosmos.yml}
DB=${DB:-cosmos_alert}
PW_PYTHON=${PW_PYTHON:-/mnt/ugnas/work/LSST/fink/bin/python}   # Playwright 入りの python
TRANSFER_TRIES=${TRANSFER_TRIES:-10}
NOTOPIC_TRIES=${NOTOPIC_TRIES:-3}     # topic が無いまま続いたら 0 件とみなす回数

SRC=ftransfer_cosmos_$DATE
LOGDIR=logs/cosmos/$DATE
mkdir -p "$LOGDIR"
exec > >(tee -a "logs/cosmos/$DATE.log") 2>&1

exec 9>logs/cosmos/.lock
flock -n 9 || { echo "another daily_cosmos.sh is running" >&2; exit 4; }

step() { echo; echo "=== $(date '+%F %T') $* ==="; }
step "daily_cosmos DATE=$DATE (dayObs $(date -d "$DATE - 1 day" +%F)) -> $SRC -> $DB"

# ---- 1-2. Fink から取得 ----------------------------------------------------------
if [ -f "$SRC/.transfer_done" ]; then
    step "skip transfer: $SRC/.transfer_done ($(cat "$SRC/.transfer_done"))"
else
    step "1. submit Fink job ($YML_TEMPLATE)"
    YML=$LOGDIR/datatransfer.yml
    sed -E "s/^(  startdate: ).*/\1'$DATE'/; s/^(  stopdate: ).*/\1'$DATE'/" "$YML_TEMPLATE" > "$YML"
    grep -q "startdate: '$DATE'" "$YML" && grep -q "stopdate: '$DATE'" "$YML" \
        || { echo "$YML_TEMPLATE の日付を書き換えられません" >&2; exit 1; }

    rc=0
    TOPIC=$("$PW_PYTHON" fink_submit.py "$YML" --shot_dir "$LOGDIR" | tail -1) || rc=$?
    if [ $rc -eq 3 ]; then
        step "no alerts on $DATE (portal estimate = 0); nothing to do"
        exit 0
    elif [ $rc -ne 0 ] || [[ ! "$TOPIC" =~ ^ftransfer_ ]]; then
        echo "fink_submit.py failed (rc=$rc, topic='$TOPIC'); see $LOGDIR/portal_error.*" >&2
        exit 1
    fi
    echo "topic: $TOPIC"

    step "2. finkctl transfer -> $SRC"
    lag=-1; notopic=0
    for i in $(seq 1 "$TRANSFER_TRIES"); do
        out=$(./bin/finkctl transfer -survey lsst -topic "$TOPIC" -outdir "$SRC" \
                --dump_schemas --verbose -outformat avro 2>&1) || true
        echo "$out" | tail -20
        lag=$(echo "$out" | awk -v t="$TOPIC" '$1=="Total" && $3==t {print $NF}' | tail -1)
        if [ "${lag:-x}" = "0" ]; then break; fi
        # 領域内のアラートが 0 件だと Fink は topic を作らず、finkctl は UNKNOWN_TOPIC_OR_PART で失敗する。
        # ジョブは "Data available" で終わっているので、NOTOPIC_TRIES 回続いたら 0 件として扱う。
        if echo "$out" | grep -q UNKNOWN_TOPIC_OR_PART; then
            notopic=$((notopic + 1))
            if [ "$notopic" -ge "$NOTOPIC_TRIES" ]; then
                echo "topic $TOPIC does not exist ($notopic tries): no alerts in the field"
                lag=0; TOPIC="$TOPIC (no topic: 0 alerts)"; break
            fi
        fi
        echo "transfer not complete (try $i/$TRANSFER_TRIES, lag='${lag}'); retry in 60 s"
        sleep 60
    done
    [ "${lag:-x}" = "0" ] || { echo "transfer did not finish" >&2; exit 1; }
    mkdir -p "$SRC"
    echo "$TOPIC $(date '+%F %T')" > "$SRC/.transfer_done"
fi

NFILES=$(find "$SRC" -maxdepth 1 -name '*.avro' | wc -l)
echo "$SRC: $NFILES avro files"
if [ "$NFILES" -eq 0 ]; then
    step "no alerts in the field on $DATE; nothing to do"
    exit 0
fi

# ---- 3. DB 取り込み --------------------------------------------------------------
step "3. avro_to_fits_pg.py -> $DB"
./bin/python avro_to_fits_pg.py --source_dir "$SRC" --dbname "$DB"

step "done: $SRC -> $DB"
