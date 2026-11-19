#!/bin/bash
# XMM-LSS 領域の Rubin アラートを取得し、超新星候補を選んで PFS ターゲットリストを更新する (pgx 側の毎日の処理)。
#
#   ./daily_targets.sh [DATE]        DATE = Fink の転送日 (YYYY-MM-DD, 省略時は前日)
#
# Fink の日付は UTC 日付で、DATE に入るのは観測夜 dayObs = DATE-1 のデータ
# (例: DATE=2026-11-05 の転送には 2026-11-04 の夜のアラートが入っている)。
# 候補抽出は dayObs で NIGHT-WINDOW_DAYS 〜 NIGHT (NIGHT = DATE-1)。
#
# 処理の流れ (Analysis_Procedure.txt の pgx 側の手順):
#   1. datatransfer.yml の日付を DATE にして Fink ポータルにジョブを投入する (fink_submit.py)
#      ポータルの推定アラート数が 0、または領域内のアラートが 0 件なら、ここで終了する
#   2. finkctl transfer で受信する (Lag が 0 になるまで繰り返す)
#   3. avro_to_fits_pg.py で DB に取り込む
#   4. find_sn_candidates.py で NIGHT-WINDOW_DAYS 〜 NIGHT の候補を選ぶ (ob_code は台帳から引き継ぐ)
#   5. 台帳 (ob_registry.py) に新規天体を追加し、3 連画像の PDF を作る
#   6. S26B-047 の rubin_targets.csv を置き換えて concatenate_targets.py を実行する
#
# 同じ DATE で再実行すると、受信済み (ftransfer_xmm_lss_DATE/.transfer_done がある) なら 1-2 は飛ばす。
# 設定は下の変数か、同名の環境変数で変えられる。ログは logs/daily/DATE.log。
set -euo pipefail

BASE=/mnt/ugnas/work/LSST/fink2
cd "$BASE"

DATE=${1:-$(date -d yesterday +%F)}
date -d "$DATE" >/dev/null 2>&1 || { echo "bad date: $DATE" >&2; exit 2; }
DATE=$(date -d "$DATE" +%F)

WINDOW_DAYS=${WINDOW_DAYS:-14}   # 2026-11-09: 7 -> 14 (10-29, 10-30 の検出を観測夜まで残す)
DB=${DB:-xmm_lss_alert}
BUNDLE=${BUNDLE:-snp_v1}
SEMESTER=${SEMESTER:-s26b}
REGISTRY=${REGISTRY:-reports/ob_registry_${SEMESTER}.csv}
PROPOSAL_DIR=${PROPOSAL_DIR:-/mnt/ugnas/work/LSST/PFS/S26B-047}
PW_PYTHON=${PW_PYTHON:-/mnt/ugnas/work/LSST/fink/bin/python}   # Playwright 入りの python
TRANSFER_TRIES=${TRANSFER_TRIES:-10}
NOTOPIC_TRIES=${NOTOPIC_TRIES:-3}     # topic が無いまま続いたら 0 件とみなす回数

NIGHT=$(date -d "$DATE - 1 day" +%F)
START=$(date -d "$NIGHT - $WINDOW_DAYS days" +%F)
SRC=ftransfer_xmm_lss_$DATE
OUT=reports/sn_${DB}_${START}_${NIGHT}
LOGDIR=logs/daily/$DATE
mkdir -p "$LOGDIR"
exec > >(tee -a "logs/daily/$DATE.log") 2>&1

exec 9>logs/daily/.lock
flock -n 9 || { echo "another daily_targets.sh is running" >&2; exit 4; }

step() { echo; echo "=== $(date '+%F %T') $* ==="; }
step "daily_targets DATE=$DATE (dayObs $START .. $NIGHT) OUT=$OUT"

[ -f "$REGISTRY" ] || { echo "ob_code 台帳がありません: $REGISTRY" >&2; exit 2; }

# ---- 1-2. Fink から取得 ----------------------------------------------------------
if [ -f "$SRC/.transfer_done" ]; then
    step "skip transfer: $SRC/.transfer_done ($(cat "$SRC/.transfer_done"))"
else
    step "1. submit Fink job"
    YML=$LOGDIR/datatransfer.yml
    sed -E "s/^(  startdate: ).*/\1'$DATE'/; s/^(  stopdate: ).*/\1'$DATE'/" datatransfer.yml > "$YML"
    grep -q "startdate: '$DATE'" "$YML" && grep -q "stopdate: '$DATE'" "$YML" \
        || { echo "datatransfer.yml の日付を書き換えられません" >&2; exit 1; }

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

# ---- 4. 候補抽出 -----------------------------------------------------------------
step "4. find_sn_candidates.py $START .. $NIGHT"
./bin/python find_sn_candidates.py run --db "$DB" --start "$START" --end "$NIGHT" \
    --bundle "$BUNDLE" --semester "$SEMESTER" --old_target_list "$REGISTRY" --out "$OUT"

# diaObjectId が無い (obj_id=0) 行は天体を特定できず、ob_code も安定しないので除く
./bin/python - "$OUT/rubin_targets.csv" <<'EOF'
import sys, pandas as pd
p = sys.argv[1]
df = pd.read_csv(p, dtype={"ob_code": str})
bad = df["obj_id"] <= 0
if bad.any():
    print(f"drop {bad.sum()} rows with obj_id<=0:", df.loc[bad, ["ob_code", "ra", "dec"]].values.tolist())
    df[~bad].to_csv(p, index=False, na_rep="")
print(f"{p}: {(~bad).sum()} targets")
EOF

# ---- 5. 台帳更新と PDF -----------------------------------------------------------
step "5. update ob_code registry and PDF"
./bin/python ob_registry.py update --registry "$REGISTRY" --targets "$OUT/rubin_targets.csv" --date "$DATE"
./bin/python make_pfs_targets.py pdf --targets "$OUT/rubin_targets.csv"
# 前回 (latest が指していたレポート) との差分: diff_added / diff_removed の CSV と 3 連画像 PDF
# 比較相手は、期間の終わりが NIGHT より前のレポートのうち最新のもの (同じ DATE で再実行しても前日と比べる)
PREV=""
for d in reports/sn_${DB}_????-??-??_????-??-??; do
    [ -f "$d/rubin_targets.csv" ] || continue
    end=${d##*_}
    if [[ "$end" < "$NIGHT" ]] && { [ -z "$PREV" ] || [[ "$end" > "${PREV##*_}" ]] || \
         { [ "$end" = "${PREV##*_}" ] && [[ "$d" > "$PREV" ]]; }; }; then
        PREV=$d
    fi
done
if [ -n "$PREV" ]; then
    ./bin/python diff_targets.py --old "$PREV" --new "$OUT" || echo "WARNING: diff_targets.py failed"
else
    echo "diff: no previous report to compare"
fi
ln -sfn "$(basename "$OUT")" reports/sn_${DB}_latest

# ---- 6. 他のターゲットと統合 -----------------------------------------------------
step "6. concatenate_targets.py in $PROPOSAL_DIR"
if [ -f "$PROPOSAL_DIR/rubin_targets.csv" ]; then
    mkdir -p "$PROPOSAL_DIR/rubin_targets_history"
    # 同じ DATE で再実行したときは、最初の実行で取ったバックアップ (前日のリスト) を残す
    BK=$PROPOSAL_DIR/rubin_targets_history/rubin_targets_before_$DATE.csv
    if [ -f "$BK" ]; then
        echo "keep existing backup $BK (rerun)"
    else
        cp -p "$PROPOSAL_DIR/rubin_targets.csv" "$BK"
    fi
fi
cp "$OUT/rubin_targets.csv" "$PROPOSAL_DIR/rubin_targets.csv"
(cd "$PROPOSAL_DIR" && ../.venv/bin/python concatenate_targets.py)

step "done: $OUT/rubin_targets.csv -> $PROPOSAL_DIR/targets_all.csv"
