#!/usr/bin/env python
"""
PFS ターゲットの ob_code (rubin_{semester}_XXXX) と diaObjectId (obj_id) の対応を保持する台帳。

一度付けた ob_code は台帳に残り続けるので、ターゲットリストを作る期間を変えても
同じ天体には同じ番号が付く。find_sn_candidates.py run の --old_target_list に台帳を渡すと、
既存天体は台帳の番号、新規天体は台帳の最大番号の続きが付く。

1) 既存のターゲットリストから台帳を作る (最初に 1 回):
   bin/python ob_registry.py init --registry reports/ob_registry_s26b.csv \
       reports/sn_xmm_lss_merged/rubin_targets.csv reports/sn_xmm_lss_alert_*/rubin_targets.csv

2) 新しく作ったターゲットリストの新規天体を台帳に追加する:
   bin/python ob_registry.py update --registry reports/ob_registry_s26b.csv \
       --targets reports/sn_xmm_lss_alert_2026-10-29_2026-11-05/rubin_targets.csv --date 2026-11-05

台帳の列: ob_code, obj_id, ra, dec, first_seen (台帳に初めて載った処理日)
同じ obj_id に別の ob_code が付いている、または同じ ob_code が別の obj_id に付いている場合はエラーで止まる。
"""

import argparse
import os
import sys

import pandas as pd

COLUMNS = ["ob_code", "obj_id", "ra", "dec", "first_seen"]


def _read_targets(path):
    """obj_id <= 0 (diaObjectId が無いソース) は天体を特定できないので台帳には載せない"""
    df = pd.read_csv(path, dtype={"obj_id": "int64", "ob_code": str})
    bad = df["obj_id"] <= 0
    for r in df[bad].itertuples():
        print(f"  skip (obj_id={r.obj_id}): {r.ob_code} ra={r.ra:.5f} dec={r.dec:.5f}  [{path}]")
    return df.loc[~bad, ["ob_code", "obj_id", "ra", "dec"]].copy()


def _check_consistent(reg):
    errors = []
    for col, other in (("obj_id", "ob_code"), ("ob_code", "obj_id")):
        n = reg.groupby(col)[other].nunique()
        for key in n[n > 1].index:
            vals = sorted(map(str, reg.loc[reg[col] == key, other].unique()))
            errors.append(f"{col}={key} に複数の {other}: {', '.join(vals)}")
    return errors


def _write(reg, path):
    reg = reg.copy()
    reg["_seq"] = reg["ob_code"].str.extract(r"_(\d+)$")[0].astype(float)
    reg = reg.sort_values(["_seq", "ob_code"]).drop(columns="_seq")
    tmp = path + ".tmp"
    reg[COLUMNS].to_csv(tmp, index=False)
    os.replace(tmp, path)


def cmd_init(a):
    if os.path.exists(a.registry) and not a.force:
        sys.exit(f"{a.registry} は既にあります (上書きするなら --force)")
    parts = []
    for p in a.targets:
        df = _read_targets(p)
        df["first_seen"] = a.date or ""
        parts.append(df)
        print(f"{p}: {len(df)} rows")
    reg = pd.concat(parts, ignore_index=True)
    errors = _check_consistent(reg)
    if errors:
        sys.exit("番号が食い違っています:\n  " + "\n  ".join(errors))
    reg = reg.drop_duplicates(subset=["ob_code", "obj_id"], keep="first")
    os.makedirs(os.path.dirname(os.path.abspath(a.registry)), exist_ok=True)
    _write(reg, a.registry)
    print(f"Registry: {len(reg)} objects -> {a.registry}")


def cmd_update(a):
    if not os.path.exists(a.registry):
        sys.exit(f"台帳がありません: {a.registry}")
    reg = pd.read_csv(a.registry, dtype={"obj_id": "int64", "ob_code": str, "first_seen": str})
    new = _read_targets(a.targets)
    new["first_seen"] = a.date
    combined = pd.concat([reg, new], ignore_index=True)
    errors = _check_consistent(combined)
    if errors:
        sys.exit("台帳と新しいリストで番号が食い違っています:\n  " + "\n  ".join(errors))
    added = new[~new["obj_id"].isin(reg["obj_id"])]
    if len(added):
        _write(pd.concat([reg, added], ignore_index=True), a.registry)
    print(f"Registry: {len(reg)} + {len(added)} new = {len(reg) + len(added)} objects -> {a.registry}")
    for r in added.itertuples():
        print(f"  new: {r.ob_code} {r.obj_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("init", help="既存のターゲットリストから台帳を作る")
    i.add_argument("--registry", required=True)
    i.add_argument("--date", default=None, help="first_seen に入れる日付 (省略時は空)")
    i.add_argument("--force", action="store_true")
    i.add_argument("targets", nargs="+", help="rubin_targets.csv (複数可)")
    u = sub.add_parser("update", help="ターゲットリストの新規天体を台帳に追加する")
    u.add_argument("--registry", required=True)
    u.add_argument("--targets", required=True)
    u.add_argument("--date", required=True)
    a = p.parse_args()
    cmd_init(a) if a.cmd == "init" else cmd_update(a)


if __name__ == "__main__":
    main()
