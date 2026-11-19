"""
Evaluation Label Adjudication
監査 (eval_audit) で元のラベルと再判定が一致しなかった評価サンプルについて、両方の判定を並べて表示し、
最終ラベルを決めて評価用ラベルに反映する。

- 裁定結果は {eval}_adjudicated.csv に保存 (中断・再開可)。
- 反映時は評価用ラベル CSV に label_orig / class_name_orig (最初の判定) と adjudicated 列を加え、
  label / class_name を最終ラベルに置き換える。初回反映前に .bak_before_adjudication を作成する。
- 反映は常に label_orig を起点にやり直すため、何度実行しても結果は同じ (冪等)。
"""

import os
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

def adjudication_paths(eval_labels_file: str) -> Dict[str, str]:
    base, _ = os.path.splitext(eval_labels_file)
    return {
        "adjudicated": f"{base}_adjudicated.csv",
        "backup": f"{eval_labels_file}.bak_before_adjudication",
    }

def disagreements(df_list: pd.DataFrame, df_audit: pd.DataFrame) -> pd.DataFrame:
    """監査リストと再判定から、不一致 (判定不能を含む) のサンプルを抽出"""
    a = df_audit[["diaSourceId", "label", "class_name"]].rename(columns={"label": "audit_label", "class_name": "audit_class"})
    df = df_list.merge(a, on="diaSourceId", how="inner")
    return df[df["audit_label"] != df["orig_label"]].sort_values("order").reset_index(drop=True)

def apply_adjudication(eval_labels_file: str, adjudicated_file: str, backup_file: str) -> pd.DataFrame:
    """裁定結果を評価用ラベルに反映 (冪等)"""
    if not os.path.exists(adjudicated_file):
        print("[!] No adjudication results to apply.")
        return pd.read_csv(eval_labels_file)
    if not os.path.exists(backup_file):
        shutil.copy2(eval_labels_file, backup_file)
        print(f"[*] Backed up evaluation labels -> {backup_file}")

    df = pd.read_csv(eval_labels_file)
    if "label_orig" not in df.columns:
        df["label_orig"] = df["label"]
        df["class_name_orig"] = df["class_name"]
    # 新たに追加された評価ラベル (label_orig が空) は現在のラベルを最初の判定とする
    miss = df["label_orig"].isna()
    df.loc[miss, "label_orig"] = df.loc[miss, "label"]
    df.loc[miss, "class_name_orig"] = df.loc[miss, "class_name"]
    df["label_orig"] = df["label_orig"].astype(int)

    # 起点 (最初の判定) に戻してから裁定結果を上書き
    df["label"] = df["label_orig"]
    df["class_name"] = df["class_name_orig"]
    df["adjudicated"] = False

    adj = pd.read_csv(adjudicated_file)
    adj = adj[adj["label"] >= 0]
    final = dict(zip(adj["diaSourceId"].astype(np.int64), zip(adj["label"].astype(int), adj["class_name"])))
    hit = df["diaSourceId"].astype(np.int64).map(lambda s: s in final)
    for i in np.where(hit)[0]:
        lab, cname = final[int(df.at[i, "diaSourceId"])]
        df.at[i, "label"] = lab
        df.at[i, "class_name"] = cname
        df.at[i, "adjudicated"] = True
    df.to_csv(eval_labels_file, index=False)

    changed = df[df["adjudicated"] & (df["label"] != df["label_orig"])]
    print(f"[+] Applied {int(df['adjudicated'].sum())} adjudicated labels to {eval_labels_file} "
          f"({len(changed)} changed from the original label).")
    if len(changed):
        print(pd.crosstab(changed["class_name_orig"], changed["class_name"]).to_string())
    print("Evaluation set class distribution (after adjudication):")
    print(df["class_name"].value_counts().to_string())
    return df
