"""
Re-review (Relabel) of Existing Labels
これまでに付けた学習用ラベル・評価用ラベルを、もう一度目視で判定し直す。

- 対象 (target): 'train' (学習用), 'eval' (評価用), 'both'
- 判定リストは初回に固定シードで並べ替えて保存し、中断しても同じ順で再開する。
- 既定ではブラインド (以前のラベルもモデルの予測も表示しない)。--show_previous で以前のラベルを表示できる。
- [u] Unsure (判定不能) を選べる。
- 再判定の結果は別ファイルに保存し、relabel_apply で元のラベルファイルに反映する。
  反映時は、初回だけタイムスタンプ付きのバックアップを作り、以前のラベルを label_before_relabel 列に残す。
  Unsure のサンプルはラベルファイルから外し、relabel_step{N}_unsure.csv に移す。
  反映は何度実行しても結果は同じ (冪等)。
"""

import os
import shutil
import time
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import StepConfig, DATA_DIR
from .labeling_viewer import LabelingViewer, load_human_labels

def relabel_paths(step: int) -> Dict[str, str]:
    base = os.path.join(DATA_DIR, f"relabel_step{step}")
    return {
        "list": f"{base}_list.csv",
        "labels": f"{base}_labels.csv",
        "report": f"{base}_report.csv",
        "unsure": f"{base}_unsure.csv",
    }

def _target_files(step_config: StepConfig, target: str) -> Dict[str, str]:
    files = {}
    if target in ("train", "both"):
        files["train"] = step_config.labels_file
    if target in ("eval", "both"):
        if not step_config.eval_labels_file:
            raise ValueError(f"Step {step_config.step} has no evaluation label file configured.")
        files["eval"] = step_config.eval_labels_file
    return files

def build_relabel_list(step_config: StepConfig, target: str, seed: int = 20261020) -> pd.DataFrame:
    parts = []
    for name, path in _target_files(step_config, target).items():
        df = load_human_labels(path)
        if df.empty:
            print(f"[!] No {name} labels in {path}")
            continue
        parts.append(pd.DataFrame({
            "target": name,
            "diaSourceId": df["diaSourceId"].astype(np.int64).values,
            "diaObjectId": df["diaObjectId"].astype(np.int64).values,
            "prev_label": df["label"].astype(int).values,
            "prev_class": df["class_name"].values,
        }))
    if not parts:
        raise RuntimeError("No labels to re-review.")
    out = pd.concat(parts, ignore_index=True).drop_duplicates("diaSourceId")
    out = out.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["order"] = np.arange(len(out))
    return out

def run_relabel(step_config: StepConfig, source_meta: Dict[int, Dict], target: str = "both",
                show_previous: bool = False, interactive: bool = True) -> pd.DataFrame:
    """
    source_meta: diaSourceId -> {'midpointMjdTai', 'band'} (画像のパスを作るため)
    """
    paths = relabel_paths(step_config.step)
    if os.path.exists(paths["list"]):
        df_list = pd.read_csv(paths["list"])
        if set(df_list["target"]) != set(_target_files(step_config, target)):
            raise RuntimeError(
                f"An existing re-review list ({paths['list']}) covers targets {sorted(set(df_list['target']))}, "
                f"but --relabel_target {target} was given. Use the same target, or move the list away to start over.")
    else:
        df_list = build_relabel_list(step_config, target)
        os.makedirs(os.path.dirname(paths["list"]) or ".", exist_ok=True)
        df_list.to_csv(paths["list"], index=False)
        print(f"[+] Created re-review list of {len(df_list)} labels "
              f"({df_list['target'].value_counts().to_dict()}) -> {paths['list']}")

    done_df = load_human_labels(paths["labels"])
    done = set(done_df["diaSourceId"].astype(np.int64)) if not done_df.empty else set()
    todo = df_list[~df_list["diaSourceId"].isin(done)].sort_values("order")
    print(f"=== Re-review of labels (Step {step_config.step}): {len(done)} / {len(df_list)} done -> {paths['labels']} ===")
    print("    Blind review: previous labels are hidden." if not show_previous else "    Previous labels are shown.")

    viewer = LabelingViewer(step=step_config.step)
    viewer.labels_file = paths["labels"]
    n_total = len(df_list)
    n_done = len(done)
    for r in todo.itertuples():
        meta = source_meta.get(int(r.diaSourceId))
        if meta is None:
            print(f"[!] diaSourceId {r.diaSourceId} not in the embeddings cache; skipped.")
            continue
        rec = {"diaSourceId": int(r.diaSourceId), "diaObjectId": int(r.diaObjectId),
               "midpointMjdTai": float(meta["midpointMjdTai"]), "band": str(meta["band"])}
        note = f"Re-review {n_done + 1}/{n_total}"
        if show_previous:
            note += f"  |  previous ({r.target}) label: [{r.prev_class}]"
        print(f"\n[Re-review {n_done + 1}/{n_total}]")
        ans = viewer.review_sample(rec, pred_class=None, confidence=None, margin=None, interactive=interactive,
                                   selection=f"relabel_{r.target}", allow_unsure=True, title_note=note)
        if ans == "QUIT":
            print("[*] Re-review paused. Re-run the same command to resume.")
            break
        if ans is None or ans == "SKIP_EMPTY":
            continue
        n_done += 1

    return relabel_report(step_config)

def relabel_report(step_config: StepConfig, verbose: bool = True) -> pd.DataFrame:
    paths = relabel_paths(step_config.step)
    if not os.path.exists(paths["list"]):
        print("[!] No re-review list yet. Run --mode relabel first.")
        return pd.DataFrame()
    df_list = pd.read_csv(paths["list"])
    new = load_human_labels(paths["labels"])
    if new.empty:
        print("[!] No re-review results yet.")
        return pd.DataFrame()
    df = df_list.merge(new[["diaSourceId", "label", "class_name"]].rename(
        columns={"label": "new_label", "class_name": "new_class"}), on="diaSourceId", how="inner")
    df["unsure"] = df["new_label"] < 0
    df["changed"] = ~df["unsure"] & (df["new_label"] != df["prev_label"])
    df.sort_values(["changed", "unsure", "order"], ascending=[False, False, True]).to_csv(paths["report"], index=False)
    if verbose:
        print(f"\n=== Re-review summary (Step {step_config.step}): {len(df)} of {len(df_list)} re-reviewed ===")
        for t, g in df.groupby("target"):
            sure = g[~g["unsure"]]
            kappa = cohen_kappa_score(sure["prev_label"], sure["new_label"]) if len(sure) > 1 and sure["prev_label"].nunique() > 1 else np.nan
            print(f"  [{t}] n={len(g)}  agreement={1 - sure['changed'].mean() if len(sure) else np.nan:.3f}  "
                  f"changed={int(g['changed'].sum())}  unsure={int(g['unsure'].sum())}  kappa={kappa:.3f}")
            print("    previous (rows) vs new (cols):")
            print(pd.crosstab(g["prev_class"], g["new_class"]).to_string().replace("\n", "\n    ").join(["    ", ""]))
        print(f"  Per-sample report (changed first) -> {paths['report']}")
        print("  Apply with: --mode relabel_apply")
    return df

def apply_relabel(step_config: StepConfig) -> None:
    paths = relabel_paths(step_config.step)
    df = relabel_report(step_config, verbose=False)
    if df.empty:
        return
    stamp = time.strftime("%Y%m%d_%H%M%S")
    unsure_rows = []
    for target, path in _target_files(step_config, "both").items():
        g = df[df["target"] == target]
        if g.empty or not os.path.exists(path):
            continue
        lab = pd.read_csv(path)
        backup = f"{path}.bak_before_relabel"
        if not os.path.exists(backup):
            shutil.copy2(path, backup)
            print(f"[*] Backed up {path} -> {backup}")
        else:
            shutil.copy2(path, f"{path}.bak_{stamp}")
        if "label_before_relabel" not in lab.columns:
            lab["label_before_relabel"] = np.nan
            lab["class_name_before_relabel"] = None
            lab["relabeled"] = False
        sid = lab["diaSourceId"].astype(np.int64)
        res = dict(zip(g["diaSourceId"].astype(np.int64), zip(g["new_label"].astype(int), g["new_class"])))
        n_changed = 0
        drop = []
        for i in np.where(sid.isin(res.keys()))[0]:
            new_label, new_class = res[int(sid.iat[i])]
            if pd.isna(lab.at[i, "label_before_relabel"]):
                lab.at[i, "label_before_relabel"] = lab.at[i, "label"]
                lab.at[i, "class_name_before_relabel"] = lab.at[i, "class_name"]
            if new_label < 0:
                row = lab.loc[i].to_dict()
                row["target"] = target
                unsure_rows.append(row)
                drop.append(i)
                continue
            n_changed += int(new_label != int(lab.at[i, "label_before_relabel"]))
            lab.at[i, "label"] = new_label
            lab.at[i, "class_name"] = new_class
            lab.at[i, "relabeled"] = True
        lab = lab.drop(index=drop)
        lab.to_csv(path, index=False)
        print(f"[+] {target}: applied {int(lab['relabeled'].sum())} re-reviewed labels to {path} "
              f"({n_changed} changed, {len(drop)} unsure removed)")
        if target == "eval" and "label_orig" in lab.columns:
            print("    [!] This evaluation file has adjudication columns (label_orig). Do not run adjudicate_eval "
                  "afterwards, or it will overwrite the re-reviewed labels.")
    if unsure_rows:
        un = pd.DataFrame(unsure_rows)
        if os.path.exists(paths["unsure"]):
            un = pd.concat([pd.read_csv(paths["unsure"]), un]).drop_duplicates("diaSourceId", keep="last")
        un.to_csv(paths["unsure"], index=False)
        print(f"[+] {len(unsure_rows)} unsure samples moved to {paths['unsure']}")
