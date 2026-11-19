"""
Evaluation Label Audit
評価用セットのラベルの揺らぎ (判定者の自己一致率) を、ブラインド再判定で測る。

- 監査対象: 現在のモデルが誤った評価サンプル全件 + 正解した評価サンプルからランダム抽出した n_correct 件。
  2 群を混ぜて並べ替えるため、判定時にどちらの群かは分からない。元のラベルもモデル予測も表示しない。
- 監査リストは初回に固定保存する (後でモデルが変わっても対象は変わらない。中断後は同じ順序で再開)。
- 集計では、群の抽出率 (誤り群 = 全件, 正解群 = n_correct / 正解総数) と評価用セットの層の重みで
  重み付けし、評価用セット全体 / 母集団に対する自己一致率を推定する。
"""

import os
from typing import Dict, Optional, List

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .eval_strata import stratum_of

def audit_paths(eval_labels_file: str) -> Dict[str, str]:
    base, _ = os.path.splitext(eval_labels_file)
    return {
        "list": f"{base}_audit_list.csv",
        "labels": f"{base}_audit_labels.csv",
        "report": f"{base}_audit_report.csv",
    }

def create_audit_list(list_file: str, df_eval_rows: pd.DataFrame, model_pred: np.ndarray,
                      n_correct: int = 50, seed: int = 20261019) -> pd.DataFrame:
    """誤り全件 + 正解からランダム n_correct 件の監査リストを作成して固定保存"""
    df = df_eval_rows[["diaSourceId", "diaObjectId", "label", "class_name"]].copy().reset_index(drop=True)
    df = df.rename(columns={"label": "orig_label", "class_name": "orig_class"})
    df["model_pred"] = np.asarray(model_pred).astype(int)
    df["group"] = np.where(df["model_pred"] == df["orig_label"], "correct", "error")
    n_err_total = int((df["group"] == "error").sum())
    n_cor_total = int((df["group"] == "correct").sum())

    rng = np.random.default_rng(seed)
    err = df[df["group"] == "error"]
    cor = df[df["group"] == "correct"]
    cor = cor.iloc[rng.choice(len(cor), size=min(n_correct, len(cor)), replace=False)]
    out = pd.concat([err, cor]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    out["group_total"] = np.where(out["group"] == "error", n_err_total, n_cor_total)
    out["group_sampled"] = np.where(out["group"] == "error", len(err), len(cor))
    out["order"] = np.arange(len(out))
    os.makedirs(os.path.dirname(list_file) or ".", exist_ok=True)
    out.to_csv(list_file, index=False)
    print(f"[+] Created audit list: {len(err)} model errors (all) + {len(cor)} of {n_cor_total} correct (random) -> {list_file}")
    return out

def audit_report(df_list: pd.DataFrame, df_audit: pd.DataFrame, classes: List[str],
                 strata: Optional[Dict] = None, eval_source_ids: Optional[np.ndarray] = None,
                 report_file: Optional[str] = None, verbose: bool = True) -> Dict:
    """監査結果を集計 (自己一致率, κ, 判定不能率, 覆ったサンプル一覧, 再判定ラベルに対するモデル精度)"""
    if df_audit is None or df_audit.empty:
        print("[!] No audit labels yet.")
        return {}
    a = df_audit[["diaSourceId", "label", "class_name"]].rename(columns={"label": "audit_label", "class_name": "audit_class"})
    df = df_list.merge(a, on="diaSourceId", how="inner")
    if df.empty:
        print("[!] No audit labels match the audit list.")
        return {}
    df["unsure"] = df["audit_label"] < 0
    df["agree"] = (~df["unsure"]) & (df["audit_label"] == df["orig_label"])
    df["model_vs_audit"] = (~df["unsure"]) & (df["audit_label"] == df["model_pred"])

    # 重み: 群の抽出率の逆数 (評価用セット内) × 層の重み (母集団)
    n_done_group = df.groupby("group")["diaSourceId"].transform("size")
    df["w_eval"] = df["group_total"] / n_done_group
    if strata is not None and eval_source_ids is not None:
        # 層の重み N_h / n_h (n_h は評価用セット全体での層内ラベル数)
        h_eval = stratum_of(strata, eval_source_ids)
        n_h = np.bincount(h_eval[h_eval >= 0], minlength=len(strata["sizes"])).astype(float)
        with np.errstate(divide="ignore", invalid="ignore"):
            w_h = np.where(n_h > 0, strata["sizes"] / n_h, 0.0)
        h = stratum_of(strata, df["diaSourceId"].values)
        df["stratum"] = h
        df["w_pop"] = np.where(h >= 0, df["w_eval"] * w_h[np.maximum(h, 0)], 0.0)

    def wmean(x, w):
        w = np.asarray(w, dtype=float)
        return float(np.sum(np.asarray(x, dtype=float) * w) / np.sum(w)) if np.sum(w) > 0 else np.nan

    res = {"n_audited": int(len(df)), "n_unsure": int(df["unsure"].sum())}
    for g in ["error", "correct"]:
        d = df[df["group"] == g]
        sure = d[~d["unsure"]]
        res[f"{g}_n"] = int(len(d))
        res[f"{g}_unsure_rate"] = float(d["unsure"].mean()) if len(d) else np.nan
        res[f"{g}_agree_rate"] = float(sure["agree"].mean()) if len(sure) else np.nan
        if g == "error":
            # 誤り群で、再判定がモデルの予測と一致した割合 (= 元のラベルの方が誤りだった可能性)
            res["error_flipped_to_model_rate"] = float(sure["model_vs_audit"].mean()) if len(sure) else np.nan

    sure = df[~df["unsure"]]
    res["eval_unsure_rate_w"] = wmean(df["unsure"], df["w_eval"])
    res["eval_self_agreement_w"] = wmean(sure["agree"], sure["w_eval"])
    res["eval_model_acc_vs_orig_w"] = wmean(sure["model_pred"] == sure["orig_label"], sure["w_eval"])
    res["eval_model_acc_vs_audit_w"] = wmean(sure["model_vs_audit"], sure["w_eval"])
    if "w_pop" in df:
        res["pop_self_agreement_w"] = wmean(sure["agree"], sure["w_pop"])
        res["pop_model_acc_vs_audit_w"] = wmean(sure["model_vs_audit"], sure["w_pop"])
    res["kappa_unweighted"] = float(cohen_kappa_score(sure["orig_label"], sure["audit_label"])) if len(sure) > 1 else np.nan

    if report_file:
        cols = ["order", "diaSourceId", "diaObjectId", "group", "orig_class", "model_pred", "audit_class", "agree", "unsure"]
        out = df.copy()
        out["model_pred"] = out["model_pred"].map(lambda k: classes[int(k)])
        out[cols].sort_values(["agree", "group", "order"]).to_csv(report_file, index=False)
        res["report_file"] = report_file

    if verbose:
        print(f"\n=== Evaluation Label Audit: {res['n_audited']} re-reviewed ({res['n_unsure']} unsure) ===")
        for g, name in [("error", "Model errors"), ("correct", "Model correct (random)")]:
            if res.get(f"{g}_n"):
                line = (f"  {name:24s} n={res[f'{g}_n']:3d}  self-agreement={res[f'{g}_agree_rate']:.3f}  "
                        f"unsure={res[f'{g}_unsure_rate']:.3f}")
                if g == "error":
                    line += f"  re-label == model prediction: {res['error_flipped_to_model_rate']:.3f}"
                print(line)
        print(f"  Estimated over the whole evaluation set (group-weighted):")
        print(f"    self-agreement      = {res['eval_self_agreement_w']:.3f}  (unsure rate {res['eval_unsure_rate_w']:.3f})")
        print(f"    model acc vs orig   = {res['eval_model_acc_vs_orig_w']:.3f}")
        print(f"    model acc vs re-lab = {res['eval_model_acc_vs_audit_w']:.3f}")
        if "pop_self_agreement_w" in res:
            print(f"  Estimated over the population (strata-weighted):")
            print(f"    self-agreement      = {res['pop_self_agreement_w']:.3f}")
            print(f"    model acc vs re-lab = {res['pop_model_acc_vs_audit_w']:.3f}")
        print(f"  Cohen's kappa (orig vs re-label, audited samples, unweighted) = {res['kappa_unweighted']:.3f}")
        if len(df):
            print("  Original label (rows) vs re-label (cols), audited samples (counts):")
            ct = pd.crosstab(df["orig_class"], df["audit_class"])
            print(ct.to_string().replace("\n", "\n    ").join(["    ", ""]))
        if report_file:
            print(f"  Per-sample report (disagreements first) -> {report_file}")
    return res
