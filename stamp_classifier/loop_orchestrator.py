"""
Active Learning & Reinforcement Loop Orchestrator
Coordinates active sampling, human visual labeling, Supervised Contrastive reward fine-tuning, and Random Forest retraining.
"""

import os
import time
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import StepConfig, STEP1_CONFIG, STEP2_CONFIG, ActiveLearningConfig, DATA_DIR, FITS_ROOT
from .feature_extractor import load_cached_embeddings, extract_embeddings
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report

from .rf_classifier import StampRandomForest
from .active_sampler import sample_candidates_for_labeling, valid_embedding_mask
from .eval_strata import strata_file_path, load_strata, create_strata, stratum_of, print_strata_summary, weighted_metrics
from .labeling_viewer import LabelingViewer, load_human_labels
from .eval_audit import audit_paths, create_audit_list, audit_report
from .eval_adjudication import adjudication_paths, disagreements, apply_adjudication
from .eval_curves import targets_for, threshold_curves, plot_threshold_curves
from .model_registry import save_frozen_model, decide
from .models import SimCLRModel
from .losses import SupConLoss
from .dataset import load_catalog_from_db, PostageStampDataset, DBConfig
from .train_supervised import train_supervised

def fine_tune_encoder_with_supcon(
    step: int,
    df_cat: pd.DataFrame,
    df_labels: pd.DataFrame,
    epochs: int = 5,
    lr: float = 1e-4
) -> str:
    """
    目視判定ラベルを用いて、SimCLR エンコーダを教師あり対照学習 (SupCon) で微調整。
    同一クラスの潜在表現を強く引き寄せ、誤判定した境界事例を正解クラスタへ是正する。
    """
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    label_map = dict(zip(df_labels["diaSourceId"], df_labels["label"]))
    labeled_cat = df_cat[df_cat["diaSourceId"].isin(label_map.keys())].copy()
    
    if len(labeled_cat) < 10 or len(df_labels["label"].unique()) < 2:
        print("[!] Not enough labeled samples for SupCon fine-tuning yet (need >= 10 across 2+ classes). Skipping.")
        return None

    print(f"=== Fine-tuning Encoder with SupCon on {len(labeled_cat)} Human-labeled Stamps ===")
    
    dataset = PostageStampDataset(
        df_records=labeled_cat,
        step_config=step_config,
        fits_root=FITS_ROOT,
        mode="labeled",
        labeled_data=label_map
    )
    # 端数バッチが 1 件になると BatchNorm1d が学習時エラーになるため、端数は捨てる
    dataloader = DataLoader(dataset, batch_size=min(64, len(dataset)), shuffle=True, drop_last=(len(dataset) > 64))

    ckpt_path = os.path.join(step_config.checkpoint_dir, "best_simclr_encoder.pt")
    model = SimCLRModel(in_channels=step_config.in_channels, latent_dim=128, proj_dim=64).to(device)
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = SupConLoss(temperature=0.07)

    model.train()
    for ep in range(1, epochs + 1):
        tot_loss = 0.0
        n_b = 0
        for x, y, _ in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            z = model(x)
            loss = criterion(z, y)
            if loss.requires_grad and loss.item() > 0:
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
                n_b += 1
        avg_loss = tot_loss / max(n_b, 1)
        print(f"  SupCon Epoch {ep}/{epochs}: Loss = {avg_loss:.4f}")

    # 微調整後のエンコーダを上書き保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "encoder_state_dict": model.encoder.state_dict(),
        "finetuned": True
    }, ckpt_path)
    print(f"[+] Encoder fine-tuned and updated at {ckpt_path}")
    return ckpt_path

def generate_latent_space_animation(step: int, target_duration_sec: float = 10.0) -> Optional[str]:
    """
    pca_history/ 内の全スナップショット画像をつなぎ合わせて、
    強化学習・微調整による潜在空間クラスタの分離進化を表現する
    約10秒のプレゼンテーション用アニメーションGIFを自動生成。
    """
    try:
        from PIL import Image
        step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
        history_dir = os.path.join(step_config.checkpoint_dir, "pca_history")
        if not os.path.exists(history_dir):
            return None

        png_files = sorted([f for f in os.listdir(history_dir) if f.endswith(".png")])
        if len(png_files) < 2:
            return None

        images = []
        for pf in png_files:
            img_path = os.path.join(history_dir, pf)
            im = Image.open(img_path).convert("RGBA")
            images.append(im)

        # 全体で約 10 秒になるようにフレームごとの表示時間（ミリ秒）を計算
        # 最終フレームは分離した結果をじっくり見せるため長め (2500ms) に設定
        n_frames = len(images)
        final_hold_ms = 2500
        remaining_time_ms = max(1000, int(target_duration_sec * 1000) - final_hold_ms)
        step_duration_ms = max(400, remaining_time_ms // (n_frames - 1))

        durations = [step_duration_ms] * (n_frames - 1) + [final_hold_ms]

        gif_path = os.path.join(step_config.checkpoint_dir, "latent_space_evolution.gif")
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0
        )
        print(f"[★ Presentation GIF] Exported 10-second evolution animation to: {gif_path}")
        return gif_path
    except Exception as e:
        print(f"Notice: Could not generate animation GIF: {e}")
        return None

def update_pca_visualization(step: int, embeddings: np.ndarray, df_labels: pd.DataFrame, source_ids: np.ndarray, out_png: str, cycle_idx: int = 1):
    """
    アクティブラーニング各サイクル後に潜在空間の PCA プロットを自動更新保存。
    未ラベルの背景サンプルと、人間が目視判定した各クラスの分布を色分け表示。
    """
    try:
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        # 128次元のSimCLR潜在ベクトルを抽出し、超球面上の対照学習表現に基づいてL2正規化を実施
        simclr_emb = embeddings[:, :128] if embeddings.shape[1] >= 128 else embeddings
        norms = np.linalg.norm(simclr_emb, axis=1, keepdims=True)
        normed_emb = simclr_emb / np.maximum(norms, 1e-9)
        
        N = len(normed_emb)
        sample_size = min(2500, N)
        rng = np.random.RandomState(42)
        sample_indices = rng.choice(N, size=sample_size, replace=False)
        bg_emb = normed_emb[sample_indices]
        
        pca = PCA(n_components=2, random_state=42)
        pca.fit(bg_emb)
        bg_proj = pca.transform(bg_emb)
        
        fig, ax = plt.subplots(figsize=(8.5, 6.5))
        # 未ラベル背景データ
        ax.scatter(bg_proj[:, 0], bg_proj[:, 1], c="#d0d7de", alpha=0.35, s=12, label="Unlabeled Background")
        
        # 目視ラベル付きデータ
        if not df_labels.empty:
            id_to_idx = {sid: i for i, sid in enumerate(source_ids)}
            valid_pairs = [(id_to_idx[int(row["diaSourceId"])], str(row["class_name"])) 
                           for _, row in df_labels.iterrows() if int(row["diaSourceId"]) in id_to_idx]
            
            if valid_pairs:
                valid_indices, class_names = zip(*valid_pairs)
                labeled_emb = normed_emb[list(valid_indices)]
                labeled_proj = pca.transform(labeled_emb)
                
                palette = {
                    "positive_real": "#1f77b4", # 青 (train_simclrと統一)
                    "negative_real": "#d62728", # 赤 (train_simclrと統一)
                    "bogus": "#ff7f0e",         # オレンジ
                    "star": "#2ca02c",
                    "galaxy": "#9467bd",
                    "no_host": "#8c564b"
                }
                
                for cname in sorted(list(set(class_names))):
                    mask = [cn == cname for cn in class_names]
                    c_proj = labeled_proj[mask]
                    color = palette.get(cname, "black")
                    ax.scatter(c_proj[:, 0], c_proj[:, 1], c=color, s=55, edgecolors="black", linewidth=0.6, 
                               label=f"Labeled: {cname} (N={sum(mask)})", zorder=5)
                    
        var_ratio = pca.explained_variance_ratio_.sum() * 100
        ax.set_title(f"Step {step} Latent Space PCA (Cycle {cycle_idx}: {len(df_labels)} Labels)\nExplained Variance: {var_ratio:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper right", framealpha=0.9)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=140)

        # 履歴ディレクトリ（プレゼン・アニメーション用）にもナンバリングして保存
        history_dir = os.path.join(os.path.dirname(out_png), "pca_history")
        os.makedirs(history_dir, exist_ok=True)
        history_frame = os.path.join(history_dir, f"cycle_{cycle_idx:03d}_N{len(df_labels):04d}.png")
        plt.savefig(history_frame, dpi=140)
        plt.close(fig)
        print(f"[+] Updated PCA latent space plot with human labels: {out_png}")
        print(f"[+] Saved historical snapshot: {history_frame}")

        # 10秒アニメーションGIFを自動更新
        generate_latent_space_animation(step, target_duration_sec=10.0)
    except Exception as e:
        print(f"Notice: Failed to update PCA plot: {e}")

def labeled_xy(embeddings: np.ndarray, source_ids: np.ndarray, df_labels: pd.DataFrame):
    """ラベル表と埋め込みを diaSourceId で対応付け、(X, y, 対応したラベル行) を返す"""
    if df_labels is None or df_labels.empty:
        return np.empty((0, embeddings.shape[1])), np.empty(0, dtype=int), df_labels
    id_to_idx = {int(sid): i for i, sid in enumerate(source_ids)}
    rows = df_labels[df_labels["diaSourceId"].astype(np.int64).isin(id_to_idx.keys())]
    idx = [id_to_idx[int(s)] for s in rows["diaSourceId"]]
    return embeddings[idx], rows["label"].astype(int).values, rows

def evaluate_on_eval_set(rf: StampRandomForest, embeddings: np.ndarray, source_ids: np.ndarray,
                         df_eval: pd.DataFrame, classes: List[str], verbose: bool = True,
                         strata: Optional[Dict] = None) -> Dict:
    """
    固定評価用セットで RF を評価 (学習には一切使っていないラベル)。
    strata が与えられた場合は層の重み付きで母集団に対する精度を推定する (eval_acc / eval_bacc は重み付き値)。
    """
    X_ev, y_ev, rows = labeled_xy(embeddings, source_ids, df_eval)
    if len(y_ev) == 0:
        return {}
    preds, _, _, _ = rf.predict_with_uncertainty(X_ev)
    pred_labels = np.asarray(rf.clf.classes_)[preds].astype(int)
    metrics = {
        "n_eval": int(len(y_ev)),
        "eval_acc_unweighted": float(accuracy_score(y_ev, pred_labels)),
        "eval_bacc_unweighted": float(balanced_accuracy_score(y_ev, pred_labels)),
    }
    C_raw = confusion_matrix(y_ev, pred_labels, labels=list(range(len(classes))))

    if strata is not None:
        h = stratum_of(strata, rows["diaSourceId"].values)
        wm = weighted_metrics(y_ev, pred_labels, h, strata["sizes"], len(classes))
        p, se = wm["point"], wm["se"]
        metrics.update({
            "eval_weighted": True,
            "eval_acc": float(p["acc"]), "eval_acc_se": float(se["acc"]),
            "eval_bacc": float(p["bacc"]), "eval_bacc_se": float(se["bacc"]),
        })
        for k, c in enumerate(classes):
            metrics[f"eval_recall_{c}"] = float(p["recall"][k])
            metrics[f"eval_precision_{c}"] = float(p["precision"][k])
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            recall = np.diag(C_raw) / C_raw.sum(axis=1)
            precision = np.diag(C_raw) / C_raw.sum(axis=0)
        metrics.update({
            "eval_weighted": False,
            "eval_acc": metrics["eval_acc_unweighted"],
            "eval_bacc": metrics["eval_bacc_unweighted"],
        })
        for k, c in enumerate(classes):
            metrics[f"eval_recall_{c}"] = float(recall[k])
            metrics[f"eval_precision_{c}"] = float(precision[k])

    if verbose:
        print(f"\n[Eval set] N={metrics['n_eval']}")
        if strata is not None:
            print_strata_summary(strata, rows["diaSourceId"].values)
            if wm["empty_strata"]:
                print(f"  [!] Strata without eval labels are excluded: {[classes[k] for k in wm['empty_strata']]} "
                      f"(estimates cover {wm['coverage'] * 100:.1f}% of the population)")
            print(f"Weighted (population) Accuracy = {p['acc']:.4f} ± {se['acc']:.4f}, "
                  f"Balanced Accuracy = {p['bacc']:.4f} ± {se['bacc']:.4f}")
            print(pd.DataFrame({
                "est_prevalence": p["prevalence"],
                "recall": p["recall"], "recall_se": se["recall"],
                "precision": p["precision"], "precision_se": se["precision"],
            }, index=classes).round(3))
            print("Estimated population confusion matrix (counts):")
            print(pd.DataFrame(np.rint(wm["confusion_weighted"]).astype(np.int64),
                               index=[f"true:{c}" for c in classes], columns=[f"pred:{c}" for c in classes]))
            print(f"(Unweighted: Accuracy = {metrics['eval_acc_unweighted']:.4f}, Balanced Accuracy = {metrics['eval_bacc_unweighted']:.4f})")
        else:
            print(f"Accuracy = {metrics['eval_acc']:.4f}, Balanced Accuracy = {metrics['eval_bacc']:.4f} (unweighted; no strata file)")
        print("Raw confusion matrix of evaluation labels:")
        print(pd.DataFrame(C_raw, index=[f"true:{c}" for c in classes], columns=[f"pred:{c}" for c in classes]))
    return metrics

class ActiveLearningOrchestrator:
    def __init__(self, step: int = 1, config: Optional[ActiveLearningConfig] = None):
        self.step = step
        self.step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
        self.config = config or ActiveLearningConfig()
        self.viewer = LabelingViewer(step=step)
        self.rf = StampRandomForest(step=step)
        self.history_file = os.path.join(DATA_DIR, f"active_learning_history_step{step}.csv")
        self.cycle_count = 0
        if os.path.exists(self.history_file):
            try:
                self.cycle_count = len(pd.read_csv(self.history_file))
            except Exception:
                pass

    def _load_eval_labels(self) -> pd.DataFrame:
        if not self.step_config.eval_labels_file:
            return pd.DataFrame(columns=["diaSourceId", "diaObjectId", "label", "class_name", "timestamp"])
        return load_human_labels(self.step_config.eval_labels_file)

    def _full_probs(self, probs: np.ndarray) -> np.ndarray:
        """RF の predict_proba (学習時に存在したクラスのみの列) を全クラス列に並べ直す"""
        full = np.zeros((len(probs), len(self.step_config.classes)))
        for j, c in enumerate(self.rf.clf.classes_):
            full[:, int(c)] = probs[:, j]
        return full

    def _append_history(self, entry: Dict):
        """履歴 CSV に追記 (列が増えても既存行と整合するよう全体を書き直す)"""
        df_new = pd.DataFrame([entry])
        if os.path.exists(self.history_file):
            try:
                df_new = pd.concat([pd.read_csv(self.history_file), df_new], ignore_index=True)
            except Exception:
                pass
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        df_new.to_csv(self.history_file, index=False)

    def _strata_file(self) -> Optional[str]:
        return strata_file_path(self.step_config.eval_labels_file) if self.step_config.eval_labels_file else None

    def _load_strata(self) -> Optional[Dict]:
        path = self._strata_file()
        return load_strata(path) if path else None

    def _ensure_strata(self, embeddings: np.ndarray, source_ids: np.ndarray, obj_ids: np.ndarray) -> Dict:
        """層ファイルが無ければ、現在の RF の予測クラスで評価対象の母集団を層に分けて固定保存"""
        strata = self._load_strata()
        if strata is not None:
            return strata
        print("[*] No evaluation strata found. Creating strata from the current Random Forest predictions...")
        df_train = load_human_labels(self.step_config.labels_file)
        df_eval = self._load_eval_labels()
        # 評価対象の母集団: 有効な埋め込みのうち、学習用ラベルのソース・天体を除いたもの (build_eval_set と同じ条件)
        eligible = valid_embedding_mask(embeddings)
        if not df_train.empty:
            eligible &= ~np.isin(source_ids, df_train["diaSourceId"].astype(np.int64).values)
            eligible &= ~np.isin(obj_ids, df_train["diaObjectId"].astype(np.int64).values)
        # 既存の評価用ラベルは必ず母集団に含める
        if not df_eval.empty:
            eligible |= np.isin(source_ids, df_eval["diaSourceId"].astype(np.int64).values)
        _, probs, _, _ = self.rf.predict_with_uncertainty(embeddings)
        pred = np.argmax(self._full_probs(probs), axis=1)
        return create_strata(self._strata_file(), source_ids, pred, eligible, self.step_config.classes, self.rf.model_path)

    def build_eval_set(self, n_target: int = None, interactive: bool = True, seed: int = 20261018,
                       stratum: Optional[str] = None) -> pd.DataFrame:
        """
        能動学習とは独立な固定評価用セットを目視判定する (ブラインド: 予測は表示しない)。
        - stratum=None: 母集団全体から一様ランダムに選び、評価用セット全体が n_target 件になるまで判定
        - stratum='negative_real' など: 予測クラスで固定した層の中から一様ランダムに選び、
          その層の評価ラベルが n_target 件になるまで判定 (層ファイルが無ければ現在の RF で作成)
        固定シードの順列を先頭から辿るため、中断しても同じ順序で再開できる。
        評価用ラベルは学習用ラベルとは別ファイルに保存され、学習・能動サンプリングからは除外される。
        """
        eval_file = self.step_config.eval_labels_file
        cache = load_cached_embeddings(self.step)
        embeddings = cache["embeddings"]
        source_ids = cache["diaSourceIds"]
        obj_ids = cache["diaObjectIds"]
        mjds = cache["midpointMjdTai"]
        bands = cache["bands"]
        id_to_idx = {int(s): i for i, s in enumerate(source_ids)}

        df_eval = self._load_eval_labels()
        df_train = load_human_labels(self.step_config.labels_file)
        done = set(df_eval["diaSourceId"].astype(np.int64)) if not df_eval.empty else set()
        exclude = done | (set(df_train["diaSourceId"].astype(np.int64)) if not df_train.empty else set())
        # 学習用ラベルと同一天体 (別観測) は評価から除外し、天体単位のリークを防ぐ
        train_objs = set(df_train["diaObjectId"].astype(np.int64)) if not df_train.empty else set()

        if stratum is None:
            n_target = n_target or self.config.eval_set_size
            n_done = len(done)
            selection = "eval_random"
            valid = np.where(valid_embedding_mask(embeddings))[0]
            order_sids = source_ids[np.random.default_rng(seed).permutation(valid)]
            print(f"=== Building Random Evaluation Set (Step {self.step}): {n_done} / {n_target} labeled -> {eval_file} ===")
        else:
            if stratum not in self.step_config.classes:
                raise ValueError(f"Unknown stratum '{stratum}'. Choose from {self.step_config.classes}")
            strata = self._ensure_strata(embeddings, source_ids, obj_ids)
            k = self.step_config.classes.index(stratum)
            n_target = n_target or 50
            h_done = stratum_of(strata, list(done)) if done else np.array([], dtype=int)
            n_done = int((h_done == k).sum())
            selection = f"eval_strat:{stratum}"
            members = strata["source_ids"][strata["strata"] == k]
            order_sids = np.random.default_rng(seed + 1000 * (k + 1)).permutation(members)
            print(f"=== Building Stratified Evaluation Set (Step {self.step}), stratum pred:{stratum} "
                  f"(N_h={strata['sizes'][k]}): {n_done} / {n_target} labeled -> {eval_file} ===")

        if n_done >= n_target:
            print("[+] Target already reached.")
        else:
            eval_viewer = LabelingViewer(step=self.step)
            eval_viewer.labels_file = eval_file
            for sid in order_sids:
                if n_done >= n_target:
                    break
                sid = int(sid)
                idx = id_to_idx.get(sid)
                if idx is None or sid in exclude or int(obj_ids[idx]) in train_objs:
                    continue
                rec = {"diaSourceId": sid, "diaObjectId": int(obj_ids[idx]), "midpointMjdTai": float(mjds[idx]), "band": str(bands[idx])}
                print(f"\n[Eval {n_done + 1}/{n_target}{'' if stratum is None else ' in pred:' + stratum}]")
                ans = eval_viewer.review_sample(rec, pred_class=None, confidence=None, margin=None,
                                                interactive=interactive, selection=selection)
                if ans == "QUIT":
                    print("[*] Evaluation labeling paused. Re-run to resume from the same random order.")
                    break
                elif ans == "SKIP_EMPTY" or ans is None:
                    # スキップは評価対象から外す (ランダム順は維持)
                    exclude.add(sid)
                    continue
                n_done += 1

        df_eval = self._load_eval_labels()
        print(f"\n[+] Evaluation set: {len(df_eval)} labels. Class distribution:")
        if not df_eval.empty:
            print(df_eval["class_name"].value_counts())
            strata = self._load_strata()
            if strata is not None:
                print_strata_summary(strata, df_eval["diaSourceId"].values)
        return df_eval

    def source_meta(self) -> Dict[int, Dict]:
        """diaSourceId -> 観測時刻とバンド (画像のパスを作るため)"""
        cache = load_cached_embeddings(self.step)
        return {int(s): {"midpointMjdTai": float(m), "band": str(b)}
                for s, m, b in zip(cache["diaSourceIds"], cache["midpointMjdTai"], cache["bands"])}

    def retrain_rf(self) -> Dict:
        """現在の学習用ラベルで RF を学び直し、評価用セットで評価して履歴に記録 (目視判定はしない)"""
        cache = load_cached_embeddings(self.step)
        df_labels = load_human_labels(self.step_config.labels_file)
        df_eval = self._load_eval_labels()
        if not df_eval.empty and not df_labels.empty:
            df_labels = df_labels[~df_labels["diaSourceId"].astype(np.int64).isin(df_eval["diaSourceId"].astype(np.int64))]
        X, y, _ = labeled_xy(cache["embeddings"], cache["diaSourceIds"], df_labels)
        print(f"=== Retraining Random Forest (Step {self.step}) on {len(y)} labels ===")
        metrics = self.rf.train(X, y)
        if not df_eval.empty:
            metrics.update(evaluate_on_eval_set(self.rf, cache["embeddings"], cache["diaSourceIds"], df_eval,
                                                self.step_config.classes, strata=self._load_strata()))
        entry = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "total_labeled": len(y), "newly_added": 0,
                 "selection": "retrain_rf", "feedback_mode": "retrain_rf"}
        entry.update({k: v for k, v in metrics.items() if not isinstance(v, (list, dict, np.ndarray))})
        self._append_history(entry)
        return metrics

    def evaluate(self) -> Dict:
        """現在の埋め込みと RF を固定評価用セットで評価 (層ファイルがあれば重み付き)"""
        cache = load_cached_embeddings(self.step)
        df_eval = self._load_eval_labels()
        if df_eval.empty:
            print(f"[!] No evaluation labels found at {self.step_config.eval_labels_file}. Run --mode build_eval_set first.")
            return {}
        return evaluate_on_eval_set(self.rf, cache["embeddings"], cache["diaSourceIds"], df_eval,
                                    self.step_config.classes, strata=self._load_strata())

    def _audit_report(self, paths: Dict[str, str]) -> Dict:
        df_list = pd.read_csv(paths["list"])
        df_audit = load_human_labels(paths["labels"])
        df_eval = self._load_eval_labels()
        return audit_report(df_list, df_audit, self.step_config.classes, strata=self._load_strata(),
                            eval_source_ids=df_eval["diaSourceId"].values if not df_eval.empty else None,
                            report_file=paths["report"])

    def audit_eval(self, n_correct: int = 50, interactive: bool = True, report_only: bool = False) -> Dict:
        """
        評価用ラベルのブラインド再判定による監査。
        初回は現在の RF の予測で「誤り全件 + 正解からランダム n_correct 件」の監査リストを作って固定し、
        元のラベル・モデル予測を隠して再判定する ([u] Unsure を選択可)。終了後に自己一致率などを集計。
        """
        paths = audit_paths(self.step_config.eval_labels_file)
        if not os.path.exists(paths["list"]):
            if report_only:
                print(f"[!] No audit list at {paths['list']}. Run --mode audit_eval first.")
                return {}
            df_eval = self._load_eval_labels()
            if df_eval.empty:
                print(f"[!] No evaluation labels found at {self.step_config.eval_labels_file}.")
                return {}
            cache = load_cached_embeddings(self.step)
            X_ev, _, rows = labeled_xy(cache["embeddings"], cache["diaSourceIds"], df_eval)
            preds, probs, _, _ = self.rf.predict_with_uncertainty(X_ev)
            pred_labels = np.argmax(self._full_probs(probs), axis=1)
            create_audit_list(paths["list"], rows, pred_labels, n_correct=n_correct)

        if not report_only:
            df_list = pd.read_csv(paths["list"]).sort_values("order")
            done = set(load_human_labels(paths["labels"])["diaSourceId"].astype(np.int64))
            todo = df_list[~df_list["diaSourceId"].astype(np.int64).isin(done)]
            print(f"=== Evaluation Label Audit (Step {self.step}): {len(df_list) - len(todo)} / {len(df_list)} re-reviewed -> {paths['labels']} ===")
            if len(todo) > 0:
                cache = load_cached_embeddings(self.step)
                id_to_idx = {int(s): i for i, s in enumerate(cache["diaSourceIds"])}
                viewer = LabelingViewer(step=self.step)
                viewer.labels_file = paths["labels"]
                n_done = len(df_list) - len(todo)
                for r in todo.itertuples():
                    i = id_to_idx.get(int(r.diaSourceId))
                    if i is None:
                        continue
                    rec = {"diaSourceId": int(r.diaSourceId), "diaObjectId": int(r.diaObjectId),
                           "midpointMjdTai": float(cache["midpointMjdTai"][i]), "band": str(cache["bands"][i])}
                    print(f"\n[Audit {n_done + 1}/{len(df_list)}]")
                    ans = viewer.review_sample(rec, pred_class=None, confidence=None, margin=None,
                                               interactive=interactive, selection="audit", allow_unsure=True)
                    if ans == "QUIT":
                        print("[*] Audit paused. Re-run to resume.")
                        break
                    elif ans is None or ans == "SKIP_EMPTY":
                        # スキップは次回また提示される
                        continue
                    n_done += 1
        return self._audit_report(paths)

    def adjudicate_eval(self, interactive: bool = True) -> pd.DataFrame:
        """
        監査で元のラベルと再判定が一致しなかった評価サンプルを、両方の判定を並べて表示して最終ラベルを決め、
        評価用ラベルに反映する (中断・再開可, 反映は冪等)。
        """
        eval_file = self.step_config.eval_labels_file
        apaths = audit_paths(eval_file)
        jpaths = adjudication_paths(eval_file)
        if not (os.path.exists(apaths["list"]) and os.path.exists(apaths["labels"])):
            print("[!] No audit results found. Run --mode audit_eval first.")
            return pd.DataFrame()
        dis = disagreements(pd.read_csv(apaths["list"]), load_human_labels(apaths["labels"]))
        done = set(load_human_labels(jpaths["adjudicated"])["diaSourceId"].astype(np.int64))
        todo = dis[~dis["diaSourceId"].astype(np.int64).isin(done)]
        print(f"=== Adjudication of Evaluation Labels (Step {self.step}): {len(dis) - len(todo)} / {len(dis)} disagreements decided ===")

        if len(todo) > 0:
            cache = load_cached_embeddings(self.step)
            id_to_idx = {int(s): i for i, s in enumerate(cache["diaSourceIds"])}
            viewer = LabelingViewer(step=self.step)
            viewer.labels_file = jpaths["adjudicated"]
            n_done = len(dis) - len(todo)
            for r in todo.itertuples():
                i = id_to_idx.get(int(r.diaSourceId))
                if i is None:
                    continue
                rec = {"diaSourceId": int(r.diaSourceId), "diaObjectId": int(r.diaObjectId),
                       "midpointMjdTai": float(cache["midpointMjdTai"][i]), "band": str(cache["bands"][i])}
                note = f"Adjudication  |  1st label: [{r.orig_class}]  /  re-review: [{r.audit_class}]\nChoose the final label"
                print(f"\n[Adjudicate {n_done + 1}/{len(dis)}] 1st: {r.orig_class} / re-review: {r.audit_class}")
                ans = viewer.review_sample(rec, pred_class=None, confidence=None, margin=None,
                                           interactive=interactive, selection="adjudication", title_note=note)
                if ans == "QUIT":
                    print("[*] Adjudication paused. Re-run to resume.")
                    break
                elif ans is None or ans == "SKIP_EMPTY":
                    continue
                n_done += 1

        return apply_adjudication(eval_file, jpaths["adjudicated"], jpaths["backup"])

    def threshold_analysis(self, out_csv: Optional[str] = None, out_png: Optional[str] = None) -> pd.DataFrame:
        """
        現在の RF のスコア閾値ごとに、母集団に対する純度・完全性・選択数の推定値を評価用セット (層の重み付き) で求める。
        """
        df_eval = self._load_eval_labels()
        strata = self._load_strata()
        if df_eval.empty or strata is None:
            print("[!] Need evaluation labels and strata (run build_eval_set with --eval_stratum first).")
            return pd.DataFrame()
        cache = load_cached_embeddings(self.step)
        X_ev, y_ev, rows = labeled_xy(cache["embeddings"], cache["diaSourceIds"], df_eval)
        _, probs, _, _ = self.rf.predict_with_uncertainty(X_ev)
        probs = self._full_probs(probs)
        h = stratum_of(strata, rows["diaSourceId"].values)

        targets = list(targets_for(self.step_config.classes))
        df = pd.concat([threshold_curves(probs, y_ev, h, strata["sizes"], self.step_config.classes, target=t)
                        for t in targets], ignore_index=True)

        out_csv = out_csv or os.path.join(DATA_DIR, f"threshold_curves_step{self.step}.csv")
        out_png = out_png or os.path.join(self.step_config.checkpoint_dir, "threshold_curves.png")
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        df.to_csv(out_csv, index=False)
        plot_threshold_curves(df, out_png, title=f"Step {self.step} RF on evaluation set (N={len(y_ev)}, strata-weighted)")

        print(f"\n=== Threshold Analysis (Step {self.step}, evaluation set N={len(y_ev)}, strata-weighted) ===")
        for t in targets:
            d = df[df["target"] == t]
            print(f"\n[target: {t}]  (purity / completeness with 95% CI, estimated N selected in population)")
            for r in d[d["threshold"].isin([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])].itertuples():
                print(f"  thr {r.threshold:.2f}: purity {r.purity:.3f} [{r.purity_lo:.3f}, {r.purity_hi:.3f}]  "
                      f"completeness {r.completeness:.3f} [{r.completeness_lo:.3f}, {r.completeness_hi:.3f}]  "
                      f"N_sel ~{r.n_selected_est:,.0f}  (eval n={r.n_eval_selected})")
            for goal in [0.90, 0.95]:
                ok = d[d["purity"] >= goal]
                if len(ok):
                    r = ok.iloc[0]
                    print(f"  -> lowest threshold with purity >= {goal:.2f}: {r.threshold:.2f} "
                          f"(completeness {r.completeness:.3f}, N_sel ~{r.n_selected_est:,.0f})")
                else:
                    print(f"  -> no threshold reaches purity >= {goal:.2f}")
        print(f"\n[+] Saved curves -> {out_csv}, plot -> {out_png}")
        print("    Note: purity/completeness are measured against single human labels (self-agreement ~0.94).")
        return df

    def freeze_model(self, version: str, thresholds: Dict[str, float]) -> Dict:
        """
        現在の RF を閾値と一緒にバージョン名付きで固定保存し、評価用セットでの判定性能 (層の重み付き) を記録する。
        """
        classes = self.step_config.classes
        extra = {}
        df_labels = load_human_labels(self.step_config.labels_file)
        extra["n_training_labels"] = int(len(df_labels))
        df_eval = self._load_eval_labels()
        strata = self._load_strata()
        if not df_eval.empty and strata is not None:
            cache = load_cached_embeddings(self.step)
            X_ev, y_ev, rows = labeled_xy(cache["embeddings"], cache["diaSourceIds"], df_eval)
            _, probs, _, _ = self.rf.predict_with_uncertainty(X_ev)
            probs = self._full_probs(probs)
            h = stratum_of(strata, rows["diaSourceId"].values)
            keep = h >= 0
            n_h = np.bincount(h[keep], minlength=len(strata["sizes"])).astype(float)
            w = np.where(keep, (strata["sizes"] / np.maximum(n_h, 1))[np.maximum(h, 0)], 0.0)
            dec = decide(probs, classes, thresholds)
            true_names = np.array(classes, dtype=object)[y_ev]
            perf = {}
            for c in classes + ["uncertain"]:
                sel = dec == c
                entry = {"n_eval_assigned": int(sel.sum()), "est_fraction_of_population": float(w[sel].sum() / w.sum())}
                if c in classes:
                    tp = w[sel & (true_names == c)].sum()
                    entry["purity"] = float(tp / w[sel].sum()) if w[sel].sum() > 0 else None
                    entry["completeness"] = float(tp / w[true_names == c].sum()) if w[true_names == c].sum() > 0 else None
                else:
                    # uncertain に入ったソースの真のクラス構成
                    entry["true_class_mix"] = {k: float(w[sel & (true_names == k)].sum() / w[sel].sum()) if w[sel].sum() > 0 else None
                                               for k in classes}
                perf[c] = entry
            extra["eval_set"] = {"n_eval": int(len(y_ev)), "strata_created": strata["created"],
                                 "decision_performance_strata_weighted": perf,
                                 "note": "Measured on the same evaluation set used to choose the thresholds (optimistic); "
                                         "labels are single human judgments (self-agreement ~0.94)."}

        ckpt = os.path.join(self.step_config.checkpoint_dir, "best_simclr_encoder.pt")
        paths = save_frozen_model(self.step_config.checkpoint_dir, version, self.rf.model_path, ckpt,
                                  classes, thresholds, extra)
        print(f"[+] Frozen model '{version}':\n    RF   -> {paths['rf']}\n    meta -> {paths['meta']}")
        print(f"    thresholds: {thresholds}")
        if "eval_set" in extra:
            print("    Decision performance on evaluation set (strata-weighted):")
            for c, e in extra["eval_set"]["decision_performance_strata_weighted"].items():
                if c in classes:
                    pur = f"{e['purity']:.3f}" if e["purity"] is not None else "n/a"
                    com = f"{e['completeness']:.3f}" if e["completeness"] is not None else "n/a"
                    print(f"      {c:14s} purity {pur}  completeness {com}  share {e['est_fraction_of_population']:.3f}")
                else:
                    mix = ", ".join(f"{k}={v:.2f}" for k, v in e["true_class_mix"].items() if v is not None)
                    print(f"      {c:14s} share {e['est_fraction_of_population']:.3f}  (true mix: {mix})")
        return paths

    def run_cycle(self, n_samples: int = None, interactive: bool = True) -> Dict:
        """
        能動・強化学習の 1 サイクルを実行:
        1. 潜在埋め込みのロード
        2. 既存目視ラベルの確認 (未訓練なら初期学習)
        3. 複数枠 (不確実性・real 確率・異常度・ランダム・多様性) によるサンプリング
        4. ユーザーによる目視判定
        5. モデル更新 (SupCon微調整 & RF再学習)
        6. 固定評価用セットでの評価
        """
        n_samples = n_samples or self.config.batch_size_per_iter
        print(f"\n=======================================================")
        print(f"  Starting Active Learning Cycle (Step {self.step}: {self.step_config.name})")
        print(f"=======================================================")

        # 1. 潜在ベクトルのロード
        cache = load_cached_embeddings(self.step)
        embeddings = cache["embeddings"]
        source_ids = cache["diaSourceIds"]
        obj_ids = cache["diaObjectIds"]
        mjds = cache["midpointMjdTai"]
        bands = cache["bands"]

        # 2. 既存ラベルのロード (評価用セットは学習に使わず、サンプリング対象からも除外)
        df_labels = load_human_labels(self.step_config.labels_file)
        df_eval = self._load_eval_labels()
        eval_ids = set(df_eval["diaSourceId"].astype(np.int64)) if not df_eval.empty else set()
        if eval_ids and not df_labels.empty:
            df_labels = df_labels[~df_labels["diaSourceId"].astype(np.int64).isin(eval_ids)]
        labeled_map = dict(zip(df_labels["diaSourceId"], df_labels["label"])) if not df_labels.empty else {}
        print(f"Current Labeled Stamps: {len(labeled_map)} (+ {len(eval_ids)} held-out evaluation labels)")

        # インデックスと diaSourceId の相互マップ
        id_to_idx = {sid: idx for idx, sid in enumerate(source_ids)}
        labeled_indices = {id_to_idx[sid] for sid in labeled_map if sid in id_to_idx}
        excluded_indices = labeled_indices | {id_to_idx[sid] for sid in eval_ids if sid in id_to_idx}
        # 評価用セットと同一天体の別観測もサンプリング対象から除外 (天体単位のリーク防止)
        if eval_ids:
            eval_objs = np.array(sorted(set(df_eval["diaObjectId"].astype(np.int64))))
            excluded_indices |= set(np.where(np.isin(obj_ids, eval_objs))[0].tolist())
        # 指定した天体 (例: 分光確認済みの SN リスト) は候補から除き、評価用にとっておく
        if getattr(self.config, "exclude_objects_file", ""):
            ex = pd.read_csv(self.config.exclude_objects_file)
            col = next((c for c in ["diaObjectId", "objId", "obj_id"] if c in ex.columns), None)
            if col is None:
                raise ValueError(f"{self.config.exclude_objects_file} needs a diaObjectId / objId / obj_id column")
            ex_ids = ex[col].astype(np.int64).values
            ex_idx = set(np.where(np.isin(obj_ids, ex_ids))[0].tolist())
            excluded_indices |= ex_idx
            print(f"[*] Excluding {len(ex_ids)} objects ({len(ex_idx)} sources) listed in {self.config.exclude_objects_file}")

        # 3. Random Forest の学習または推論
        if len(labeled_indices) >= 6 and len(set(labeled_map.values())) >= 2:
            # 既にラベルがある場合は学習
            X_train, y_train, _ = labeled_xy(embeddings, source_ids, df_labels)
            rf_metrics = self.rf.train(X_train, y_train)
            preds, probs, margins, entropies = self.rf.predict_with_uncertainty(embeddings)
            probs = self._full_probs(probs)
            preds = np.argmax(probs, axis=1)
            if eval_ids:
                rf_metrics.update(evaluate_on_eval_set(self.rf, embeddings, source_ids, df_eval, self.step_config.classes,
                                                     strata=self._load_strata()))
        else:
            # 初期状態 (異常度/ランダム/クラスタ代表サンプリングでシードラベルを取得)
            print("[!] Initial stage: No/insufficient labels. Selecting diverse seed cutouts.")
            preds = np.zeros(len(embeddings), dtype=int)
            probs = None
            margins = None
            entropies = None
            rf_metrics = {"oob_accuracy": 0.0, "cv_accuracy": None}

        # 4. 能動的サンプリング (不確実性 + real 確率 + 異常度 + ランダム + 多様性)
        candidates = sample_candidates_for_labeling(
            embeddings=embeddings,
            margins=margins,
            entropies=entropies,
            already_labeled_indices=excluded_indices,
            n_samples=n_samples,
            uncertainty_method=self.config.uncertainty_method,
            probs=probs,
            classes=self.step_config.classes,
            uncertainty_ratio=self.config.uncertainty_ratio,
            exploit_ratio=self.config.exploit_ratio,
            anomaly_ratio=self.config.anomaly_ratio,
            random_ratio=self.config.random_ratio,
            diversity_ratio=self.config.diversity_ratio,
            focus_classes=getattr(self.config, "focus_classes", None)
        )
        reasons = pd.Series([r for _, r in candidates]).value_counts().to_dict()
        print(f"Selected {len(candidates)} candidates for human review: {reasons}")

        # 5. 目視判定の実施
        newly_labeled_count = 0
        for idx, reason in candidates:
            sid = int(source_ids[idx])
            oid = int(obj_ids[idx])
            mjd = float(mjds[idx])
            band = str(bands[idx])

            rec = {
                "diaSourceId": sid,
                "diaObjectId": oid,
                "midpointMjdTai": mjd,
                "band": band
            }
            if probs is not None:
                pred_cname = self.step_config.classes[preds[idx]]
                conf = float(np.max(probs[idx]))
                margin = float(margins[idx])
            else:
                pred_cname, conf, margin = "N/A", 0.0, 0.0

            ans = self.viewer.review_sample(rec, pred_class=pred_cname, confidence=conf, margin=margin,
                                            interactive=interactive, selection=reason)
            if ans == "QUIT":
                print("[*] Human labeling paused by user.")
                break
            elif ans == "SKIP_EMPTY":
                continue
            elif ans is not None:
                newly_labeled_count += 1

        from .labeling_viewer import cleanup_gui
        cleanup_gui()
        print(f"\n[+] Added {newly_labeled_count} newly verified labels.")

        # 6. フィードバック学習 (微調整 または フル学習 & RF 再学習)
        df_labels_updated = load_human_labels(self.step_config.labels_file)
        if eval_ids and not df_labels_updated.empty:
            df_labels_updated = df_labels_updated[~df_labels_updated["diaSourceId"].astype(np.int64).isin(eval_ids)]
        feedback_mode = getattr(self.config, "feedback_mode", "fine_tune")
        if len(df_labels_updated) > len(df_labels):
            if feedback_mode == "rf_only":
                # エンコーダ・埋め込みは固定したまま、RF の再学習のみ
                print(f"\n[*] Feedback mode 'rf_only': encoder and embeddings unchanged, retraining Random Forest only.")
            elif feedback_mode == "full_train":
                print(f"\n[*] Executing Supervised Full Training on {len(df_labels_updated)} human labels...")
                train_supervised(
                    step=self.step,
                    epochs=getattr(self.config, "full_train_epochs", 30),
                    loss_type="joint"
                )
            else:
                db_cfg = DBConfig()
                df_cat = load_catalog_from_db(db_cfg)
                fine_tune_encoder_with_supcon(
                    step=self.step,
                    df_cat=df_cat,
                    df_labels=df_labels_updated,
                    epochs=self.config.supcon_epochs,
                    lr=self.config.supcon_lr
                )
            if feedback_mode == "rf_only":
                new_emb = embeddings
                new_sids = source_ids
            else:
                # 再度埋め込みをキャッシュ更新
                extract_embeddings(step=self.step)
                # 新しい埋め込みで RF を再学習 (diaSourceId で X と y を対応付け)
                new_cache = load_cached_embeddings(self.step)
                new_emb = new_cache["embeddings"]
                new_sids = new_cache["diaSourceIds"]
            X_new, y_new, _ = labeled_xy(new_emb, new_sids, df_labels_updated)
            rf_metrics = self.rf.train(X_new, y_new)
            if eval_ids:
                rf_metrics.update(evaluate_on_eval_set(self.rf, new_emb, new_sids, df_eval, self.step_config.classes,
                                                     strata=self._load_strata()))

            self.cycle_count += 1
            # PCA 潜在空間プロットを更新保存 (目視ラベル付きサンプルを色分け表示)
            pca_png = os.path.join(self.step_config.checkpoint_dir, "latent_space_pca.png")
            update_pca_visualization(self.step, new_emb, df_labels_updated, new_sids, pca_png, cycle_idx=self.cycle_count)

        # 履歴記録
        history_entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_labeled": len(df_labels_updated),
            "newly_added": newly_labeled_count,
            "oob_accuracy": rf_metrics.get("oob_accuracy", 0.0),
            "cv_accuracy": rf_metrics.get("cv_accuracy"),
            "n_eval": rf_metrics.get("n_eval"),
            "eval_acc": rf_metrics.get("eval_acc"),
            "eval_bacc": rf_metrics.get("eval_bacc"),
            "selection": ";".join(f"{k}={v}" for k, v in reasons.items()),
            "feedback_mode": feedback_mode
        }
        # 重み付き評価の標準誤差・クラス別 recall / precision も記録
        history_entry.update({k: v for k, v in rf_metrics.items() if k.startswith("eval_") and k not in history_entry})
        self._append_history(history_entry)

        eval_str = f", Eval Acc: {history_entry['eval_acc']:.4f} (N={history_entry['n_eval']})" if history_entry["eval_acc"] is not None else ""
        print(f"Cycle Complete. Current Accuracy (OOB): {history_entry['oob_accuracy']:.4f}{eval_str}, Total Labeled: {history_entry['total_labeled']}")
        return history_entry
