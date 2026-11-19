"""
Unified CLI Pipeline Runner for Postage Stamp Classifier
Supports Step 1 (Difference Real/Bogus) and Step 2 (Host Star/Galaxy/No Host/Bogus).
"""

import os
import sys

# プロジェクトルート (/mnt/ugnas/work/LSST/fink2) を sys.path に確実に追加
proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

import argparse
import time
import numpy as np
import pandas as pd
import matplotlib

# デスクトップ環境 (DISPLAY が設定されている) の場合は GUI バックエンド (TkAgg) を優先
if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
    try:
        matplotlib.use("TkAgg")
    except Exception:
        pass

if __package__ is None or __package__ == "":
    from stamp_classifier.config import STEP1_CONFIG, STEP2_CONFIG, DATA_DIR, ActiveLearningConfig
    from stamp_classifier.train_simclr import train_simclr
    from stamp_classifier.train_supervised import train_supervised
    from stamp_classifier.feature_extractor import extract_embeddings, load_cached_embeddings, append_morphology_to_cache
    from stamp_classifier.rf_classifier import StampRandomForest
    from stamp_classifier.loop_orchestrator import ActiveLearningOrchestrator
    from stamp_classifier.labeling_viewer import load_human_labels
else:
    from .config import STEP1_CONFIG, STEP2_CONFIG, DATA_DIR, ActiveLearningConfig
    from .train_simclr import train_simclr
    from .train_supervised import train_supervised
    from .feature_extractor import extract_embeddings, load_cached_embeddings, append_morphology_to_cache
    from .rf_classifier import StampRandomForest
    from .loop_orchestrator import ActiveLearningOrchestrator
    from .labeling_viewer import load_human_labels

def classify_all(step: int = 1, out_csv: str = None, model_version: str = None):
    """
    全スタンプを分類。model_version を指定すると、freeze_model で固定した RF と閾値を使い、
    判定列 (decision: クラス名 または 'uncertain') を追加する。
    """
    from stamp_classifier.model_registry import load_frozen_model_meta, decide
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    classes = step_config.classes
    print(f"=== Classifying All Stamps for Step {step}: {step_config.name} ===")

    meta = None
    rf = StampRandomForest(step=step)
    if model_version:
        meta = load_frozen_model_meta(step_config.checkpoint_dir, model_version)
        rf.model_path = meta["rf_path"]
        print(f"Using frozen model '{model_version}' ({meta['rf_path']}), thresholds: {meta['thresholds']}")

    cache = load_cached_embeddings(step)
    embeddings = cache["embeddings"]
    source_ids = cache["diaSourceIds"]
    obj_ids = cache["diaObjectIds"]
    mjds = cache["midpointMjdTai"]
    bands = cache["bands"]

    _, raw_probs, margins, entropies = rf.predict_with_uncertainty(embeddings)
    # RF のクラス列 (clf.classes_) を設定のクラス順に並べ直す
    probs = np.zeros((len(raw_probs), len(classes)))
    for j, c in enumerate(rf.clf.classes_):
        probs[:, int(c)] = raw_probs[:, j]
    preds = probs.argmax(axis=1)

    df_out = pd.DataFrame({
        "diaSourceId": source_ids,
        "diaObjectId": obj_ids,
        "midpointMjdTai": mjds,
        "band": bands,
        "predicted_class": [classes[p] for p in preds],
        "confidence": probs.max(axis=1),
        "margin": margins,
        "entropy": entropies
    })

    # 各クラスの予測確率列を追加
    for c_idx, c_name in enumerate(classes):
        df_out[f"prob_{c_name}"] = probs[:, c_idx]

    if meta is not None:
        df_out["decision"] = decide(probs, classes, meta["thresholds"])
        df_out["model_version"] = model_version

    if not out_csv:
        suffix = f"_{model_version}" if model_version else ""
        out_csv = os.path.join(DATA_DIR, f"predictions_step{step}{suffix}.csv")
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    print(f"[+] Classification complete for {len(df_out)} stamps -> Saved to {out_csv}")
    print("\nClass Distribution Summary (argmax):")
    print(df_out["predicted_class"].value_counts())
    if meta is not None:
        print("\nDecision Summary (thresholded):")
        print(df_out["decision"].value_counts())
    return df_out

def aggregate_objects_cli(model_version: str, min_real_sources: int = 1, step2_sources: str = "real_uncertain",
                          object_classes: str = "real,uncertain", with_history: bool = True, refresh_history: bool = False,
                          moving_min_rate: float = 0.2, moving_min_sep: float = 0.7, include_moving: bool = False):
    """
    step1 の固定モデルによる全件判定 (predictions_step1_<version>.csv) を天体単位にまとめ、
    step2 (母天体分類) の入力ソース一覧を作成する。
    """
    from stamp_classifier.object_aggregation import aggregate_objects, select_step2_sources, write_step2_source_list
    pred_csv = os.path.join(DATA_DIR, f"predictions_step1_{model_version}.csv")
    if not os.path.exists(pred_csv):
        raise FileNotFoundError(f"{pred_csv} not found. Run --mode classify_all --model_version {model_version} first.")
    print(f"=== Aggregating step 1 decisions by object: {pred_csv} ===")
    df_src = pd.read_csv(pred_csv)
    # 除外する visit (data/excluded_visits.csv) のソースは bogus 扱いにする (DB と step1 の出力は変更しない)
    from stamp_classifier.visit_quality import excluded_source_ids, load_excluded_visits
    ex_ids = excluded_source_ids()
    df_src["excluded_visit"] = df_src["diaSourceId"].isin(ex_ids)
    if df_src["excluded_visit"].any():
        print(f"[*] Excluded visits {load_excluded_visits()['visit'].tolist()}: {int(df_src['excluded_visit'].sum())} sources "
              f"treated as bogus (step1 decisions there: {df_src.loc[df_src['excluded_visit'], 'decision'].value_counts().to_dict()})")
        df_src.loc[df_src["excluded_visit"], "decision"] = "bogus"
    df_obj = aggregate_objects(df_src, min_real_sources=min_real_sources)
    if with_history:
        from stamp_classifier.object_aggregation import load_detection_history, add_detection_history
        from stamp_classifier.dataset import DBConfig
        hist = load_detection_history(DBConfig(), os.path.join(DATA_DIR, "cache", "detection_history.parquet"),
                                      refresh=refresh_history)
        df_obj = add_detection_history(df_obj, hist)
        from stamp_classifier.object_aggregation import flag_moving
        df_obj = flag_moving(df_obj, min_rate=moving_min_rate, min_sep=moving_min_sep)
    obj_csv = os.path.join(DATA_DIR, f"objects_step1_{model_version}.csv")
    df_obj.to_csv(obj_csv, index=False)
    print(f"[+] {len(df_obj)} objects -> {obj_csv}  (object 'real' requires >= {min_real_sources} real source(s))")
    print("\nObject class summary:")
    print(df_obj["object_class"].value_counts().to_string())
    print("\nReal objects by type:")
    print(df_obj.loc[df_obj["object_class"] == "real", "real_type"].value_counts().to_string())
    if "detection_pattern" in df_obj.columns:
        print("\nDetection pattern (diaSource + prvDiaSources) by object class:")
        print(pd.crosstab(df_obj["object_class"], df_obj["detection_pattern"]).to_string())
        print(f"\nMoving-object candidates (single night, sep >= {moving_min_sep}\" and rate >= {moving_min_rate}\"/min):")
        print(pd.crosstab(df_obj["object_class"], df_obj["moving_candidate"]).to_string())

    classes = [c.strip() for c in object_classes.split(",") if c.strip()]
    df_sel = select_step2_sources(df_src, df_obj, classes, mode=step2_sources, include_moving=include_moving)
    versioned = os.path.join(DATA_DIR, f"step2_sources_{model_version}.csv")
    provenance = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step1_model_version": model_version,
        "predictions_file": pred_csv,
        "objects_file": obj_csv,
        "object_classes": classes,
        "min_real_sources": min_real_sources,
        "step2_sources_mode": step2_sources,
        "include_moving": include_moving,
        "moving_min_rate_arcsec_per_min": moving_min_rate if with_history else None,
        "moving_min_sep_arcsec": moving_min_sep if with_history else None,
        "n_objects": int(df_sel["diaObjectId"].nunique()),
        "n_sources": int(len(df_sel)),
    }
    write_step2_source_list(df_sel, versioned, STEP2_CONFIG.source_list_file, provenance)
    print(f"\n[+] Step 2 input: {provenance['n_sources']} sources of {provenance['n_objects']} objects "
          f"(objects: {classes}, sources: {step2_sources})")
    print(f"    -> {versioned}\n    -> {STEP2_CONFIG.source_list_file} (used by step 2 pretrain / extract)")
    print(pd.crosstab(df_sel["object_class"], df_sel["decision"]).to_string())
    return df_obj, df_sel

def host_morph_cli(model_version: str = None, refresh_thresholds: bool = False, sn_list: str = "data/sn_spec_list.csv",
                   level: str = "object"):
    """
    step2: 母天体が点源か広がっているかを、Template の PSF に対する大きさの比で判定する (方針 A)。
    閾値は基準の恒星 (SIMBAD の恒星 / Gaia の視差 S/N > 5) の分布から S/N ごとに決め、JSON に固定する。
    model_version を指定すると、その固定モデルの step2 の判定 (no_host / bogus) と組み合わせた最終判定を出力する。
    """
    import json
    from sqlalchemy import create_engine, text
    from stamp_classifier.dataset import DBConfig
    from stamp_classifier.host_morphology import (compute_host_morph_cache, reference_star_mask, derive_thresholds,
                                                  classify_host, combine_with_rf, object_host_morph,
                                                  HOST_FEATURE_VERSION)
    src = pd.read_csv(STEP2_CONFIG.source_list_file)
    cache = compute_host_morph_cache(src, "cutouts")
    H = pd.DataFrame(cache["features"], columns=cache["names"])
    H.insert(0, "diaSourceId", cache["diaSourceIds"])
    H = src[["diaSourceId", "diaObjectId", "band"]].merge(H, on="diaSourceId", how="inner")

    print("Reading cross-matches (xm) from the database (read-only)...")
    engine = create_engine(DBConfig().url)
    with engine.connect() as conn:
        xm = pd.read_sql_query(text('SELECT "parent_diaSourceId" AS "diaSourceId", simbad_otype, '
                                    '"gaiadr3_Plx", "gaiadr3_e_Plx" FROM "xm"'), conn)
    H = H.merge(xm.drop_duplicates("diaSourceId"), on="diaSourceId", how="left")
    is_star = reference_star_mask(H["simbad_otype"].values, pd.to_numeric(H["gaiadr3_Plx"], errors="coerce").values,
                                  pd.to_numeric(H["gaiadr3_e_Plx"], errors="coerce").values)

    thr_file = os.path.join(DATA_DIR, "host_morph_thresholds_step2.json")
    thr = None
    if os.path.exists(thr_file) and not refresh_thresholds:
        with open(thr_file) as f:
            thr = json.load(f)
        if thr.get("feature_version", 1) != HOST_FEATURE_VERSION:
            print(f"[*] Thresholds in {thr_file} were derived for feature version {thr.get('feature_version', 1)}; "
                  f"re-deriving for version {HOST_FEATURE_VERSION}.")
            os.replace(thr_file, thr_file + f".v{thr.get('feature_version', 1)}")
            thr = None
        else:
            print(f"[*] Using fixed thresholds from {thr_file} (created {thr.get('created')}). Use --refresh_thresholds to re-derive.")
    if thr is None:
        thr = derive_thresholds(H["tmp_snr"].values, H["tmp_size_ratio"].values, is_star)
        thr["created"] = time.strftime("%Y-%m-%d %H:%M:%S")
        thr["n_reference_stars"] = int(is_star.sum())
        thr["feature_version"] = HOST_FEATURE_VERSION
        with open(thr_file, "w") as f:
            json.dump(thr, f, indent=2)
        print(f"[+] Derived thresholds from {int(is_star.sum())} reference stars -> {thr_file}")
    print(pd.DataFrame(thr["bins"]).round(3).to_string(index=False))

    H["host_morph_source"] = classify_host(H["tmp_snr"].values, H["tmp_size_ratio"].values, thr)
    # 天体単位: バンドごとの中央値を S/N で重み付けして平均した値で判定 (観測ごとの再サンプリングのばらつきを抑える)
    O = object_host_morph(H, thr)
    H = H.merge(O[["diaObjectId", "host_score", "host_morph_obj", "band_scores"]], on="diaObjectId", how="left")
    # グループ: A = 中心に母天体があり S/N が十分 (形で star / galaxy / uncertain を決める)
    #           B = それ以外 (母天体が暗い・中心に無い。no_host / real(uncertain) / galaxy(銀河の裾野) / bogus を専用モデルで決める)
    H["host_group"] = np.where(H["host_morph_obj"].isin(["point", "extended", "uncertain"]), "A", "B")
    if level == "object":
        H["host_morph"] = H["host_morph_obj"]
    else:
        H["host_morph"] = H["host_morph_source"]
    print(f"[*] Host shape decision level: {level}")
    qso = H["simbad_otype"].isin(["QSO", "AGN", "Sy1", "Sy2", "BLL", "Bla", "AG?", "QSO?"]).values
    H["reference"] = np.select([is_star, qso], ["star (SIMBAD/Gaia)", "QSO/AGN (SIMBAD)"], "")
    if sn_list and os.path.exists(sn_list):
        sn = pd.read_csv(sn_list)
        col = next((c for c in ["diaObjectId", "objId", "obj_id"] if c in sn.columns), None)
        if col:
            H.loc[H["diaObjectId"].isin(sn[col]), "reference"] = "SN host (spec.)"
    print("\nValidation (fraction of sources by host_morph):")
    v = H[H["reference"] != ""]
    print(pd.crosstab(v["reference"], v["host_morph"], normalize="index").round(3)
          .assign(n=v["reference"].value_counts()).to_string())
    vo = v.drop_duplicates("diaObjectId")
    print("Validation (fraction of objects by object-level host_morph):")
    print(pd.crosstab(vo["reference"], vo["host_morph_obj"], normalize="index").round(3)
          .assign(n=vo["reference"].value_counts()).to_string())
    print("\nAll step 2 sources:", H["host_morph"].value_counts().to_dict())
    print("Host group (A: central host with enough S/N, B: otherwise): sources",
          H["host_group"].value_counts().to_dict(), "| objects",
          H.drop_duplicates("diaObjectId")["host_group"].value_counts().to_dict())

    out_csv = os.path.join(DATA_DIR, "host_morph_step2.csv")
    keep = ["diaSourceId", "diaObjectId", "band", "host_group", "host_morph", "host_morph_source", "host_morph_obj", "host_score",
            "band_scores", "reference", "simbad_otype"] + list(cache["names"])
    H[keep].to_csv(out_csv, index=False)
    print(f"[+] Host morphology classification -> {out_csv}")

    if model_version:
        print(f"[*] --model_version {model_version} is ignored: the legacy step 2 RF is no longer used.")
    # 最終判定 (グループ A: 形, グループ B: 4 クラスの RF) を data/predictions_step2_final.csv に書く
    from stamp_classifier import group_b
    group_b.write_final()

def print_summary(step: int = 1):
    step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
    print(f"\n=======================================================")
    print(f"  Summary Report: Step {step} ({step_config.name})")
    print(f"=======================================================")
    print(f"Classes: {step_config.classes}")
    print(f"Image Types: {step_config.image_types} (Channels: {step_config.in_channels})")
    
    # チェックポイント確認
    ckpt = os.path.join(step_config.checkpoint_dir, "best_simclr_encoder.pt")
    print(f"SimCLR Checkpoint: {'EXISTS' if os.path.exists(ckpt) else 'NOT FOUND'}")

    # 特徴量キャッシュ確認
    emb_exists = os.path.exists(step_config.embeddings_file)
    print(f"Embeddings Cache: {'EXISTS' if emb_exists else 'NOT FOUND'}")
    if emb_exists:
        c = load_cached_embeddings(step)
        print(f"  Total Extracted Embeddings: {len(c['embeddings'])} (dim: {c['embeddings'].shape[1]})")

    # 目視ラベル確認
    df_labels = load_human_labels(step_config.labels_file)
    print(f"Human Labeled Stamps: {len(df_labels)}")
    if not df_labels.empty:
        print("  Label breakdown:")
        print(df_labels["class_name"].value_counts())

    # 教師あり分類器確認
    sup_model = os.path.join(step_config.checkpoint_dir, "best_supervised_model.pt")
    print(f"Supervised Classifier: {'TRAINED' if os.path.exists(sup_model) else 'NOT TRAINED'}")

    # ランダムフォレスト確認
    rf_model = os.path.join(step_config.checkpoint_dir, "random_forest.joblib")
    print(f"Random Forest Model: {'TRAINED' if os.path.exists(rf_model) else 'NOT TRAINED'}")
    print("=======================================================\n")

def main():
    parser = argparse.ArgumentParser(description="Postage Stamp Classifier Pipeline Runner")
    parser.add_argument("--step", type=int, default=1, choices=[1, 2], help="1: Difference real/bogus, 2: Host star/galaxy/no-host/bogus")
    parser.add_argument("--mode", type=str, default="summary", choices=["pretrain", "semi_supervised", "full_train", "train_supervised", "extract", "add_morphology", "active_loop", "build_eval_set", "evaluate", "audit_eval", "audit_report", "adjudicate_eval", "threshold_curve", "freeze_model", "classify_all", "aggregate_objects", "relabel", "relabel_report", "relabel_apply", "retrain_rf", "host_morph", "visit_check", "exclude_visit", "b_seed", "b_eval_set", "b_active_loop", "b_retrain", "b_evaluate", "b_classify", "summary"], help="Execution mode")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs (pretrain: 25, full_train: 50)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (pretrain: 256, full_train: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--use_labels", action="store_true", default=True, help="Enable semi-supervised simultaneous training with human labels (default: True)")
    parser.add_argument("--no_labels", action="store_false", dest="use_labels", help="Disable semi-supervised training (pure SimCLR)")
    parser.add_argument("--sup_weight", type=float, default=1.0, help="Weight for SupCon loss in semi-supervised training")
    parser.add_argument("--supcon_mode", type=str, default=None, choices=["mixed", "separate"], help="Semi-supervised contrastive mode: 'mixed' (labeled mixed into unlabeled batch, default) or 'separate' (legacy)")
    parser.add_argument("--holdout_frac", type=float, default=None, help="Fraction of labels held out (by diaObjectId) for per-epoch evaluation and best-model selection (default: 0.2, 0 disables)")
    parser.add_argument("--from_scratch", action="store_true", default=False, help="Train supervised model from scratch instead of fine-tuning from SimCLR weights")
    parser.add_argument("--loss_type", type=str, default="joint", choices=["joint", "ce", "supcon"], help="Loss function for full training ('joint', 'ce', 'supcon')")
    parser.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio for full training")
    parser.add_argument("--feedback_mode", type=str, default="fine_tune", choices=["fine_tune", "full_train", "rf_only"], help="Active learning feedback mode ('fine_tune', 'full_train', or 'rf_only': keep encoder/embeddings fixed and retrain RF only)")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples to label per active loop cycle")
    parser.add_argument("--eval_size", type=int, default=None, help="build_eval_set target: total size of the random evaluation set (default: 150), or with --eval_stratum, the number of evaluation labels in that stratum (default: 50)")
    parser.add_argument("--audit_n_correct", type=int, default=50, help="audit_eval: number of randomly chosen model-correct evaluation samples to re-review (all model errors are always included)")
    parser.add_argument("--model_version", type=str, default=None, help="freeze_model: name of the frozen RF+thresholds (e.g. v20261019). classify_all: use this frozen model and add a thresholded 'decision' column")
    parser.add_argument("--thresholds", type=str, default=None, help="freeze_model: per-class probability thresholds, e.g. 'positive_real=0.55,negative_real=0.75'")
    parser.add_argument("--min_real_sources", type=int, default=1, help="aggregate_objects: number of real-decision sources required to call an object 'real'")
    parser.add_argument("--step2_sources", type=str, default="real_uncertain", choices=["real_uncertain", "all", "best"], help="aggregate_objects: which sources of the selected objects go to step 2")
    parser.add_argument("--object_classes", type=str, default="real,uncertain", help="aggregate_objects: object classes passed to step 2 (comma separated)")
    parser.add_argument("--no_history", action="store_true", help="aggregate_objects: skip detection history from diaSource + prvDiaSources")
    parser.add_argument("--refresh_history", action="store_true", help="aggregate_objects: recompute the cached detection history")
    parser.add_argument("--moving_min_rate", type=float, default=0.2, help="aggregate_objects: minimum intra-night motion rate (arcsec/min) for moving-object candidates")
    parser.add_argument("--moving_min_sep", type=float, default=0.7, help="aggregate_objects: minimum intra-night separation (arcsec) for moving-object candidates")
    parser.add_argument("--include_moving", action="store_true", help="aggregate_objects: keep moving-object candidates in the step 2 input")
    parser.add_argument("--focus_classes", type=str, default=None, help="active_loop: focus on the boundary of two classes, e.g. 'galaxy,star'")
    parser.add_argument("--exclude_objects", type=str, default=None, help="active_loop: CSV of objects (diaObjectId / objId / obj_id column) to keep out of labeling candidates")
    parser.add_argument("--relabel_target", type=str, default="both", choices=["train", "eval", "both"], help="relabel: which labels to re-review")
    parser.add_argument("--show_previous", action="store_true", help="relabel: show the previous label on screen (default: blind)")
    parser.add_argument("--host_morph_level", type=str, default="object", choices=["object", "source"], help="host_morph: decide point/extended per object (S/N-weighted over bands; default) or per observation")
    parser.add_argument("--visit", type=int, default=None, help="exclude_visit: visit number to exclude")
    parser.add_argument("--reason", type=str, default="", help="exclude_visit: reason (recorded in data/excluded_visits.csv)")
    parser.add_argument("--refresh_thresholds", action="store_true", help="host_morph: re-derive the point/extended thresholds from reference stars")
    parser.add_argument("--eval_stratum", type=str, default=None, help="build_eval_set: sample within the stratum of this predicted class (e.g. negative_real). Strata are frozen from the current RF on first use")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of catalog records (for testing)")
    parser.add_argument("--out_csv", type=str, default=None, help="Output path for classify_all")
    args = parser.parse_args()

    if args.mode == "summary":
        print_summary(step=args.step)
    elif args.mode in ["pretrain", "semi_supervised"]:
        epochs = args.epochs if args.epochs is not None else 25
        batch_size = args.batch_size if args.batch_size is not None else 256
        use_labels = True if args.mode == "semi_supervised" else args.use_labels
        train_simclr(
            step=args.step,
            epochs=epochs,
            batch_size=batch_size,
            lr=args.lr,
            limit=args.limit,
            use_labels=use_labels,
            sup_weight=args.sup_weight,
            supcon_mode=args.supcon_mode,
            holdout_frac=args.holdout_frac
        )
    elif args.mode in ["full_train", "train_supervised"]:
        epochs = args.epochs if args.epochs is not None else 50
        batch_size = args.batch_size if args.batch_size is not None else 64
        train_supervised(
            step=args.step,
            epochs=epochs,
            batch_size=batch_size,
            lr=args.lr,
            from_scratch=args.from_scratch,
            loss_type=args.loss_type,
            val_split=args.val_split
        )
    elif args.mode == "extract":
        batch_size = args.batch_size if args.batch_size is not None else 256
        extract_embeddings(step=args.step, batch_size=batch_size, limit=args.limit)
    elif args.mode == "add_morphology":
        append_morphology_to_cache(step=args.step)
    elif args.mode == "active_loop":
        cfg = ActiveLearningConfig(
            feedback_mode=args.feedback_mode,
            focus_classes=[c.strip() for c in args.focus_classes.split(",")] if args.focus_classes else None,
            exclude_objects_file=args.exclude_objects or "")
        orch = ActiveLearningOrchestrator(step=args.step, config=cfg)
        orch.run_cycle(n_samples=args.n_samples)
    elif args.mode == "build_eval_set":
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.build_eval_set(n_target=args.eval_size, stratum=args.eval_stratum)
    elif args.mode in ["audit_eval", "audit_report"]:
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.audit_eval(n_correct=args.audit_n_correct, report_only=(args.mode == "audit_report"))
    elif args.mode == "adjudicate_eval":
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.adjudicate_eval()
    elif args.mode == "threshold_curve":
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.threshold_analysis(out_csv=args.out_csv)
    elif args.mode == "evaluate":
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.evaluate()
    elif args.mode == "freeze_model":
        if not args.model_version or not args.thresholds:
            parser.error("freeze_model requires --model_version and --thresholds")
        from stamp_classifier.model_registry import parse_thresholds
        step_config = STEP1_CONFIG if args.step == 1 else STEP2_CONFIG
        orch = ActiveLearningOrchestrator(step=args.step)
        orch.freeze_model(args.model_version, parse_thresholds(args.thresholds, step_config.classes))
    elif args.mode in ["relabel", "relabel_report", "relabel_apply"]:
        from stamp_classifier.relabel import run_relabel, relabel_report, apply_relabel
        step_config = STEP1_CONFIG if args.step == 1 else STEP2_CONFIG
        if args.mode == "relabel":
            orch = ActiveLearningOrchestrator(step=args.step)
            run_relabel(step_config, orch.source_meta(), target=args.relabel_target, show_previous=args.show_previous)
        elif args.mode == "relabel_report":
            relabel_report(step_config)
        else:
            apply_relabel(step_config)
    elif args.mode.startswith("b_"):
        from stamp_classifier import group_b
        if args.mode == "b_seed":
            group_b.seed_from_step2()
        elif args.mode == "b_eval_set":
            group_b.build_eval_set(n_target=args.eval_size or 150)
        elif args.mode == "b_active_loop":
            group_b.active_loop(n_samples=args.n_samples)
        elif args.mode == "b_retrain":
            group_b.retrain()
        elif args.mode == "b_evaluate":
            import joblib
            group_b.evaluate(joblib.load(group_b.MODEL_FILE), group_b.load_group_b())
        elif args.mode == "b_classify":
            group_b.classify()
            group_b.write_final()
    elif args.mode == "visit_check":
        from stamp_classifier.visit_quality import report_suspicious_visits
        version = args.model_version or "v20261019"
        report_suspicious_visits(os.path.join(DATA_DIR, f"predictions_step1_{version}.csv"))
    elif args.mode == "exclude_visit":
        if args.visit is None:
            parser.error("exclude_visit requires --visit")
        from stamp_classifier.visit_quality import add_excluded_visit, excluded_source_ids
        add_excluded_visit(args.visit, "", args.reason)
        excluded_source_ids(refresh=True)
    elif args.mode == "host_morph":
        host_morph_cli(model_version=args.model_version, refresh_thresholds=args.refresh_thresholds,
                       level=args.host_morph_level)
    elif args.mode == "retrain_rf":
        ActiveLearningOrchestrator(step=args.step).retrain_rf()
    elif args.mode == "aggregate_objects":
        if not args.model_version:
            parser.error("aggregate_objects requires --model_version")
        aggregate_objects_cli(args.model_version, min_real_sources=args.min_real_sources,
                              step2_sources=args.step2_sources, object_classes=args.object_classes,
                              with_history=not args.no_history, refresh_history=args.refresh_history,
                              moving_min_rate=args.moving_min_rate, moving_min_sep=args.moving_min_sep,
                              include_moving=args.include_moving)
    elif args.mode == "classify_all":
        classify_all(step=args.step, out_csv=args.out_csv, model_version=args.model_version)

if __name__ == "__main__":
    main()
