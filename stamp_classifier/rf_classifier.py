"""
Random Forest Classifier on Latent Embeddings
Trains ensemble classifier, outputs class probabilities, and computes uncertainty metrics for active learning.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import STEP1_CONFIG, STEP2_CONFIG, RFConfig, MODELS_DIR

class StampRandomForest:
    def __init__(self, step: int = 1, config: Optional[RFConfig] = None):
        self.step = step
        self.step_config = STEP1_CONFIG if step == 1 else STEP2_CONFIG
        self.config = config or RFConfig()
        self.classes = self.step_config.classes
        self.model_path = os.path.join(self.step_config.checkpoint_dir, "random_forest.joblib")
        
        self.clf = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_split=self.config.min_samples_split,
            class_weight=self.config.class_weight,
            oob_score=True,
            n_jobs=self.config.n_jobs,
            random_state=self.config.random_state
        )
        self.is_fitted = False

    def train(self, X: np.ndarray, y: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        X: (N_labeled, 128)
        y: (N_labeled,) クラスインデックス (0 ... len(classes)-1)
        """
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            raise ValueError(f"At least 2 distinct classes required for training. Found: {unique_classes}")

        print(f"Training Random Forest on {len(X)} labeled stamps across classes: {unique_classes}")
        self.clf.fit(X, y, sample_weight=sample_weight)
        self.is_fitted = True

        oob_acc = self.clf.oob_score_ if hasattr(self.clf, "oob_score_") else None
        
        # 構造パラメータの重要度ランキングを出力
        if hasattr(self.clf, "feature_importances_") and X.shape[1] > 128:
            from .morphology import MORPH_FEATURE_NAMES
            importances = self.clf.feature_importances_
            morph_imp = importances[128:128+len(MORPH_FEATURE_NAMES)]
            sorted_idx = np.argsort(morph_imp)[::-1]
            print("  [*] Top Morphological Feature Importances:")
            for i in sorted_idx[:5]:
                print(f"      - {MORPH_FEATURE_NAMES[i]:20s}: {morph_imp[i]*100:.2f}%")

        # 5-fold CV (十分なサンプルがある場合)
        min_class_count = min(np.bincount(y, minlength=len(self.classes))[unique_classes])
        cv_score = None
        if min_class_count >= 3 and len(X) >= 15:
            n_splits = min(5, min_class_count)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf, X, y, cv=skf, scoring="accuracy")
            cv_score = float(np.mean(cv_scores))

        os.makedirs(self.step_config.checkpoint_dir, exist_ok=True)
        joblib.dump(self.clf, self.model_path)
        
        metrics = {
            "n_samples": len(X),
            "oob_accuracy": float(oob_acc) if oob_acc is not None else 0.0,
            "cv_accuracy": cv_score
        }
        print(f"[+] RF Trained successfully! OOB Accuracy: {metrics['oob_accuracy']:.4f}, CV Accuracy: {metrics['cv_accuracy']}")
        return metrics

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        全天体に対する確率予測と能動学習用不確実性スコアを算出
        Returns:
            preds: (N,) 最も確からしいクラスID
            probs: (N, C) 各クラスの確率
            margins: (N,) Top1確率 - Top2確率 (値が小さいほど境界付近で不確実)
            entropies: (N,) シャノンエントロピー (値が大きいほどモデルの迷いが強い)
        """
        if not self.is_fitted:
            if os.path.exists(self.model_path):
                self.clf = joblib.load(self.model_path)
                self.is_fitted = True
            else:
                raise RuntimeError("Random Forest is not fitted and no saved model found.")

        probs = self.clf.predict_proba(X)
        preds = np.argmax(probs, axis=1)

        # Margin: Top1 - Top2
        sorted_probs = np.sort(probs, axis=1)[:, ::-1]
        margins = sorted_probs[:, 0] - (sorted_probs[:, 1] if probs.shape[1] > 1 else 0.0)

        # Entropy: - sum(p * log2(p))
        eps = 1e-12
        entropies = -np.sum(probs * np.log2(probs + eps), axis=1)

        return preds, probs, margins, entropies
