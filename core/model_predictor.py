"""
Model Predictor
- Loads model.pkl + metadata.json
- Ensures features match trained order
- Converts all values to numeric before predict_proba
- Supports feature importance extraction (if model has feature_importances_)
"""

import json
import joblib
import numpy as np
import pandas as pd
import os


class ModelPredictor:
    def __init__(self, model_path="models/model.pkl", metadata_path="models/metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def predict(self, features_df: pd.DataFrame):
        if self.model is None or self.metadata is None:
            return None

        feature_names = self.metadata.get("feature_names", [])
        reverse_mapping = self.metadata.get("reverse_mapping", {})

        if not feature_names:
            return None

        # Ensure all required columns exist
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        # Reorder exactly as training
        X = features_df[feature_names].copy()

        # Force numeric
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        # Predict probabilities
        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))

        label = reverse_mapping.get(str(pred_class), "UNKNOWN")

        prob_map = {}
        for i, p in enumerate(probs):
            lab = reverse_mapping.get(str(i), str(i))
            prob_map[lab] = float(p)

        return {
            "priority": label,
            "confidence": float(np.max(probs)),
            "probabilities": prob_map,
        }

    def get_feature_importance(self):
        """
        Returns dict(feature_name -> importance) sorted descending.
        Works only if model has feature_importances_ (Tree models).
        """
        if self.model is None or self.metadata is None:
            return None

        if not hasattr(self.model, "feature_importances_"):
            return None

        feature_names = self.metadata.get("feature_names", [])
        importances = getattr(self.model, "feature_importances_", None)

        if importances is None:
            return None

        if len(feature_names) != len(importances):
            return None

        pairs = list(zip(feature_names, importances))
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        return {k: float(v) for k, v in pairs}
