import os
import json
import pickle
import numpy as np


class ModelPredictor:
    def __init__(self, model_path="models/model.pkl", metadata_path="models/metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            return
        if not os.path.exists(self.metadata_path):
            return

        try:
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)
        except Exception:
            self.model = None
            self.metadata = None

    def predict(self, features_df):
        if self.model is None or self.metadata is None:
            return None

        feature_names = self.metadata.get("feature_names", [])
        reverse_mapping = self.metadata.get("reverse_mapping", {})

        if not feature_names:
            return None

        # ensure all columns exist
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        X = features_df[feature_names]

        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))
        label = reverse_mapping.get(str(pred_class), "UNKNOWN")

        prob_map = {}
        for i, p in enumerate(probs):
            lab = reverse_mapping.get(str(i), str(i))
            prob_map[lab] = float(p)

        confidence = float(np.max(probs))

        # simple reasons from feature importances (top 5)
        reasons = []
        try:
            importances = self.model.feature_importances_
            idxs = np.argsort(importances)[::-1][:5]
            for idx in idxs:
                fname = feature_names[idx]
                reasons.append(
                    {"feature": fname, "value": float(X.iloc[0][fname]), "importance": float(importances[idx])}
                )
        except Exception:
            pass

        return {
            "priority": label,
            "confidence": confidence,
            "probabilities": prob_map,
            "reasons": reasons
        }
