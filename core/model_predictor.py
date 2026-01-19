import os
import json
import pickle
import numpy as np


class ModelPredictor:
    def __init__(self):
        self.model_path = "models/banking_scoring_model_20260118_132024.pkl"
        self.meta_path = "models/banking_scoring_model_20260118_132024_metadata.json"

        self.model = None
        self.metadata = None

        self._load()

    def _load(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, "rb") as f:
                self.model = pickle.load(f)

        if os.path.exists(self.meta_path):
            with open(self.meta_path, "r") as f:
                self.metadata = json.load(f)

    def predict(self, features_df):
        if self.model is None or self.metadata is None:
            return None

        # Ensure correct feature order
        feature_names = self.metadata.get("feature_names", [])
        X = features_df[feature_names]

        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))

        reverse_mapping = self.metadata.get("reverse_mapping", {})
        label = reverse_mapping.get(str(pred_class), "UNKNOWN")

        prob_map = {}
        for i, p in enumerate(probs):
            lab = reverse_mapping.get(str(i), str(i))
            prob_map[lab] = float(p)

        # Confidence = max prob
        confidence = float(np.max(probs))

        # Dynamic reasons (top contributing features)
        reasons = []
        try:
            importances = self.model.feature_importances_
            idxs = np.argsort(importances)[::-1][:5]
            for idx in idxs:
                fname = feature_names[idx]
                reasons.append(f"{fname} influenced score (value={X.iloc[0][fname]})")
        except:
            pass

        return {
            "priority": label,
            "confidence": confidence,
            "probabilities": prob_map,
            "reasons": reasons
        }
