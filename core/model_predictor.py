import os
import json
import joblib
import numpy as np
import pandas as pd


class ModelPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(base_dir)

        self.model_path = os.path.join(project_root, "models", "model.pkl")
        self.meta_path = os.path.join(project_root, "models", "metadata.json")

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")

        self.model = joblib.load(self.model_path)

        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        self.feature_names = self.meta["feature_names"]
        self.label_mapping = self.meta["label_mapping"]
        self.reverse_mapping = self.meta["reverse_mapping"]
        self.rev_classes = self.meta["revenue_encoder_classes"]

    def _encode_revenue_category(self, value: str) -> int:
        value = str(value)
        if value in self.rev_classes:
            return int(self.rev_classes.index(value))
        return 0

    def predict(self, features_df: pd.DataFrame) -> dict:
        if features_df is None or features_df.empty:
            raise ValueError("Empty features passed to predictor")

        X = features_df.copy()

        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_names]

        # Encode revenue_category
        if "revenue_category" in X.columns:
            X["revenue_category"] = X["revenue_category"].apply(self._encode_revenue_category).astype(int)

        pred = int(self.model.predict(X)[0])
        proba = self.model.predict_proba(X)[0]

        # reverse mapping keys may be strings or ints
        priority = self.reverse_mapping.get(str(pred), self.reverse_mapping.get(pred, "UNKNOWN"))

        prob_dict = {}
        for label, idx in self.label_mapping.items():
            prob_dict[label] = float(proba[int(idx)])

        return {
            "priority": priority,
            "numeric_score": pred,
            "confidence": float(np.max(proba)),
            "probabilities": prob_dict,
        }
