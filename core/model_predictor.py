import json
import joblib
import numpy as np
import pandas as pd
import shap
from pathlib import Path


class ModelPredictor:
    def __init__(self, model_path="models/model.pkl", metadata_path="models/metadata.json"):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.model = None
        self.metadata = None
        self._load()

    def _load(self):
        if Path(self.model_path).exists():
            self.model = joblib.load(self.model_path)

        if Path(self.metadata_path).exists():
            with open(self.metadata_path, "r") as f:
                self.metadata = json.load(f)

    def _ensure_feature_order(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.metadata:
            return df

        expected = self.metadata.get("feature_names", [])
        if not expected:
            return df

        # Add missing columns with 0
        for col in expected:
            if col not in df.columns:
                df[col] = 0

        # Drop extra columns
        df = df[expected]
        return df

    def explain_prediction(self, X_row: pd.DataFrame, top_n: int = 6) -> list:
        """
        SHAP explanation for predicted class.
        Returns list of top contributing features.
        """
        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_row)

            pred_class = int(self.model.predict(X_row)[0])

            # Multi-class: shap_values is list
            class_shap = shap_values[pred_class][0]

            feature_names = list(X_row.columns)
            feature_vals = X_row.iloc[0].to_dict()

            contributions = []
            for i, f in enumerate(feature_names):
                contributions.append({
                    "feature": f,
                    "value": feature_vals.get(f),
                    "impact": float(class_shap[i])
                })

            contributions = sorted(contributions, key=lambda x: abs(x["impact"]), reverse=True)
            return contributions[:top_n]

        except Exception:
            return []

    def predict(self, features_df: pd.DataFrame) -> dict:
        if self.model is None:
            return None

        features_df = self._ensure_feature_order(features_df)

        proba = self.model.predict_proba(features_df)[0]
        pred_class = int(np.argmax(proba))

        reverse_mapping = None
        if self.metadata:
            reverse_mapping = self.metadata.get("reverse_mapping")

        if reverse_mapping:
            priority = reverse_mapping.get(str(pred_class), reverse_mapping.get(pred_class, "UNKNOWN"))
        else:
            mapping = {0: "COLD", 1: "COOL", 2: "WARM", 3: "HOT"}
            priority = mapping.get(pred_class, "UNKNOWN")

        probabilities = {
            "COLD": float(proba[0]),
            "COOL": float(proba[1]),
            "WARM": float(proba[2]),
            "HOT": float(proba[3]),
        }

        confidence = float(np.max(proba))

        reasons = self.explain_prediction(features_df, top_n=6)

        return {
            "priority": priority,
            "confidence": confidence,
            "probabilities": probabilities,
            "reasons": reasons
        }

    def get_feature_importance(self) -> dict:
        """
        Optional: returns model feature importance if available.
        """
        try:
            booster = self.model.get_booster()
            score = booster.get_score(importance_type="gain")
            sorted_score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))
            return sorted_score
        except Exception:
            return {}
