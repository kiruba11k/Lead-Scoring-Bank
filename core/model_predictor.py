"""
Model Predictor
- Loads model.pkl + metadata.json
- Ensures feature columns match exactly
- Returns prediction + confidence + probabilities
- Provides feature importance + explanation
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Optional


class ModelPredictor:
    def __init__(self, model_path: str = "models/model.pkl", meta_path: str = "models/metadata.json"):
        self.model_path = model_path
        self.meta_path = meta_path

        self.model = None
        self.metadata = None
        self.feature_names = None
        self.reverse_mapping = None

        self._load()

    def _load(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"Metadata not found: {self.meta_path}")

        self.model = joblib.load(self.model_path)

        with open(self.meta_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        self.reverse_mapping = self.metadata.get("reverse_mapping", {})

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align features to metadata order.
        Fill missing columns with 0.
        Keep NaN activity_days -> replace with 999 (like training)
        """
        X = features_df.copy()

        # add missing columns
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        # keep only expected cols in correct order
        X = X[self.feature_names]

        # handle NaN activity_days like training (fill 999 then clip)
        if "activity_days" in X.columns:
            X["activity_days"] = pd.to_numeric(X["activity_days"], errors="coerce")
            X["activity_days"] = X["activity_days"].fillna(999).clip(0, 180)

            X["is_active_week"] = (X["activity_days"] <= 7).astype(int)
            X["is_active_month"] = (X["activity_days"] <= 30).astype(int)

        # ensure numeric
        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        return X

    def predict(self, features_df: pd.DataFrame) -> Optional[Dict]:
        try:
            X = self._prepare_features(features_df)

            probs = self.model.predict_proba(X)[0]
            pred_idx = int(np.argmax(probs))
            label = self.reverse_mapping.get(str(pred_idx), str(pred_idx))

            confidence = float(np.max(probs))

            # probabilities dict
            prob_dict = {}
            for idx, p in enumerate(probs):
                lab = self.reverse_mapping.get(str(idx), str(idx))
                prob_dict[lab] = float(p)

            return {
                "priority": label,
                "confidence": confidence,
                "probabilities": prob_dict
            }

        except Exception as e:
            print(f"Prediction error: {e}")
            return None

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Returns model feature importance (XGBoost).
        """
        if self.model is None:
            return {}

        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
            importance_map = dict(zip(self.feature_names, imp))
            # sort desc
            importance_map = dict(sorted(importance_map.items(), key=lambda x: x[1], reverse=True))
            return importance_map

        return {}

    def explain_prediction(self, features_df: pd.DataFrame, top_n: int = 5) -> Dict:
        """
        Dynamic explanation based on top feature importance * value.
        """
        X = self._prepare_features(features_df)
        importance = self.get_feature_importance()

        if not importance:
            return {"top_reasons": []}

        reasons = []
        for feat, imp in list(importance.items())[:30]:
            val = float(X.iloc[0][feat])
            score = imp * abs(val)
            reasons.append((feat, val, float(imp), float(score)))

        reasons = sorted(reasons, key=lambda x: x[3], reverse=True)[:top_n]

        return {
            "top_reasons": [
                {"feature": f, "value": v, "importance": imp, "impact_score": sc}
                for f, v, imp, sc in reasons
            ]
        }
