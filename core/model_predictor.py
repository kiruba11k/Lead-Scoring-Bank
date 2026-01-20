"""
Model Predictor for Lead Scoring (COLD / COOL / WARM / HOT)

- Loads trained XGBoost model (model.pkl)
- Loads metadata (metadata.json)
- Ensures input features match model training feature_names exactly
- Returns prediction + confidence + probability distribution
- Provides dynamic "WHY predicted" using feature importance
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List


class ModelPredictor:
    def __init__(
        self,
        model_path: str = "models/model.pkl",
        metadata_path: str = "models/metadata.json",
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path

        self.model = None
        self.metadata = None
        self.feature_names = []
        self.reverse_mapping = {}

        self._load_all()

    # -----------------------------
    # Load Model + Metadata
    # -----------------------------
    def _load_all(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.model = joblib.load(self.model_path)

        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.feature_names = self.metadata.get("feature_names", [])
        self.reverse_mapping = self.metadata.get("reverse_mapping", {})

        if not self.feature_names:
            raise ValueError("metadata.json missing feature_names")

        if not self.reverse_mapping:
            raise ValueError("metadata.json missing reverse_mapping")

    # -----------------------------
    # Feature Alignment
    # -----------------------------
    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Align input DataFrame columns to match training feature order.
        Missing columns -> filled with 0.
        Extra columns -> removed.
        """
        X = X.copy()

        # Add missing cols
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0

        # Keep only training columns in correct order
        X = X[self.feature_names]

        # Ensure numeric dtype
        for col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        return X

    # -----------------------------
    # Feature Importance / Explanation
    # -----------------------------
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Returns global feature importance from XGBoost model.
        """
        if not hasattr(self.model, "feature_importances_"):
            return pd.DataFrame(columns=["feature", "importance"])

        importances = self.model.feature_importances_
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances
        }).sort_values("importance", ascending=False)

        return df_imp

    def explain_prediction(
        self,
        X_row: pd.DataFrame,
        top_n: int = 6
    ) -> pd.DataFrame:
        """
        Provide dynamic reasons based on:
        - feature importance
        - feature value in the input row
        """
        X_row = self._align_features(X_row)

        imp_df = self.get_feature_importance()
        if imp_df.empty:
            return pd.DataFrame(columns=["feature", "value", "importance"])

        # Merge with input values
        values = X_row.iloc[0].to_dict()

        imp_df["value"] = imp_df["feature"].map(values)

        # Keep only features that actually contribute (importance > 0)
        imp_df = imp_df[imp_df["importance"] > 0]

        # Prefer features that have non-zero values
        imp_df["abs_value"] = imp_df["value"].abs()
        imp_df = imp_df.sort_values(
            by=["importance", "abs_value"],
            ascending=[False, False]
        )

        return imp_df[["feature", "value", "importance"]].head(top_n)

    # -----------------------------
    # Predict
    # -----------------------------
    def predict(
        self,
        X: pd.DataFrame,
        return_debug: bool = True
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Predict lead label + confidence.
        Returns:
          result dict + debug dict
        """
        if X is None or len(X) == 0:
            return {"error": "Empty input features"}, {}

        # Align features exactly
        X_aligned = self._align_features(X)

        # Debug values BEFORE sending to model
        debug_info = {}
        if return_debug:
            debug_info["features_before_model"] = X_aligned.iloc[0].to_dict()

        # Predict probabilities
        try:
            probs = self.model.predict_proba(X_aligned)[0]
        except Exception as e:
            return {"error": f"Model returned no prediction: {str(e)}"}, debug_info

        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        # Convert class index -> label
        pred_label = self.reverse_mapping.get(str(pred_class), "UNKNOWN")

        # Probability distribution
        prob_dist = {}
        for class_idx, p in enumerate(probs):
            label = self.reverse_mapping.get(str(class_idx), str(class_idx))
            prob_dist[label] = float(p)

        result = {
            "priority": pred_label,
            "confidence": round(confidence * 100, 2),
            "score": round(confidence * 100),
            "probability_distribution": prob_dist
        }

        # Explanation (WHY predicted)
        try:
            explanation_df = self.explain_prediction(X_aligned, top_n=6)
            result["top_reasons"] = explanation_df.to_dict(orient="records")
        except Exception:
            result["top_reasons"] = []

        return result, debug_info
