import os
import json
import joblib
import numpy as np
import pandas as pd


class ModelPredictor:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "..", "models")

        self.model_path = os.path.join(models_dir, "model.pkl")
        self.metadata_path = os.path.join(models_dir, "metadata.json")

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
            raise ValueError("feature_names missing in metadata.json")

    def _align_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Ensures incoming features match training feature order exactly.
        Missing columns filled with 0.
        Extra columns removed.
        """
        if isinstance(features, dict):
            features = pd.DataFrame([features])

        # Ensure all expected columns exist
        for col in self.feature_names:
            if col not in features.columns:
                features[col] = 0

        # Remove extra columns
        features = features[self.feature_names]

        # Convert everything numeric safely
        features = features.apply(pd.to_numeric, errors="coerce").fillna(0)

        return features

    def predict(self, features):
        """
        Predict priority class + confidence + probability distribution
        """
        X = self._align_features(features)

        probs = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(probs))

        priority = self.reverse_mapping.get(str(pred_class), self.reverse_mapping.get(pred_class, "UNKNOWN"))
        confidence = float(np.max(probs))

        labels = ["COLD", "COOL", "WARM", "HOT"]
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return {
            "priority": priority,
            "confidence": confidence,
            "probabilities": prob_dict
        }

    def get_feature_importance(self):
        """
        Return feature importance in descending order (if available)
        """
        try:
            importance = self.model.feature_importances_
            imp_dict = dict(zip(self.feature_names, importance))
            imp_dict = dict(sorted(imp_dict.items(), key=lambda x: x[1], reverse=True))
            return imp_dict
        except Exception:
            return {}
