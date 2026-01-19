import json
import joblib
import numpy as np
import pandas as pd


class ModelPredictor:
    def __init__(self, model_path="models/model.pkl", metadata_path="models/metadata.json"):
        self.model = joblib.load(model_path)

        with open(metadata_path, "r") as f:
            self.meta = json.load(f)

        self.feature_names = self.meta["feature_names"]
        self.reverse_mapping = {int(k): v for k, v in self.meta["reverse_mapping"].items()}

    def predict(self, features_df: pd.DataFrame):
        # align columns
        for col in self.feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        features_df = features_df[self.feature_names].copy()

        probs = self.model.predict_proba(features_df)[0]
        pred_class = int(np.argmax(probs))

        priority = self.reverse_mapping[pred_class]

        labels = ["COLD", "COOL", "WARM", "HOT"]
        prob_dict = {labels[i]: float(probs[i]) for i in range(len(labels))}

        confidence = float(np.max(probs))

        return {
            "priority": priority,
            "confidence": confidence,
            "probabilities": prob_dict
        }
