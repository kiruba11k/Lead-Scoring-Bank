import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Dict

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

        # Ensure all columns exist
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0

        # IMPORTANT: reorder exactly as trained model
        X = features_df[feature_names].copy()

        # Convert all to numeric safely
        for c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

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

    
