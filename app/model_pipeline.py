import pickle
import pandas as pd
from pathlib import Path
from app.preprocessing import preprocessing_raw

class ModelPipeline:
    def __init__(self):
        self.model = None
        self.features = None
        self.threshold = 0.5
        self.model_dir = Path("model")

    def load(self):
        # Load pipeline (preprocess + PCA + model)
        with open(self.model_dir / "pipeline.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Load feature list
        with open(self.model_dir / "features.pkl", "rb") as f:
            self.features = pickle.load(f)

        # Load threshold
        thr_path = self.model_dir / "threshold.pkl"
        if thr_path.exists():
            with open(thr_path, "rb") as f:
                self.threshold = pickle.load(f)

    def predict(self, df_raw):
        # (1) PREPROCESS RAW INPUT
        df = preprocessing_raw(df_raw)

        # (2) FILTER ONLY TRAINING FEATURES
        df = df[self.features]

        # (3) PREDICT PROBA USING LOADED PIPELINE
        proba = self.model.predict_proba(df)[:, 1]

        # (4) APPLY THRESHOLD
        preds = (proba >= self.threshold).astype(int)

        return preds, proba
