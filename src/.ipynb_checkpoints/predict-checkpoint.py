import os
import joblib
import numpy as np

class LCAPredictor:
    """
    Loads models, imputers, and label encoders from the model directory.
    Provides unified prediction and preprocessing utilities.
    """
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.models = []
        self.imputers = {}
        self.label_encoders = None
        self._load_all()

    def _load_all(self):
        """
        Loads models, imputers, and label encoders from model_dir.
        """
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")

        for file in os.listdir(self.model_dir):
            file_path = os.path.join(self.model_dir, file)

            # Load ML models (.pkl)
            if file.endswith(".pkl") and "imputer" not in file and "label_encoders" not in file:
                model = joblib.load(file_path)
                self.models.append(model)

            # Load imputers
            elif "imputer" in file and file.endswith(".pkl"):
                name = file.replace(".pkl", "")
                self.imputers[name] = joblib.load(file_path)

            # Load label encoders
            elif file == "label_encoders.pkl":
                self.label_encoders = joblib.load(file_path)

        print(f"[LCAPredictor] Loaded {len(self.models)} model(s), "
              f"{len(self.imputers)} imputers, "
              f"{'with' if self.label_encoders else 'without'} label encoders.")

    def predict(self, X):
        """
        Runs predictions using all loaded models.
        If multiple models exist, averages their outputs.
        """
        if not self.models:
            raise ValueError("No models loaded. Check your model directory.")

        all_preds = [model.predict(X) for model in self.models]

        if len(all_preds) > 1:
            return np.mean(all_preds, axis=0)
        return all_preds[0]


def make_prediction(predictor, df_imputed):
    """
    Wrapper to call predictor.predict() on preprocessed data.
    """
    predictions = predictor.predict(df_imputed)
    return np.array(predictions)
