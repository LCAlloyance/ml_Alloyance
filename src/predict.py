
import os
import joblib
import numpy as np

class LCAPredictor:
    """
    Loads one or multiple trained models from a directory
    and provides a unified predict() interface.
    """
    def __init__(self, model_dir):
        self.models = []
        self._load_models(model_dir)

    def _load_models(self, model_dir):
        for file in os.listdir(model_dir):
            if file.endswith(".pkl"):
                model_path = os.path.join(model_dir, file)
                model = joblib.load(model_path)
                self.models.append(model)
        print(f"[LCAPredictor] Loaded {len(self.models)} model(s) from {model_dir}")

    def predict(self, X):
        """
        Runs predictions using all loaded models.
        If multiple models exist, averages their outputs.
        """
        if not self.models:
            raise ValueError("No models loaded. Check your model directory.")

        # Run prediction for each model
        all_preds = [model.predict(X) for model in self.models]

        # If multiple models, average predictions
        if len(all_preds) > 1:
            return np.mean(all_preds, axis=0)
        return all_preds[0]  # Single model case


def make_prediction(predictor, df_imputed):
    """
    Wrapper function to call predictor.predict() on imputed data.
    """
    predictions = predictor.predict(df_imputed)

    # Ensure it's always a NumPy array for consistency
    predictions = np.array(predictions)
    return predictions
