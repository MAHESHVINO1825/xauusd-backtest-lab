"""Average-probability ensemble of several fitted classifiers (saved with joblib)."""
import numpy as np


class AvgEnsemble:
    def __init__(self, models):
        self.models = list(models)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)
