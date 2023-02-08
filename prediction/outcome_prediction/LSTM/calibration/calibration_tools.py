import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.base import ClassifierMixin as ScikitClassifier
from sklearn.calibration import calibration_curve
from keras import Model as KerasBaseModel
from sklearn.metrics import matthews_corrcoef


class SigmoidCalibrator:
    def __init__(self, prob_pred, prob_true):
        prob_pred, prob_true = self._filter_out_of_domain(prob_pred, prob_true)
        prob_true = np.log(prob_true / (1 - prob_true))
        self.regressor = LinearRegression().fit(
            prob_pred.reshape(-1, 1), prob_true.reshape(-1, 1)
        )

    def calibrate(self, probabilities):
        return 1 / (1 + np.exp(-self.regressor.predict(probabilities.reshape(-1, 1)).flatten()))

    def _filter_out_of_domain(self, prob_pred, prob_true):
        filtered = list(zip(*[p for p in zip(prob_pred, prob_true) if 0 < p[1] < 1]))
        return np.array(filtered)


class IsotonicCalibrator:
    def __init__(self, prob_pred, prob_true):
        self.regressor = IsotonicRegression(out_of_bounds="clip")
        self.regressor.fit(prob_pred, prob_true)

    def calibrate(self, probabilities):
        return self.regressor.predict(probabilities)


class CalibratableModelFactory:
    def get_model(self, base_model):
        if isinstance(base_model, ScikitClassifier):
            return ScikitModel(base_model)
        elif isinstance(base_model, KerasBaseModel):
            return KerasModel(base_model)
        raise ValueError("Unsupported model passed as an argument")


class CalibratableModelMixin:
    def __init__(self, model):
        self.model = model
        self.name = model.__class__.__name__
        self.sigmoid_calibrator = None
        self.isotonic_calibrator = None
        self.calibrators = {
            "sigmoid": None,
            "isotonic": None,
        }

    def calibrate(self, X, y):
        predictions = self.predict(X)
        prob_true, prob_pred = calibration_curve(y, predictions, n_bins=10)
        self.calibrators["sigmoid"] = SigmoidCalibrator(prob_pred, prob_true)
        self.calibrators["isotonic"] = IsotonicCalibrator(prob_pred, prob_true)

    def calibrate_probabilities(self, probabilities, method="isotonic"):
        if method not in self.calibrators:
            raise ValueError("Method has to be either 'sigmoid' or 'isotonic'")
        if self.calibrators[method] is None:
            raise ValueError("Fit the calibrators first")
        return self.calibrators[method].calibrate(probabilities)

    def predict_calibrated(self, X, method="isotonic"):
        return self.calibrate_probabilities(self.predict(X), method)

    def score(self, X, y):
        return self._get_accuracy(y, self.predict(X)), self._get_MCC(y, self.predict(X))

    def score_calibrated(self, X, y, method="isotonic"):
        return self._get_accuracy(y, self.predict_calibrated(X, method)), self._get_MCC(y, self.predict_calibrated(X, method))

    def _get_accuracy(self, y, preds):
        return np.mean(np.equal(y.astype(np.bool), preds >= 0.5))

    def _get_MCC(self, y, preds):
        return matthews_corrcoef(y, preds >= 0.5)


class KerasModel(CalibratableModelMixin):
    def train(self, X, y, n_epochs, batch_size):
        self.model.fit(X, y, epochs=n_epochs,
                                   batch_size=batch_size,
                                   verbose=0)

    def predict(self, X):
        return self.model.predict(X).flatten()

class ScikitModel(CalibratableModelMixin):
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]