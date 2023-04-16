import numpy as np
import torch as ch
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
from sklearn.base import ClassifierMixin as ScikitClassifier
from sklearn.calibration import calibration_curve
from keras import Model as KerasBaseModel
from sklearn.metrics import matthews_corrcoef

from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel


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
    def get_model(self, base_model, trainer=None):
        if isinstance(base_model, ScikitClassifier):
            return ScikitModel(base_model)
        elif isinstance(base_model, KerasBaseModel):
            return KerasModel(base_model)
        elif isinstance(base_model, LitModel):
            return LightningModel(base_model, trainer)
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

class LightningModel(CalibratableModelMixin):
    def __init__(self, model, trainer):
        super().__init__(model)
        self.trainer = trainer

    def predict(self, X):
        return ch.sigmoid(self.trainer.predict(self.model, X)[0])[:, -1]

class ScikitModel(CalibratableModelMixin):
    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]


import torch
from torch import nn, optim
from torch.nn import functional as F


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """
    def __init__(self, model, use_gpu=True):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.use_gpu = use_gpu


    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        if self.use_gpu:
            self.cuda()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            ece_criterion = _ECELoss().cuda()
        else:
            nll_criterion = nn.CrossEntropyLoss()
            ece_criterion = _ECELoss()

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for input, label in valid_loader:
                if self.use_gpu:
                    input = input.cuda()
                logits = self.model(input)
                logits_list.append(logits)
                labels_list.append(label)
            if self.use_gpu:
                logits = torch.cat(logits_list).cuda()
                labels = torch.cat(labels_list).cuda()
            else:
                logits = torch.cat(logits_list)
                labels = torch.cat(labels_list)

        # add axis
        logits = logits.squeeze()

        # Calculate NLL and ECE before temperature scaling
        # before_temperature_nll = nll_criterion(logits, labels).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
        # print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.temperature_scale(logits), labels)
            loss.backward()
            return loss
        optimizer.step(eval)

        # Calculate NLL and ECE after temperature scaling
        # after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
        # after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        # print('Optimal temperature: %.3f' % self.temperature.item())
        # print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

