import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from netcal.metrics import ECE
from netcal.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration
from prediction.outcome_prediction.Transformer.calibration.calibration_measures import cox_calibration_coefficients


def evaluate_calibration(y:np.ndarray, probs:np.ndarray) -> pd.DataFrame:
    results = {}
    cox_coeffs = cox_calibration_coefficients(y, probs)
    results['slope'] = cox_coeffs['slope']
    results['intercept'] = cox_coeffs['intercept']
    expected_calibration_error = ECE(bins=10)
    results['ECE'] = expected_calibration_error.measure(probs, y)
    results['Brier'] = brier_score_loss(y, probs)

    return pd.DataFrame(results, index=[0])


def evaluate_recalibration_methods(y_val, proba_val, y_test, proba_test, use_gpu=False):

    # Method that is used to obtain a calibration mapping: - ‘mle’: Maximum likelihood estimate without uncertainty using a convex optimizer.
    method = 'mle'
    hist_bins = 20

    # kwargs for uncertainty mode. Those can also be safely set on MLE
    uncertainty_kwargs = {'mcmc_chains': 1,
                          'mcmc_samples': 100,
                          'mcmc_warmup_steps': 10,
                          'vi_samples': 300,
                          'vi_epochs': 100}

    histogram = HistogramBinning(hist_bins)
    iso = IsotonicRegression()
    bbq = BBQ()
    enir = ENIR()

    lr_calibration = LogisticCalibration(detection=False, method=method, use_cuda=use_gpu, **uncertainty_kwargs)
    temperature = TemperatureScaling(detection=False, method=method, use_cuda=use_gpu, **uncertainty_kwargs)
    betacal = BetaCalibration(detection=False, method=method, use_cuda=use_gpu, **uncertainty_kwargs)

    models = [("hist", histogram),
              ("iso", iso),
              ("bbq", bbq),
              ("enir", enir),
              ("lr", lr_calibration),
              ("temperature", temperature),
              ("beta", betacal)]

    overall_results = pd.DataFrame()
    for model in models:
        name, instance = model
        # Fit the calibration model
        instance.fit(proba_val, y_val)
        # Calibrate the predictions
        calibrated_prediction = instance.transform(proba_test)

        # Evaluate the calibration
        results_df = evaluate_calibration(y_test, calibrated_prediction)
        results_df['method'] = name
        overall_results = pd.concat([overall_results, results_df], axis=0)

    return overall_results

