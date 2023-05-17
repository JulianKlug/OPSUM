import pandas as pd
import numpy as np
import pickle
import os
import torch as ch

from netcal.binning import HistogramBinning, IsotonicRegression, ENIR, BBQ
from netcal.scaling import LogisticCalibration, TemperatureScaling, BetaCalibration

from prediction.outcome_prediction.Transformer.calibration.calibration_measures import evaluate_calibration


def evaluate_recalibration_methods(y_val, proba_val, y_test, proba_test, apply_sigmoid=False, use_gpu=False):
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
        calibrated_prediction = instance.transform(proba_test).astype(np.float32)
        # constrained to [0,1]
        calibrated_prediction = np.clip(calibrated_prediction, 0, 1)

        if apply_sigmoid:
            calibrated_prediction = ch.sigmoid(ch.tensor(calibrated_prediction)).numpy()

        # Evaluate the calibration
        results_df = evaluate_calibration(y_test, calibrated_prediction)
        results_df['method'] = name
        overall_results = pd.concat([overall_results, results_df], axis=0)

    return overall_results


if __name__ == '__main__':
    val_predictions_path = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_Death/all_sets_predictions/val_predictions_and_gt.pkl'
    test_predictions_path = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_Death/all_sets_predictions/test_predictions_and_gt.pkl'
    train_predictions_path = '/Users/jk1/temp/opsum_prediction_output/transformer/3M_Death/all_sets_predictions/train_predictions_and_gt.pkl'

    with open(val_predictions_path, 'rb') as f:
        raw_predictions_validation, sigm_predictions_validation, fold_y_val = pickle.load(f)
        raw_predictions_validation = np.array(raw_predictions_validation)
        sigm_predictions_validation = np.array(sigm_predictions_validation)

    with open(test_predictions_path, 'rb') as f:
        raw_predictions_test, sigm_predictions_test, y_test = pickle.load(f)
        raw_predictions_test = np.array(raw_predictions_test)
        sigm_predictions_test = np.array(sigm_predictions_test)

    with open(train_predictions_path, 'rb') as f:
        raw_predictions_train, sigm_predictions_train, fold_y_train = pickle.load(f)
        raw_predictions_train = np.array(raw_predictions_train)
        sigm_predictions_train = np.array(sigm_predictions_train)

    output_dir = '/Users/jk1/Downloads'


    overall_results_after_sigmoid = pd.DataFrame()
    # Recalibration after Sigmoid layer
    # calibration of validation set & recalibration with training set
    initial_calibration_results_df = evaluate_calibration(fold_y_val, sigm_predictions_validation)
    initial_calibration_results_df['method'] = 'initial'
    recalibration_results_df = evaluate_recalibration_methods(fold_y_train, sigm_predictions_train, fold_y_val, sigm_predictions_validation)
    recalibration_results_df = pd.concat([initial_calibration_results_df, recalibration_results_df], axis=0)
    recalibration_results_df['measured_on'] = 'validation'
    recalibration_results_df['recalibration_derived_from'] = 'training'
    overall_results_after_sigmoid = pd.concat([overall_results_after_sigmoid, recalibration_results_df], axis=0)

    # calibration of test set & recalibration with validation set
    initial_calibration_results_df = evaluate_calibration(y_test, sigm_predictions_test)
    initial_calibration_results_df['method'] = 'initial'
    recalibration_results_df = evaluate_recalibration_methods(fold_y_val, sigm_predictions_validation, y_test, sigm_predictions_test)
    recalibration_results_df = pd.concat([initial_calibration_results_df, recalibration_results_df], axis=0)
    recalibration_results_df['measured_on'] = 'test'
    recalibration_results_df['recalibration_derived_from'] = 'validation'
    overall_results_after_sigmoid = pd.concat([overall_results_after_sigmoid, recalibration_results_df], axis=0)

    # calibration of test set & recalibration with training set
    initial_calibration_results_df = evaluate_calibration(y_test, sigm_predictions_test)
    initial_calibration_results_df['method'] = 'initial'
    recalibration_results_df = evaluate_recalibration_methods(fold_y_train, sigm_predictions_train, y_test, sigm_predictions_test)
    recalibration_results_df = pd.concat([initial_calibration_results_df, recalibration_results_df], axis=0)
    recalibration_results_df['measured_on'] = 'test'
    recalibration_results_df['recalibration_derived_from'] = 'training'
    overall_results_after_sigmoid = pd.concat([overall_results_after_sigmoid, recalibration_results_df], axis=0)

    overall_results_after_sigmoid.to_csv(os.path.join(output_dir, 'calibration_results_after_sigmoid_3m_death.csv'), index=False)


