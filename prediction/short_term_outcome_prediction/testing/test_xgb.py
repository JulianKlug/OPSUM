import argparse
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch as ch
import numpy as np
import os
import json
import pickle
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix, average_precision_score, fbeta_score, roc_curve, precision_recall_curve
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir

def test_model(test_data_path:str, train_data_path:str,model_config_path:str, model_path:str=None,
                eval_n_time_steps_before_event = 6, target_interval=True, restrict_to_first_event=False,
                all_folds=False):


    if model_config_path.endswith('.json'):
        model_config = json.load(open(model_config_path))
    else:
        model_config = pd.read_csv(model_config_path)
        model_config = model_config.to_dict(orient='records')[0]

    if all_folds:
        # find all model files in the model path directory
        fold_model_paths = [f for f in os.listdir(model_path) if f.endswith('.model')]
    else:
        # Select model
        fold_model_paths = [model_path]

    X_test, full_y_test = ch.load(test_data_path)
    n_time_steps = X_test.shape[1]
    n_features = X_test.shape[2]
    test_data, test_labels = aggregate_and_label_timeseries(X_test, full_y_test, target_time_to_outcome=eval_n_time_steps_before_event,
                                                            target_interval=target_interval, restrict_to_first_event=restrict_to_first_event)
    test_data = np.concatenate(test_data)
    y_test = np.concatenate(test_labels)

    fold_results = []
    for model_path in fold_model_paths:
        cv_fold = int(model_path.split('_')[-1].split('.')[0])
        X_train, _, y_train, _ = ch.load(train_data_path)[cv_fold]
        train_data, train_labels = aggregate_and_label_timeseries(X_train, y_train, target_time_to_outcome=eval_n_time_steps_before_event,
                                                              target_interval=target_interval, restrict_to_first_event=restrict_to_first_event)
        train_data = np.concatenate(train_data)

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        X_test = scaler.transform(test_data)

        # load model
        xgb_model = xgb.XGBClassifier(
            learning_rate=model_config['learning_rate'],
            max_depth=model_config['max_depth'],
            n_estimators=model_config['n_estimators'],
            reg_lambda=model_config['reg_lambda'],
            reg_alpha=model_config['alpha'],
            scale_pos_weight=model_config['scale_pos_weight'],
            min_child_weight=model_config['min_child_weight'],
            subsample=model_config['subsample'],
            colsample_bytree=model_config['colsample_bytree'],
            colsample_bylevel=model_config['colsample_bylevel'],
            booster=model_config['booster'],
            grow_policy=model_config['grow_policy'],
            gamma=model_config['gamma'],
            num_boost_round=model_config['num_boost_round'],
        )
        xgb_model.load_model(model_path)


        # calculate overall model prediction
        y_pred_test = xgb_model.predict_proba(X_test)[:, 1].astype('float32')

        # Bootstrapped testing
        roc_auc_scores = []
        matthews_scores = []
        accuracy_scores = []
        precision_scores = []
        recall_scores = []
        specificity_scores = []
        neg_pred_value_scores = []

        bootstrapped_ground_truth = []
        bootstrapped_predictions = []

        n_iterations = 1000
        for i in tqdm(range(n_iterations)):
            X_bs, y_bs = resample(X_test, y_test, replace=True)
            # make predictions
            y_pred_bs = xgb_model.predict_proba(X_bs)[:, 1].astype('float32')
            threshold = 0.5
            y_pred_bs_binary = (y_pred_bs > threshold).astype('int32')

            bootstrapped_ground_truth.append(y_bs)
            bootstrapped_predictions.append(y_pred_bs)

            # evaluate model
            roc_auc_bs = roc_auc_score(y_bs, y_pred_bs)
            roc_auc_scores.append(roc_auc_bs)
            matthews_bs = matthews_corrcoef(y_bs, y_pred_bs_binary)
            matthews_scores.append(matthews_bs)
            accuracy_bs = accuracy_score(y_bs, y_pred_bs_binary)
            accuracy_scores.append(accuracy_bs)
            precision_bs = precision_score(y_bs, y_pred_bs_binary)  # == PPV
            recall_bs = recall_score(y_bs, y_pred_bs_binary) # == sensitivity
            precision_scores.append(precision_bs)
            recall_scores.append(recall_bs)

            mcm = multilabel_confusion_matrix(y_bs, y_pred_bs_binary)
            tn = mcm[:, 0, 0]
            tp = mcm[:, 1, 1]
            fn = mcm[:, 1, 0]
            fp = mcm[:, 0, 1]
            specificity_bs = tn / (tn + fp)
            specificity_scores.append(specificity_bs)
            neg_pred_value_bs = tn / (tn + fn)
            neg_pred_value_scores.append(neg_pred_value_bs)


        # get medians
        median_roc_auc = np.percentile(roc_auc_scores, 50)
        median_matthews = np.percentile(matthews_scores, 50)
        median_accuracy = np.percentile(accuracy_scores, 50)
        median_precision = np.percentile(precision_scores, 50)
        median_recall = np.percentile(recall_scores, 50)
        median_specificity = np.percentile(specificity_scores, 50)
        median_neg_pred_value = np.percentile(neg_pred_value_scores, 50)

        # get 95% interval
        alpha = 100 - 95
        lower_ci_roc_auc = np.percentile(roc_auc_scores, alpha / 2)
        upper_ci_roc_auc = np.percentile(roc_auc_scores, 100 - alpha / 2)
        lower_ci_matthews = np.percentile(matthews_scores, alpha / 2)
        upper_ci_matthews = np.percentile(matthews_scores, 100 - alpha / 2)
        lower_ci_accuracy = np.percentile(accuracy_scores, alpha / 2)
        upper_ci_accuracy = np.percentile(accuracy_scores, 100 - alpha / 2)
        lower_ci_precision = np.percentile(precision_scores, alpha / 2)
        upper_ci_precision = np.percentile(precision_scores, 100 - alpha / 2)
        lower_ci_recall = np.percentile(recall_scores, alpha / 2)
        upper_ci_recall = np.percentile(recall_scores, 100 - alpha / 2)
        lower_ci_specificity = np.percentile(specificity_scores, alpha / 2)
        upper_ci_specificity = np.percentile(specificity_scores, 100 - alpha / 2)
        lower_ci_neg_pred_value = np.percentile(neg_pred_value_scores, alpha / 2)
        upper_ci_neg_pred_value = np.percentile(neg_pred_value_scores, 100 - alpha / 2)

        result_df = pd.DataFrame([{
            'auc_test': median_roc_auc,
            'auc_test_lower_ci': lower_ci_roc_auc,
            'auc_test_upper_ci': upper_ci_roc_auc,
            'matthews_test': median_matthews,
            'matthews_test_lower_ci': lower_ci_matthews,
            'matthews_test_upper_ci': upper_ci_matthews,
            'accuracy_test': median_accuracy,
            'accuracy_test_lower_ci': lower_ci_accuracy,
            'accuracy_test_upper_ci': upper_ci_accuracy,
            'precision_test': median_precision,
            'precision_test_lower_ci': lower_ci_precision,
            'precision_test_upper_ci': upper_ci_precision,
            'recall_test': median_recall,
            'recall_test_lower_ci': lower_ci_recall,
            'recall_test_upper_ci': upper_ci_recall,
            'specificity_test': median_specificity,
            'specificity_test_lower_ci': lower_ci_specificity,
            'specificity_test_upper_ci': upper_ci_specificity,
            'neg_pred_value_test': median_neg_pred_value,
            'neg_pred_value_test_lower_ci': lower_ci_neg_pred_value,
            'neg_pred_value_test_upper_ci': upper_ci_neg_pred_value,
            'outcome': 'end',
            'model_weights_path': model_path,
            'cv_fold': cv_fold,
        }], index=[0])

        model_config_df = pd.DataFrame([model_config])
        result_df = pd.concat([result_df, model_config_df], axis=1)
        fold_results.append((result_df, (bootstrapped_ground_truth, bootstrapped_predictions), (y_test, y_pred_test)))

    return fold_results


def _compute_metrics(y_true, y_prob, threshold, auroc=None, auprc=None):
    """Compute all metrics at a given threshold.

    Args:
        y_true: true binary labels
        y_prob: predicted probabilities
        threshold: decision threshold
        auroc: precomputed AUROC (optional, avoids redundant computation)
        auprc: precomputed AUPRC (optional, avoids redundant computation)
    """
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    n = tp + tn + fp + fn

    if auroc is None:
        try:
            auroc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auroc = float('nan')
    if auprc is None:
        try:
            auprc = float(average_precision_score(y_true, y_prob))
        except ValueError:
            auprc = float('nan')

    acc = (tp + tn) / n if n > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    f2 = 5 * prec * rec / (4 * prec + rec) if (4 * prec + rec) > 0 else 0.0
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0
    youden_j = rec + spec - 1

    return {
        'auroc': float(auroc), 'auprc': float(auprc),
        'accuracy': float(acc), 'precision': float(prec), 'recall': float(rec),
        'specificity': float(spec), 'mcc': float(mcc), 'f1': float(f1), 'f2': float(f2),
        'npv': float(npv), 'youden_j': float(youden_j),
    }


def test_final_model(test_data_path, model_dir, threshold_results_path=None,
                     output_dir=None, n_bootstrap=1000):
    """Test the final XGBoost model with bootstrap confidence intervals.

    Args:
        test_data_path: path to test data .pth file (X_test, y_test tuple)
        model_dir: directory containing xgb_final_model.model, scaler.pkl, final_model_config.json
        threshold_results_path: path to threshold tuning results JSON (optional)
        output_dir: directory to save results (default: model_dir/test_results)
        n_bootstrap: number of bootstrap iterations
    """
    # 1. Load model artifacts
    print("Loading model artifacts...")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'xgb_final_model.model'))

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, 'final_model_config.json')) as f:
        config = json.load(f)

    add_lag_features = config.get('add_lag_features', False)
    add_rolling_features = config.get('add_rolling_features', False)
    target_interval = config.get('target_interval', True)
    restrict_to_first_event = config.get('restrict_to_first_event', False)
    n_features_expected = config.get('n_features')

    # 2. Load thresholds
    thresholds = {'default_0.5': 0.5}
    if threshold_results_path is not None:
        with open(threshold_results_path) as f:
            threshold_data = json.load(f)
        median_thresholds = threshold_data.get('median_thresholds', {})
        for metric_name, info in median_thresholds.items():
            thresholds[metric_name] = info['median_threshold']
    print(f"Thresholds: {thresholds}")

    # 3. Prepare test data
    print("Loading and preparing test data...")
    X_test_raw, y_test_raw = ch.load(test_data_path)
    n_patients = X_test_raw.shape[0]
    n_time_steps = X_test_raw.shape[1]

    test_data_list, test_labels_list = aggregate_and_label_timeseries(
        X_test_raw, y_test_raw,
        target_time_to_outcome=6,
        target_interval=target_interval,
        restrict_to_first_event=restrict_to_first_event,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
    )

    # Build timestep index before concatenation
    timestep_index = np.concatenate([np.arange(len(labels)) for labels in test_labels_list])

    test_data = np.concatenate(test_data_list)
    y_test = np.concatenate(test_labels_list)

    X_test_scaled = scaler.transform(test_data)

    n_features = X_test_scaled.shape[1]
    if n_features_expected is not None:
        assert n_features == n_features_expected, \
            f"Feature count mismatch: got {n_features}, expected {n_features_expected}"

    n_samples = len(y_test)
    n_positive = int(y_test.sum())
    print(f"Test data: {n_samples} samples, {n_positive} positive ({n_positive/n_samples*100:.2f}%)")
    print(f"Features: {n_features}")

    # 4. Compute predictions
    print("Computing predictions...")
    y_prob = model.predict_proba(X_test_scaled)[:, 1].astype('float32')

    # 5. Point estimates and curve data
    print("Computing point estimates...")
    point_estimates = {}
    for thresh_name, thresh_val in thresholds.items():
        point_estimates[thresh_name] = _compute_metrics(y_test, y_prob, thresh_val)

    # ROC curve (point estimate)
    fpr_raw, tpr_raw, roc_thresholds_raw = roc_curve(y_test, y_prob)
    # PR curve (point estimate)
    prec_raw, rec_raw, pr_thresholds_raw = precision_recall_curve(y_test, y_prob)

    # Grids for interpolation
    fpr_grid = np.linspace(0, 1, 200)
    recall_grid = np.linspace(0, 1, 200)

    tpr_point_interp = np.interp(fpr_grid, fpr_raw, tpr_raw)
    prec_point_interp = np.interp(recall_grid, rec_raw[::-1], prec_raw[::-1])

    # Per-timepoint point estimates
    unique_timesteps = np.unique(timestep_index).astype(int)
    per_timepoint_point = {}
    for t in unique_timesteps:
        mask = timestep_index == t
        y_t = y_test[mask]
        p_t = y_prob[mask]
        if len(y_t) < 2 or y_t.sum() == 0 or y_t.sum() == len(y_t):
            continue
        auroc_t = float(roc_auc_score(y_t, p_t))
        auprc_t = float(average_precision_score(y_t, p_t))
        per_timepoint_point[int(t)] = {}
        for thresh_name, thresh_val in thresholds.items():
            per_timepoint_point[int(t)][thresh_name] = _compute_metrics(
                y_t, p_t, thresh_val, auroc=auroc_t, auprc=auprc_t)

    # 6. Stratified bootstrap
    print(f"Running {n_bootstrap} bootstrap iterations...")
    bs_results = {thresh_name: [] for thresh_name in thresholds}
    bs_tpr = []
    bs_prec = []
    per_timepoint_bs = defaultdict(list)

    for b in tqdm(range(n_bootstrap)):
        idx = resample(np.arange(n_samples), stratify=y_test, random_state=b)
        y_bs = y_test[idx]
        p_bs = y_prob[idx]

        # Overall metrics at each threshold
        auroc_bs = float(roc_auc_score(y_bs, p_bs))
        auprc_bs = float(average_precision_score(y_bs, p_bs))
        for thresh_name, thresh_val in thresholds.items():
            m = _compute_metrics(y_bs, p_bs, thresh_val, auroc=auroc_bs, auprc=auprc_bs)
            bs_results[thresh_name].append(m)

        # ROC curve
        fpr_bs, tpr_bs, _ = roc_curve(y_bs, p_bs)
        bs_tpr.append(np.interp(fpr_grid, fpr_bs, tpr_bs))

        # PR curve
        prec_bs, rec_bs, _ = precision_recall_curve(y_bs, p_bs)
        bs_prec.append(np.interp(recall_grid, rec_bs[::-1], prec_bs[::-1]))

        # Per-timepoint
        ts_for_idx = timestep_index[idx]
        for t in unique_timesteps:
            ts_mask = ts_for_idx == t
            if ts_mask.sum() < 2:
                continue
            y_t_bs = y_bs[ts_mask]
            p_t_bs = p_bs[ts_mask]
            if y_t_bs.sum() == 0 or y_t_bs.sum() == len(y_t_bs):
                continue
            try:
                auroc_t_bs = float(roc_auc_score(y_t_bs, p_t_bs))
                auprc_t_bs = float(average_precision_score(y_t_bs, p_t_bs))
            except ValueError:
                continue
            for thresh_name, thresh_val in thresholds.items():
                m = _compute_metrics(y_t_bs, p_t_bs, thresh_val, auroc=auroc_t_bs, auprc=auprc_t_bs)
                per_timepoint_bs[(int(t), thresh_name)].append(m)

    # 7. Aggregate results
    print("Aggregating results...")
    metric_names = list(point_estimates[list(thresholds.keys())[0]].keys())

    # Overall bootstrap stats
    bootstrap_stats = {}
    for thresh_name in thresholds:
        bootstrap_stats[thresh_name] = {}
        for metric in metric_names:
            values = np.array([m[metric] for m in bs_results[thresh_name]])
            values = values[~np.isnan(values)]
            bootstrap_stats[thresh_name][metric] = {
                'median': float(np.percentile(values, 50)),
                'std': float(np.std(values)),
                'lower_ci': float(np.percentile(values, 2.5)),
                'upper_ci': float(np.percentile(values, 97.5)),
            }

    # ROC curve bands
    bs_tpr_arr = np.array(bs_tpr)
    roc_curve_data = {
        'fpr_grid': fpr_grid,
        'tpr_point': tpr_point_interp,
        'tpr_mean': np.mean(bs_tpr_arr, axis=0),
        'tpr_lower_ci': np.percentile(bs_tpr_arr, 2.5, axis=0),
        'tpr_upper_ci': np.percentile(bs_tpr_arr, 97.5, axis=0),
        'fpr': fpr_raw,
        'tpr': tpr_raw,
        'thresholds': roc_thresholds_raw,
    }

    # PR curve bands
    bs_prec_arr = np.array(bs_prec)
    pr_curve_data = {
        'recall_grid': recall_grid,
        'precision_point': prec_point_interp,
        'precision_mean': np.mean(bs_prec_arr, axis=0),
        'precision_lower_ci': np.percentile(bs_prec_arr, 2.5, axis=0),
        'precision_upper_ci': np.percentile(bs_prec_arr, 97.5, axis=0),
        'precision': prec_raw,
        'recall': rec_raw,
        'thresholds': pr_thresholds_raw,
    }

    # Per-timepoint aggregation
    per_timepoint_rows = []
    for t in unique_timesteps:
        for thresh_name in thresholds:
            if int(t) not in per_timepoint_point:
                continue
            pt_point = per_timepoint_point[int(t)].get(thresh_name, {})
            bs_list = per_timepoint_bs.get((int(t), thresh_name), [])
            for metric in metric_names:
                row = {
                    'timestep': int(t),
                    'threshold_name': thresh_name,
                    'threshold_value': thresholds[thresh_name],
                    'metric': metric,
                    'point_estimate': pt_point.get(metric, float('nan')),
                }
                if bs_list:
                    values = np.array([m[metric] for m in bs_list])
                    values = values[~np.isnan(values)]
                    if len(values) > 0:
                        row['median'] = float(np.percentile(values, 50))
                        row['std'] = float(np.std(values))
                        row['lower_ci'] = float(np.percentile(values, 2.5))
                        row['upper_ci'] = float(np.percentile(values, 97.5))
                    else:
                        row['median'] = row['std'] = row['lower_ci'] = row['upper_ci'] = float('nan')
                else:
                    row['median'] = row['std'] = row['lower_ci'] = row['upper_ci'] = float('nan')
                per_timepoint_rows.append(row)

    # 8. Save outputs
    if output_dir is None:
        output_dir = os.path.join(model_dir, 'test_results')
    ensure_dir(output_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # test_results.json
    results_json = {
        'metadata': {
            'model_dir': model_dir,
            'test_data_path': test_data_path,
            'threshold_results_path': threshold_results_path,
            'n_test_samples': n_samples,
            'n_positive': n_positive,
            'positive_rate': n_positive / n_samples,
            'n_features': n_features,
            'n_bootstrap': n_bootstrap,
            'n_patients': int(n_patients),
            'n_time_steps': int(n_time_steps),
            'timestamp': timestamp,
        },
        'thresholds': {},
    }
    for thresh_name, thresh_val in thresholds.items():
        results_json['thresholds'][thresh_name] = {
            'threshold_value': thresh_val,
            'point_estimates': point_estimates[thresh_name],
            'bootstrap': bootstrap_stats[thresh_name],
        }

    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump(results_json, f, indent=2)

    # test_results.csv
    csv_rows = []
    for thresh_name, thresh_val in thresholds.items():
        for metric in metric_names:
            csv_rows.append({
                'threshold_name': thresh_name,
                'threshold_value': thresh_val,
                'metric': metric,
                'point_estimate': point_estimates[thresh_name][metric],
                'median': bootstrap_stats[thresh_name][metric]['median'],
                'std': bootstrap_stats[thresh_name][metric]['std'],
                'lower_ci': bootstrap_stats[thresh_name][metric]['lower_ci'],
                'upper_ci': bootstrap_stats[thresh_name][metric]['upper_ci'],
            })
    pd.DataFrame(csv_rows).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    # test_predictions.pkl
    with open(os.path.join(output_dir, 'test_predictions.pkl'), 'wb') as f:
        pickle.dump((y_test, y_prob), f)

    # roc_curve_data.pkl
    with open(os.path.join(output_dir, 'roc_curve_data.pkl'), 'wb') as f:
        pickle.dump(roc_curve_data, f)

    # pr_curve_data.pkl
    with open(os.path.join(output_dir, 'pr_curve_data.pkl'), 'wb') as f:
        pickle.dump(pr_curve_data, f)

    # per_timepoint_results.csv
    if per_timepoint_rows:
        pd.DataFrame(per_timepoint_rows).to_csv(
            os.path.join(output_dir, 'per_timepoint_results.csv'), index=False)

    # 10. Print summary
    print(f"\n{'='*70}")
    print(f"TEST RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Samples: {n_samples} ({n_positive} positive, {n_positive/n_samples*100:.2f}%)")
    print(f"Features: {n_features}")
    print(f"Bootstrap iterations: {n_bootstrap}")
    print()

    for thresh_name, thresh_val in thresholds.items():
        print(f"--- Threshold: {thresh_name} = {thresh_val:.4f} ---")
        pe = point_estimates[thresh_name]
        bs = bootstrap_stats[thresh_name]
        for metric in metric_names:
            print(f"  {metric:>15s}: {pe[metric]:.4f}  "
                  f"[{bs[metric]['lower_ci']:.4f}, {bs[metric]['upper_ci']:.4f}]")
        print()

    print(f"Results saved to {output_dir}")
    return results_json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true',
                        help='Test final model (retrained on all data)')
    parser.add_argument('-d', '--test_data_path', type=str, help='Path to test data')
    parser.add_argument('-t', '--train_data_path', type=str,
                        help='Path to train data (or threshold results file in --final mode)')
    parser.add_argument('-c', '--model_config_path', type=str,
                        help='Path to model config file (json or csv)')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to model file or model directory (--final mode)')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('-a', '--all_folds', type=bool, default=False,
                        help='whether to test on all folds or just the best one')
    parser.add_argument('-n', '--eval_n_time_steps_before_event', type=int, default=None,
                        help='eval_n_time_steps (default: 6) or n_bootstrap in --final mode (default: 1000)')
    cli_args = parser.parse_args()

    if cli_args.final:
        test_final_model(
            test_data_path=cli_args.test_data_path,
            model_dir=cli_args.model_path,
            threshold_results_path=cli_args.train_data_path,
            output_dir=cli_args.output_dir,
            n_bootstrap=cli_args.eval_n_time_steps_before_event if cli_args.eval_n_time_steps_before_event is not None else 1000,
        )
    else:
        eval_n = cli_args.eval_n_time_steps_before_event if cli_args.eval_n_time_steps_before_event is not None else 6
        fold_results = test_model(
            test_data_path=cli_args.test_data_path,
            train_data_path=cli_args.train_data_path,
            model_config_path=cli_args.model_config_path,
            model_path=cli_args.model_path,
            all_folds=cli_args.all_folds,
            eval_n_time_steps_before_event=eval_n,
        )
        output_path = cli_args.output_dir
        if output_path is None:
            output_path = os.path.join(os.path.dirname(cli_args.model_config_path),
                                       f'test_results_{eval_n}h')
        ensure_dir(output_path)

        # save results
        for fidx in range(len(fold_results)):
            result_df = fold_results[fidx][0]
            bootstrapping_data = fold_results[fidx][1]
            testing_data = fold_results[fidx][2]
            cv_fold = result_df['cv_fold'].values[0]
            result_df.to_csv(os.path.join(output_path, f'test_XGB_cv_{cv_fold}_results.csv'), index=False)

            # save bootstrapped ground truth and predictions
            pickle.dump(bootstrapping_data, open(os.path.join(output_path, f'bootstrapped_gt_and_pred_cv_{cv_fold}.pkl'), 'wb'))
            pickle.dump(testing_data, open(os.path.join(output_path, f'test_gt_and_pred_cv_{cv_fold}.pkl'), 'wb'))