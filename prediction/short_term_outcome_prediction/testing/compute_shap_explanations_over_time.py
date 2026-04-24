import argparse
import os
import json
import numpy as np
import pickle
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
import pandas as pd
import xgboost as xgb
import torch as ch
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir


def _compute_shap_over_time(booster, test_X, n_time_steps):
    """Compute per-timestep SHAP values using pred_contribs.

    Args:
        booster: xgboost Booster object
        test_X: array of shape (n_subjects, n_time_steps, n_features_per_ts)
        n_time_steps: number of timesteps

    Returns:
        list of arrays, one per timestep, each (n_subjects, n_features+1)
    """
    shap_values_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        test_X_with_first_n_ts = test_X[:, ts, :]
        dtest = xgb.DMatrix(test_X_with_first_n_ts)
        # shap values in margin space (values + base value = margin prediction)
        shap_values = booster.predict(dtest, pred_contribs=True)
        shap_values_over_ts.append(shap_values)
    return shap_values_over_ts


def compute_shap_explanations_over_time(model_config_path:str, model_weights_path:str, train_X:np.ndarray, test_X:np.ndarray,
                                        n_time_steps:int):

    if model_config_path.endswith('.json'):
        model_config = json.load(open(model_config_path))
    else:
        model_config = pd.read_csv(model_config_path)
        model_config = model_config.to_dict(orient='records')[0]

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

    xgb_model.load_model(model_weights_path)
    booster = xgb_model.get_booster()

    return _compute_shap_over_time(booster, test_X, n_time_steps)


def _build_aggregated_feature_names(raw_feature_names, add_lag_features=False,
                                    add_rolling_features=False):
    """Build the list of aggregated feature names matching aggregate_features_over_time output order.

    Order: features, avg_, min_, max_, std_, diff_, timestep_idx,
           [lag2_, lag3_], [rolling_mean_, rolling_std_, rolling_trend_]
    """
    names = list(raw_feature_names)
    for prefix in ['avg_', 'min_', 'max_', 'std_', 'diff_']:
        names += [f'{prefix}{f}' for f in raw_feature_names]
    names += ['timestep_idx']

    if add_lag_features:
        for prefix in ['lag2_', 'lag3_']:
            names += [f'{prefix}{f}' for f in raw_feature_names]

    if add_rolling_features:
        for prefix in ['rolling_mean_', 'rolling_std_', 'rolling_trend_']:
            names += [f'{prefix}{f}' for f in raw_feature_names]

    return names


def compute_shap_final_model(test_data_path, model_dir, output_dir):
    """Compute SHAP explanations over time for the final XGBoost model.

    Args:
        test_data_path: path to test data .pth file (X_test, y_test tuple)
        model_dir: directory containing xgb_final_model.model, scaler.pkl, final_model_config.json
        output_dir: directory to save SHAP results
    """
    # 1. Load model artifacts
    print("Loading model artifacts...")
    model = xgb.XGBClassifier()
    model.load_model(os.path.join(model_dir, 'xgb_final_model.model'))
    booster = model.get_booster()

    with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)

    with open(os.path.join(model_dir, 'final_model_config.json')) as f:
        config = json.load(f)

    add_lag_features = config.get('add_lag_features', False)
    add_rolling_features = config.get('add_rolling_features', False)
    target_interval = config.get('target_interval', True)
    restrict_to_first_event = config.get('restrict_to_first_event', False)

    # 2. Prepare test data
    print("Loading and preparing test data...")
    X_test_raw, y_test_raw = ch.load(test_data_path)
    n_time_steps = X_test_raw.shape[1]

    test_data_list, test_labels_list = aggregate_and_label_timeseries(
        X_test_raw, y_test_raw,
        target_time_to_outcome=6,
        target_interval=target_interval,
        restrict_to_first_event=restrict_to_first_event,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
    )

    test_data = np.concatenate(test_data_list)

    # 3. Scale with pre-fitted scaler
    X_test_scaled = scaler.transform(test_data)

    # 4. Reshape to (n_subjects, n_time_steps, n_features_per_ts)
    # Each row in X_test_scaled is one timestep with all aggregated features,
    # so shape[1] is already the number of features per timestep.
    n_features_per_ts = X_test_scaled.shape[1]
    X_test_reshaped = X_test_scaled.reshape(-1, n_time_steps, n_features_per_ts)
    print(f"Test data: {X_test_reshaped.shape[0]} subjects, {n_time_steps} timesteps, "
          f"{n_features_per_ts} features per timestep")

    # 5. Compute SHAP values over time
    print("Computing SHAP explanations over time...")
    shap_values_over_ts = _compute_shap_over_time(booster, X_test_reshaped, n_time_steps)

    # 6. Build feature names
    raw_feature_names = list(X_test_raw[0, 0, :, 2])
    feature_names = _build_aggregated_feature_names(
        raw_feature_names,
        add_lag_features=add_lag_features,
        add_rolling_features=add_rolling_features,
    )

    # 7. Save outputs
    ensure_dir(output_dir)

    with open(os.path.join(output_dir, 'tree_explainer_shap_values_over_ts.pkl'), 'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)

    with open(os.path.join(output_dir, 'shap_feature_names.pkl'), 'wb') as handle:
        pickle.dump(feature_names, handle)

    print(f"Saved SHAP values and feature names to {output_dir}")
    print(f"  - tree_explainer_shap_values_over_ts.pkl ({len(shap_values_over_ts)} timesteps)")
    print(f"  - shap_feature_names.pkl ({len(feature_names)} features)")

    return shap_values_over_ts, feature_names



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHAP Explanation along time for XGB model')
    parser.add_argument('--final', action='store_true',
                        help='Compute SHAP for the final model (retrained on all data)')
    parser.add_argument('-d', '--test_data_path', required=True, type=str, help='path to test data')
    parser.add_argument('-t', '--train_data_path', type=str, help='Path to train data')
    parser.add_argument('-c', '--model_config_path', type=str, help='Path to model config file (json or csv)')
    parser.add_argument('-m', '--model_path', type=str,
                        help='Path to model file, or model directory in --final mode')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('-n', '--eval_n_time_steps_before_event', type=int, default=6,
                        help='Number of time steps before the event to evaluate the model on')
    parser.add_argument('-i', '--target_interval', type=int, default=1)
    parser.add_argument('-r', '--restrict_to_first_event', action='store_true',
                        help='Restrict to the first event only')
    args = parser.parse_args()

    if args.final:
        # --final mode: -m is the model directory, -d is test data
        output_path = args.output_dir
        if output_path is None:
            output_path = os.path.join(args.model_path, 'shap_explanations_over_time')
        compute_shap_final_model(
            test_data_path=args.test_data_path,
            model_dir=args.model_path,
            output_dir=output_path,
        )
    else:
        # CV fold mode (original behavior)
        output_path = args.output_dir
        if output_path is None:
            output_path = os.path.join(os.path.dirname(args.model_config_path),
                                       f'shap_explanations_over_time')
        ensure_dir(output_path)

        cv_fold = int(args.model_path.split('_')[-1].split('.')[0])
        X_train, _, y_train, _ = ch.load(args.train_data_path)[cv_fold]
        y_train = pd.DataFrame(y_train)
        train_data, train_labels = aggregate_and_label_timeseries(X_train, y_train, target_time_to_outcome=args.eval_n_time_steps_before_event,
                                                                  target_interval=args.target_interval, restrict_to_first_event=args.restrict_to_first_event)
        train_data = np.concatenate(train_data)

        X_test, full_y_test = ch.load(args.test_data_path)
        full_y_test = pd.DataFrame(full_y_test)
        n_time_steps = X_test.shape[1]
        n_features = X_test.shape[2]
        test_data, test_labels = aggregate_and_label_timeseries(X_test, full_y_test, target_time_to_outcome=args.eval_n_time_steps_before_event,
                                                                target_interval=args.target_interval, restrict_to_first_event=args.restrict_to_first_event)
        test_data = np.concatenate(test_data)

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        X_test = scaler.transform(test_data)
        # Reshape the data to have the shape (n_subj, n_time_steps, n_features)
        n_aggregated_features = X_test.shape[1] // n_time_steps
        X_test = X_test.reshape(-1, n_time_steps, n_aggregated_features)

        shap_values_over_ts = compute_shap_explanations_over_time(model_config_path=args.model_config_path,
                                                                    model_weights_path=args.model_path,
                                                                    train_X=train_data,
                                                                    test_X=X_test,
                                                                    n_time_steps=n_time_steps)

        with open(os.path.join(output_path, 'tree_explainer_shap_values_over_ts.pkl'), 'wb') as handle:
            pickle.dump(shap_values_over_ts, handle)
