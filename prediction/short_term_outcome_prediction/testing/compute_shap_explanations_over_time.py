import argparse
import shap
import os
import json
import numpy as np
import pickle
from scipy.special import expit  # sigmoid
from sklearn.discriminant_analysis import StandardScaler
from tqdm import tqdm
import pandas as pd
import xgboost as xgb
import torch as ch
from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.shap_helper_functions import check_shap_version_compatibility
from prediction.utils.utils import ensure_dir

# Shap values require very specific versions
# check_shap_version_compatibility()


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
    # Use the training data for Tree Explainer
    # explainer = shap.TreeExplainer(xgb_model, train_X)
    booster = xgb_model.get_booster()

    shap_values_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        test_X_with_first_n_ts = test_X[:, ts, :]
        dtest = xgb.DMatrix(test_X_with_first_n_ts)
        # shap values in margin space (values + base value = margin prediction) 
        shap_values = booster.predict(dtest, pred_contribs=True)
        shap_values_over_ts.append(shap_values)

    return shap_values_over_ts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHAP Explanation along time for XGB model')
    parser.add_argument('-d', '--test_data_path', required=True, type=str, help='path to test data')
    parser.add_argument('-t', '--train_data_path', type=str, help='Path to train data')
    parser.add_argument('-c', '--model_config_path', type=str, help='Path to model config file (json or csv)')
    parser.add_argument('-m', '--model_path', type=str, help='Path to model file (if not using all folds)')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('-n', '--eval_n_time_steps_before_event', type=int, default=6,
                        help='Number of time steps before the event to evaluate the model on')
    parser.add_argument('-i', '--target_interval', type=int, default=1)
    parser.add_argument('-r', '--restrict_to_first_event', action='store_true',
                        help='Restrict to the first event only')
    args = parser.parse_args()

    output_path = args.output_dir
    if output_path is None:
        output_path = os.path.join(os.path.dirname(args.model_config_path),
                                   f'shap_explanations_over_time')
    ensure_dir(output_path)

    cv_fold = int(args.model_path.split('_')[-1].split('.')[0])
    X_train, _, y_train, _ = ch.load(args.train_data_path)[cv_fold]
    train_data, train_labels = aggregate_and_label_timeseries(X_train, y_train, target_time_to_outcome=args.eval_n_time_steps_before_event,
                                                              target_interval=args.target_interval, restrict_to_first_event=args.restrict_to_first_event)
    train_data = np.concatenate(train_data)

    X_test, full_y_test = ch.load(args.test_data_path)
    n_time_steps = X_test.shape[1]
    n_features = X_test.shape[2]
    test_data, test_labels = aggregate_and_label_timeseries(X_test, full_y_test, target_time_to_outcome=args.eval_n_time_steps_before_event,
                                                            target_interval=args.target_interval, restrict_to_first_event=args.restrict_to_first_event)
    test_data = np.concatenate(test_data)

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    X_test = scaler.transform(test_data)
    # Reshape the data to have the shape (n_subj, n_time_steps, n_features)
    X_test = X_test.reshape(-1, n_time_steps, n_features*4)
    
    shap_values_over_ts = compute_shap_explanations_over_time(model_config_path=args.model_config_path,
                                                                model_weights_path=args.model_path,
                                                                train_X=train_data,
                                                                test_X=X_test,
                                                                n_time_steps=n_time_steps)

    with open(os.path.join(output_path, 'tree_explainer_shap_values_over_ts.pkl'), 'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)



