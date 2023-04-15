import argparse
import shap
import os
import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import xgboost as xgb
from prediction.outcome_prediction.data_loading.data_loader import load_data
from prediction.utils.shap_helper_functions import check_shap_version_compatibility
from prediction.utils.utils import aggregate_features_over_time

# Shap values require very specific versions
check_shap_version_compatibility()


def compute_shap_explanations_over_time(model_weights_path:str, train_X:np.ndarray, test_X:np.ndarray,
                                        n_time_steps:int, config:dict):
    xgb_model = xgb.XGBClassifier(learning_rate=config['learning_rate'], max_depth=config['max_depth'],
                                  n_estimators=config['n_estimators'], reg_lambda=config['reg_lambda'],
                                  alpha=config['alpha'])

    xgb_model.load_model(model_weights_path)
    # Use the training data for Tree Explainer
    explainer = shap.TreeExplainer(xgb_model, train_X)

    shap_values_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        test_X_with_first_n_ts = test_X[:, ts, :]
        # explaining each prediction requires 2 * background dataset size runs
        shap_values = explainer.shap_values(test_X_with_first_n_ts)
        shap_values_over_ts.append(shap_values)

    return shap_values_over_ts



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SHAP Explanation along time for XGB model')
    parser.add_argument('--outcome', required=True, type=str, help='outcome (ex. 3M mRS 0-2)')
    parser.add_argument('--features_path', required=True, type=str, help='path to features')
    parser.add_argument('--labels_path', required=True, type=str, help='path to labels')
    parser.add_argument('--model_weights_path', required=True, type=str, help='path to model weights')
    parser.add_argument('--parameters_path', required=True, type=str, help='path to model parameters')
    parser.add_argument('--test_size', required=False, type=float, help='test set size [0-1]', default=0.2)
    parser.add_argument('--seed', required=False, type=int, help='Seed', default=42)
    parser.add_argument('--n_time_steps', required=False, type=int, help='Seed', default=72)
    args = parser.parse_args()

    output_dir = os.path.dirname(args.model_weights_path)

    parameters_df = pd.read_csv(args.parameters_path)
    config = parameters_df.squeeze().to_dict()

    if 'moving_average' in parameters_df:
        moving_average = parameters_df['moving_average'][0]
    else:
        moving_average = False

    pids, training_data, test_data, splits, test_features_lookup_table = load_data(args.features_path, args.labels_path, args.outcome,
                                                                                   args.test_size, args.n_splits, args.seed)

    test_X_np, test_y_np = test_data
    X_test, _ = aggregate_features_over_time(test_X_np, test_y_np, moving_average=moving_average)
    X_test = X_test.reshape(-1, args.n_time_steps, X_test.shape[-1]).astype('float32')

    fold_X_train, fold_X_val, fold_y_train, fold_y_val = splits[int(parameters_df['CV'][0])]
    X_train, y_train = aggregate_features_over_time(fold_X_train, fold_y_train, moving_average=moving_average)

    shap_values_over_ts = compute_shap_explanations_over_time(model_weights_path=args.model_weights_path, train_X=X_train, test_X=X_test,
                                                                n_time_steps=args.n_time_steps,
                                                                config=config)

    with open(os.path.join(output_dir, 'tree_explainer_shap_values_over_ts.pkl'), 'wb') as handle:
        pickle.dump(shap_values_over_ts, handle)



