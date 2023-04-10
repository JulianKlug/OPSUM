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


def compute_predictions_over_time(model_weights_path:str, test_X:np.ndarray,
                                        n_time_steps:int, config:dict):
    xgb_model = xgb.XGBClassifier(learning_rate=config['learning_rate'], max_depth=config['max_depth'],
                                  n_estimators=config['n_estimators'], reg_lambda=config['reg_lambda'],
                                  alpha=config['alpha'])

    xgb_model.load_model(model_weights_path)

    predictions_over_ts = []
    for ts in tqdm(range(n_time_steps)):
        test_X_at_ts = test_X[:, ts, :]
        pred = xgb_model.predict_proba(test_X_at_ts)[:, 1].astype('float32')
        predictions_over_ts.append(pred)

    return predictions_over_ts



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

    prediction_over_ts = compute_predictions_over_time(model_weights_path=args.model_weights_path, test_X=X_test,
                                                                n_time_steps=args.n_time_steps,
                                                                config=config)

    with open(os.path.join(output_dir, 'predictions_over_timesteps.pkl'), 'wb') as handle:
        pickle.dump(prediction_over_ts, handle)



