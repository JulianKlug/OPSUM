import argparse
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
import torch as ch
import numpy as np
import os
import json
import pickle
from tqdm import tqdm
import xgboost as xgb
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, \
    multilabel_confusion_matrix
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--test_data_path', type=str, help='Path to test data')
    parser.add_argument('-t', '--train_data_path', type=str, help='Path to train data')
    parser.add_argument('-c', '--model_config_path', type=str, help='Path to model config file (json or csv)')
    parser.add_argument('-m', '--model_path', type=str, help='Path to model file (if not using all folds)')
    parser.add_argument('-o', '--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('-a', '--all_folds', type=bool, default=False, help='whether to test on all folds or just the best one')
    parser.add_argument('-n', '--eval_n_time_steps_before_event', type=int, default=6,
                        help='Number of time steps before the event to evaluate the model on')
    cli_args = parser.parse_args()

    fold_results = test_model(
        test_data_path=cli_args.test_data_path,
        train_data_path=cli_args.train_data_path,
        model_config_path=cli_args.model_config_path,
        model_path=cli_args.model_path,
        all_folds=cli_args.all_folds,
        eval_n_time_steps_before_event=cli_args.eval_n_time_steps_before_event
    )
    output_path = cli_args.output_dir
    if output_path is None:
        output_path = os.path.join(os.path.dirname(cli_args.model_config_path),
                                   f'test_results_{cli_args.eval_n_time_steps_before_event}h')
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