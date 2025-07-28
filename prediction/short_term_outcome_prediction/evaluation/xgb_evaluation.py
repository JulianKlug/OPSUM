import json
import os

import pandas as pd
import torch as ch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

from prediction.short_term_outcome_prediction.timeseries_decomposition import aggregate_and_label_timeseries
from prediction.utils.utils import ensure_dir

def xgb_validation_evaluation(data_path:str, model_config_path:str, model_path:str=None,
                          output_path=None, predictions_dir:str=None,
                          use_gpu = False,  n_time_steps = 72, eval_n_time_steps_before_event = 6,
                          target_interval=True, restrict_to_first_event=False,
                          use_cross_validation = True):
    """
    Evaluate the model on the validation set
    Gist: Use a separate evaluation then the one formalised in the model training
    """
    if model_config_path.endswith('.json'):
        model_config = json.load(open(model_config_path))
    else:
        model_config = pd.read_csv(model_config_path)
        model_config = model_config.to_dict(orient='records')[0]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_config_path),
                                   f'validation_evaluation_results_{eval_n_time_steps_before_event}h')
    ensure_dir(output_path)

    # Load the data
    splits = ch.load(os.path.join(data_path))

    if use_cross_validation:
        loop_range = range(len(splits))
    else:
        loop_range = [model_config['CV']]

    all_folds_results = pd.DataFrame()
    for cv_fold in loop_range:
        fold_result_dir = os.path.join(output_path, f'cv_fold_{cv_fold}')
        ensure_dir(fold_result_dir)


        full_X_train, full_X_val, y_train, y_val = splits[cv_fold]
        n_time_steps = full_X_train.shape[1]
        n_features = full_X_train.shape[2]

        train_data, train_labels = aggregate_and_label_timeseries(full_X_train, y_train, target_time_to_outcome=6,
                                                              target_interval=target_interval, restrict_to_first_event=restrict_to_first_event)
        val_data, val_labels = aggregate_and_label_timeseries(full_X_val, y_val, target_time_to_outcome=6,
                                                              target_interval=target_interval, restrict_to_first_event=restrict_to_first_event)

        
        train_data = np.concatenate(train_data)
        # train_labels = np.concatenate(train_labels)

        val_data = np.concatenate(val_data)
        # val_labels = np.concatenate(val_labels)

        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        X_val = scaler.transform(val_data)

        # reshape X val into (n_subj, n_time_steps, n_features)
        X_val = X_val.reshape((-1, n_time_steps, n_features*4))
        # val_labels = val_labels.reshape((-1, n_time_steps))

        # if predictions are precomputed, load them
        if predictions_dir is not None:
            # look for file ending in cv{cv_fold}.pt
            fold_predictions_path = [f for f in os.listdir(predictions_dir) if f.endswith(f'cv{cv_fold}.pt')][0]
            predictions_data = ch.load(os.path.join(predictions_dir, fold_predictions_path))
            pred_over_ts_np = np.squeeze(predictions_data).T

        # else load model and compute predictions
        else:
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

            # compute predictions
            pred_over_ts = []
            for ts in tqdm(range(n_time_steps)):
                X_val_with_first_n_ts = X_val[:, ts, :]
                y_pred = xgb_model.predict_proba(X_val_with_first_n_ts)[:, 1].astype('float32') 
                pred_over_ts.append(np.squeeze(y_pred))

            ch.save(pred_over_ts, os.path.join(fold_result_dir, f'predictions_cv{cv_fold}.pt'))

            pred_over_ts_np = np.squeeze(pred_over_ts).T


        # construct y
        y_val_list = []
        for cid in full_X_val[:, 0, 0, 0]:
            if cid not in y_val.case_admission_id.values:
                cid_y = np.zeros(n_time_steps)
            else:
                if restrict_to_first_event:
                    cid_event_ts = np.min(y_val[y_val.case_admission_id == cid].relative_sample_date_hourly_cat.values)
                else:
                    cid_event_ts = y_val[y_val.case_admission_id == cid].relative_sample_date_hourly_cat.values

                if (not target_interval) and (cid_event_ts < (eval_n_time_steps_before_event + 1)):
                        # if the event occurs before a detection window, ignore the patient
                        cid_y = np.array([])
                        continue
                
                if restrict_to_first_event:
                    # if we restrict to the first event, we need to get the max ts for this patient by looking at the first event
                    max_ts = np.min(cid_event_ts)
                else:
                    max_ts = n_time_steps
                    
                cid_y = []
                for ts in range(int(max_ts)):
                    if target_interval:
                        # if any of target_events_ts is between ts and ts + eval_n_time_steps_before_event:
                        if np.any((cid_event_ts > ts) & (cid_event_ts <= ts + eval_n_time_steps_before_event)):
                            cid_y.append(1)
                        else:
                            cid_y.append(0)
                    else:
                        # not targerting interval -> event occurs exactly at ts + eval_n_time_steps_before_event
                        # if any of target_events_ts is equal to ts + eval_n_time_steps_before_event:
                        if np.any(cid_event_ts == ts + eval_n_time_steps_before_event):
                            cid_y.append(1)
                        else:
                            cid_y.append(0)
                cid_y = np.array(cid_y)

            y_val_list.append(cid_y)


        # evaluate predictions
        # compute roc scores for each time step
        roc_scores = []
        auprc_scores = []
        mcc_scores = []
        accuracy_scores = []
        # count number of positive samples for each time step
        n_pos_samples = []
        timesteps = []

        overall_prediction_df = pd.DataFrame(columns=['timestep', 'prediction', 'true_label'])

        for ts in range(n_time_steps):
            pts_idx = [i for i, y in enumerate(y_val_list) if len(y) > ts]
            y_true = np.array([y[ts] for y in y_val_list if len(y) > ts])
            y_pred = pred_over_ts_np[pts_idx, ts]
            y_pred_binary = np.where(y_pred > 0.5, 1, 0)

            timestep_df = pd.DataFrame({'timestep': [ts] * len(y_true),
                                        'prediction': y_pred,
                                        'true_label': y_true})
            overall_prediction_df = pd.concat([overall_prediction_df, timestep_df])

            timesteps.append(ts)
            n_pos_samples.append(np.sum(y_true))
            accuracy_scores.append(accuracy_score(y_true, y_pred_binary))

            if len(np.unique(y_true)) == 1:
                roc_scores.append(np.nan)
                auprc_scores.append(np.nan)
                mcc_scores.append(np.nan)
            else:
                roc_scores.append(roc_auc_score(y_true, y_pred))
                auprc_scores.append(average_precision_score(y_true, y_pred))
                mcc_scores.append(matthews_corrcoef(y_true, y_pred_binary))

    # Ensure true_label is binary
        overall_prediction_df['true_label'] = overall_prediction_df['true_label'].astype(int)
    # Ensure prediction is a continuous value between 0 and 1
        overall_prediction_df['prediction'] = overall_prediction_df['prediction'].astype(float)

        def cutoff_youdens_j(fpr,tpr,thresholds):
            j_scores = tpr-fpr
            j_ordered = sorted(zip(j_scores,thresholds))
            return j_ordered[-1][1]
        
        fpr, tpr, thresholds = roc_curve(overall_prediction_df.true_label, overall_prediction_df.prediction)

        # compute overall metrics
        overall_results_df = pd.DataFrame({'overall_roc': roc_auc_score(overall_prediction_df.true_label,
                                                                         overall_prediction_df.prediction),
                                            # get youden j index
                                            'overall_youden_index': cutoff_youdens_j(fpr, tpr, thresholds),
                                        'overall_auprc': average_precision_score(overall_prediction_df.true_label,
                                                                               overall_prediction_df.prediction),
                                        'overall_mcc': matthews_corrcoef(overall_prediction_df.true_label,
                                                                         overall_prediction_df.prediction > 0.5),
                                        'overall_accuracy': accuracy_score(overall_prediction_df.true_label,
                                                                          overall_prediction_df.prediction > 0.5),
                                        'n_pos_samples': np.sum(overall_prediction_df.true_label),
                                       'n_samples': len(overall_prediction_df),
                                        'cv_fold': cv_fold
                                    }, index=[0])
        all_folds_results = pd.concat([all_folds_results, overall_results_df])
        overall_results_df.to_csv(os.path.join(fold_result_dir, 'overall_validation_results.csv'))


        # median over time for each metric
        median_results_df = pd.DataFrame({'median_roc': np.nanmedian(roc_scores),
                                        'median_auprc': np.nanmedian(auprc_scores),
                                        'median_mcc': np.nanmedian(mcc_scores),
                                       'median_accuracy': np.nanmedian(accuracy_scores),
                                       'n_pos_samples': np.nanmedian(n_pos_samples),
                                   }, index=[0])


        # save results
        median_results_df.to_csv(os.path.join(fold_result_dir, 'median_validation_results.csv'))
        # save individual scores per time step
        pd.DataFrame({'timestep': timesteps, 'roc': roc_scores, 'auprc': auprc_scores, 'mcc': mcc_scores,
                      'accuracy': accuracy_scores, 'n_pos_samples': n_pos_samples
                      }).to_csv(
            os.path.join(fold_result_dir, 'validation_scores_per_timestep.csv'))

        # plot results
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.scatterplot(x=range(1, n_time_steps + 1), y=roc_scores, label='AUROC', ax=ax)
        sns.scatterplot(x=range(1, n_time_steps + 1), y=auprc_scores, label='AUPRC', ax=ax)
        sns.scatterplot(x=range(1, n_time_steps + 1), y=mcc_scores, label='MCC', ax=ax)

        ax2 = ax.twinx()
        sns.scatterplot(x=range(1, n_time_steps + 1), y=n_pos_samples, color='red', alpha=0.3, ax=ax2,
                        label='Positive Samples', zorder=0)

        ax.set_xlabel('Time step')
        ax.set_ylabel('Score')
        ax.set_title('Validation scores over time')

        # save plot
        plt.savefig(os.path.join(fold_result_dir, 'validation_scores_over_time.png'))

        if cv_fold == model_config['CV']:
            # plot predictions over time for a subset of patients
            with PdfPages(os.path.join(fold_result_dir, 'predictions_over_time.pdf')) as pdf:
                for i in range(pred_over_ts_np.shape[0]):
                    cid = full_X_val[i, 0, 0, 0]
                    plt.figure()  # Create a new figure
                    ax = sns.scatterplot(
                        x=range(1, pred_over_ts_np.shape[1] + 1),
                        y=pred_over_ts_np[i, :],
                        hue=pred_over_ts_np[i, :] > 0.5
                    )
                    ax.set_ylim(0, 1)
                    ax.set_title(f'Prediction over time for patient {cid}')

                    if cid in y_val.case_admission_id.values:
                        for end_ts in y_val[y_val.case_admission_id == cid].relative_sample_date_hourly_cat.values:
                            # plot vertical line for each event
                            ax.axvline(x=end_ts, color='red')


                    pdf.savefig()  # Save the current figure into the PDF
                    plt.close()  # Close the figure to free memory

    # save overall results
    all_folds_results.to_csv(os.path.join(output_path, 'all_folds_overall_validation_results.csv'))

    return overall_results_df




if __name__ == '__main__':
    # Example usage
    # python val_evaluation.py --data_path /path/to/data --model_path /path/to/model --model_config_path /path/to/model_config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=False)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-p', '--predictions_dir', type=str, default=None)
    parser.add_argument('-ti', '--target_interval', default=True)
    parser.add_argument('-fi', '--restrict_to_first_event', default=False, action='store_true')
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_time_steps', type=int, default=72)
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6)
    parser.add_argument('-cv', '--use_cross_validation', action='store_true')
    args = parser.parse_args()

    if args.use_cross_validation:
        use_cross_validation = True
    else:
        use_cross_validation = False

    xgb_validation_evaluation(data_path=args.data_path,
                          model_path=args.model_path,
                          model_config_path=args.model_config_path,
                          output_path=args.output_path,
                            predictions_dir=args.predictions_dir,
                            use_gpu=args.use_gpu,
                            n_time_steps=args.n_time_steps,
                            eval_n_time_steps_before_event=args.eval_n_time_steps_before_event,
                            use_cross_validation=use_cross_validation,
                            target_interval=args.target_interval,
                            restrict_to_first_event=args.restrict_to_first_event)


