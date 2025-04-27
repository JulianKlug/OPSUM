import os
import json
import pandas as pd
import torch as ch
import numpy as np
import pandas as pd
import torch as ch
import numpy as np
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from prediction.utils.utils import ensure_dir
from prediction.short_term_outcome_prediction.inference.encoder_decoder_inference import encoder_decoder_predict


def reverse_normalisation(data, variable_name, normalisation_parameters_df):
    """
    Reverse normalisation of the data.
    :param data: The data to reverse normalise.
    :param variable_name: The name of the variable to reverse normalise.
    :return: The reverse normalised data.
    """
    # Get the original mean and std from the normalisation parameters
    # Reverse normalisation
    std = normalisation_parameters_df[normalisation_parameters_df.variable == variable_name].original_std.iloc[0]
    mean = normalisation_parameters_df[normalisation_parameters_df.variable == variable_name].original_mean.iloc[0]
    data = (data * std) + mean
    return data


def encoder_decoder_validation_evaluation(data_path:str, model_config_path:str, model_path:str=None, normalisation_data_path:str=None,
                                            outcome_data_path:str=None, predictions_path:str=None, cv_fold:int=None,
                                        output_path=None, use_gpu = False,  n_time_steps = 72, eval_n_time_steps_before_event = 6):
    
    if model_config_path.endswith('.json'):
        model_config = json.load(open(model_config_path))
    else:
        model_config = pd.read_csv(model_config_path)
        model_config = model_config.to_dict(orient='records')[0]

    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_config_path),
                                   f'decoder_validation_evaluation_results_{eval_n_time_steps_before_event}h')
    ensure_dir(output_path)

    if cv_fold is None:
        cv_fold = model_config['best_cv_fold']

    splits = ch.load(os.path.join(data_path))
    full_X_train, full_X_val, y_train, y_val = splits[cv_fold]
    val_patient_cids = full_X_val[:, 0, 0, 0]

    # prepare scaler (needed for reverse scaling)
    scaler = StandardScaler()
    X_train = full_X_train[:, :, :, -1].astype('float32')
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)

    # find index of max_NIHSS
    max_NIHSS_idx = np.where(full_X_train[0, 0, :, -2] == 'max_NIHSS')[0][0]
    # find index of min_NIHSS
    min_NIHSS_idx = np.where(full_X_train[0, 0, :, -2] == 'min_NIHSS')[0][0]

    normalisation_parameters_df = pd.read_csv(normalisation_data_path)
    outcome_df = pd.read_csv(outcome_data_path)

    if predictions_path is not None:
        pred_over_ts_np = ch.load(predictions_path)
    else:
        pred_over_ts_np = encoder_decoder_predict(
            data_path=data_path,
            model_path=model_path,
            model_config_path=model_config_path,
            predict_n_time_steps=eval_n_time_steps_before_event,
            n_time_steps=n_time_steps,
            use_gpu=use_gpu
        )

    roc_scores = []
    auprc_scores = []
    mcc_scores = []
    accuracy_scores = []
    # count number of positive samples for each time step
    n_pos_samples = []
    timesteps = []

    overall_prediction_df = pd.DataFrame(columns=['timestep', 'prediction', 'true_label'])

    for ts in range(n_time_steps):
        evaluated_ts = ts + eval_n_time_steps_before_event

        # GT at evaluated time step
        outcome_at_evaluated_ts_df = outcome_df[outcome_df['relative_sample_date_hourly_cat'] == evaluated_ts]
        # gt at ts is 0/1 if the patient is in the outcome group at the evaluated time step
        y_true_at_evaluated_ts = np.isin(val_patient_cids, outcome_at_evaluated_ts_df['case_admission_id'].values).astype(np.int32)

        # prediction at evaluated time step
        predictions_at_ts_np = pred_over_ts_np[ts]

        # reverse scaling (use unscaled full_X_val, and apply inverse scaling to the predictions)
        norm_min_NIHSS_up_to_current_timestep = np.min(full_X_val[:, 0:ts+1, min_NIHSS_idx, -1], axis=1)
        reverse_scaled_predictions = scaler.inverse_transform(predictions_at_ts_np.reshape(-1, X_train.shape[-1])).reshape(predictions_at_ts_np.shape)
        norm_max_NIHSS_at_last_prediction_timestep = reverse_scaled_predictions[:, -1, max_NIHSS_idx]
        norm_delta_NIHSS_at_last_predicted_ts = norm_max_NIHSS_at_last_prediction_timestep - norm_min_NIHSS_up_to_current_timestep

        # reverse normalisation
        max_NIHSS_at_last_prediction_timestep = reverse_normalisation(norm_max_NIHSS_at_last_prediction_timestep, 'max_NIHSS', normalisation_parameters_df)
        min_NIHSS_up_to_current_timestep = reverse_normalisation(norm_min_NIHSS_up_to_current_timestep, 'min_NIHSS', normalisation_parameters_df)
        delta_NIHSS_at_last_predicted_ts = max_NIHSS_at_last_prediction_timestep - min_NIHSS_up_to_current_timestep

        # y_pred = norm_delta_NIHSS_at_last_predicted_ts
        y_pred = delta_NIHSS_at_last_predicted_ts
        y_pred_binary = delta_NIHSS_at_last_predicted_ts >= 4

        timestep_df = pd.DataFrame({'timestep': [ts] * len(y_true_at_evaluated_ts),
                                            'prediction': y_pred,
                                            'true_label': y_true_at_evaluated_ts})
        overall_prediction_df = pd.concat([overall_prediction_df, timestep_df])
        
        timesteps.append(ts)
        n_pos_samples.append(np.sum(y_true_at_evaluated_ts))
        accuracy_scores.append(accuracy_score(y_true_at_evaluated_ts, y_pred_binary))

        if len(np.unique(y_true_at_evaluated_ts)) == 1:
            roc_scores.append(np.nan)
            auprc_scores.append(np.nan)
            mcc_scores.append(np.nan)
        else:
            roc_scores.append(roc_auc_score(y_true_at_evaluated_ts, y_pred))
            auprc_scores.append(average_precision_score(y_true_at_evaluated_ts, y_pred))
            mcc_scores.append(matthews_corrcoef(y_true_at_evaluated_ts, y_pred_binary))

    # Ensure true_label is binary
    overall_prediction_df['true_label'] = overall_prediction_df['true_label'].astype(int)
    # Ensure prediction is a continuous value between 0 and 1
    overall_prediction_df['prediction'] = overall_prediction_df['prediction'].astype(float)


    # compute overall metrics
    overall_results_df = pd.DataFrame({'overall_roc': roc_auc_score(overall_prediction_df.true_label,
                                                                        overall_prediction_df.prediction),
                                    'overall_auprc': average_precision_score(overall_prediction_df.true_label,
                                                                            overall_prediction_df.prediction),
                                    'overall_mcc': matthews_corrcoef(overall_prediction_df.true_label,
                                                                        overall_prediction_df.prediction >= 4),
                                    'overall_accuracy': accuracy_score(overall_prediction_df.true_label,
                                                                        overall_prediction_df.prediction >= 4),
                                    'n_pos_samples': np.sum(overall_prediction_df.true_label),
                                    'n_samples': len(overall_prediction_df),
                                    'cv_fold': cv_fold
                                }, index=[0])
    
    median_results_df = pd.DataFrame({'median_roc': np.nanmedian(roc_scores),
                                        'median_auprc': np.nanmedian(auprc_scores),
                                        'median_mcc': np.nanmedian(mcc_scores),
                                       'median_accuracy': np.nanmedian(accuracy_scores),
                                       'n_pos_samples': np.nanmedian(n_pos_samples),
                                   }, index=[0])
    
    # save results
    overall_prediction_df.to_csv(os.path.join(output_path, 'overall_validation_predictions.csv'))
    overall_results_df.to_csv(os.path.join(output_path, 'overall_validation_results.csv'))
    median_results_df.to_csv(os.path.join(output_path, 'median_validation_results.csv'))
    # save individual scores per time step
    pd.DataFrame({'timestep': timesteps, 'roc': roc_scores, 'auprc': auprc_scores, 'mcc': mcc_scores,
                    'accuracy': accuracy_scores, 'n_pos_samples': n_pos_samples
                    }).to_csv(
        os.path.join(output_path, 'validation_scores_per_timestep.csv'))

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
    plt.savefig(os.path.join(output_path, 'validation_scores_over_time.png'))
            
    return overall_results_df

if __name__ == '__main__':
    # Example usage
    # python dec_val_evaluation -d /path/to/data -m /path/to/model -c /path/to/model_config -n /path/to/normalisation_data -o /path/to/output
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=False)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-n', '--normalisation_data_path', type=str, required=True)
    parser.add_argument('-o', '--outcome_data_path', type=str, required=True)
    parser.add_argument('-out', '--output_path', type=str, default=None)
    parser.add_argument('-p', '--predictions_dir', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_time_steps', type=int, default=72)
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6)
    args = parser.parse_args()

    encoder_decoder_validation_evaluation(
        data_path=args.data_path,
        model_config_path=args.model_config_path,
        model_path=args.model_path,
        normalisation_data_path=args.normalisation_data_path,
        predictions_path=args.predictions_dir,
        outcome_data_path=args.outcome_data_path,
        output_path=args.output_path,
        use_gpu=args.use_gpu,
        n_time_steps=args.n_time_steps,
        eval_n_time_steps_before_event=args.eval_n_time_steps_before_event
    )
