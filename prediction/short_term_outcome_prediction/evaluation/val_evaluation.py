import os

import pandas as pd
import torch as ch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import pytorch_lightning as pl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import seaborn as sns

from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitModel
from prediction.utils.utils import ensure_dir

def validation_evaluation(data_path:str, model_config_path:str, model_path:str=None,
                          output_path=None, predictions_path:str=None,
                            use_gpu = False,  n_time_steps = 72, eval_n_time_steps_before_event = 6):
    """
    Evaluate the model on the validation set
    Gist: Use a seperate evaluation then the one formalised in the model training
    """
    model_config = pd.read_csv(model_config_path)

    if output_path is None:
        output_path = os.path.join(os.path.dirname(model_config_path), 'validation_evaluation_results')
    ensure_dir(output_path)

    # Load the data
    splits = ch.load(os.path.join(data_path))
    full_X_train, full_X_val, y_train, y_val = splits[model_config['best_cv_fold'].values[0]]

    # prepare input data
    X_train = full_X_train[:, :, :, -1].astype('float32')
    X_val = full_X_val[:, :, :, -1].astype('float32')

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)

    # if predictions are precomputed, load them
    if predictions_path is not None:
        predictions_data = ch.load(predictions_path)
        pred_over_ts_np = np.squeeze(predictions_data).T

    # else load model and compute predictions
    else:
        # load model
        accelerator = 'gpu' if use_gpu else 'cpu'

        logger = DictLogger(0)
        trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=1000,
                             gradient_clip_val=model_config['grad_clip_value'], logger=logger)

        # define model
        ff_factor = 2
        ff_dim = ff_factor * model_config['model_dim']
        pos_encode_factor = 1

        model_architecture = OPSUMTransformer(
            input_dim=84,
            num_layers=int(model_config['num_layers']),
            model_dim=int(model_config['model_dim']),
            dropout=int(model_config['dropout']),
            ff_dim=int(ff_dim),
            num_heads=int(model_config['num_head']),
            num_classes=1,
            max_dim=500,
            pos_encode_factor=pos_encode_factor
        )

        trained_model = LitModel.load_from_checkpoint(checkpoint_path=model_path, model=model_architecture,
                                                      lr=model_config['lr'],
                                                      wd=model_config['weight_decay'],
                                                      train_noise=model_config['train_noise'],
                                                      imbalance_factor=ch.tensor(model_config['imbalance_factor']))

        # compute predictions
        pred_over_ts = []
        for ts in tqdm(range(n_time_steps)):
            modified_time_steps = ts + 1

            X_val_with_first_n_ts = X_val[:, 0:modified_time_steps, :]
            y_placeholder = ch.zeros((X_val_with_first_n_ts.shape[0], 1))
            if use_gpu:
                val_dataset = TensorDataset(ch.from_numpy(X_val_with_first_n_ts).cuda(), y_placeholder.cuda())
            else:
                val_dataset = TensorDataset(ch.from_numpy(X_val_with_first_n_ts), y_placeholder)

            val_loader = DataLoader(val_dataset, batch_size=1024)
            if ts == 0:
                y_pred = np.array(ch.sigmoid(trainer.predict(trained_model, val_loader)[0]))
            else:
                y_pred = np.array(ch.sigmoid(trainer.predict(trained_model, val_loader)[0])[:, -1])

            pred_over_ts.append(np.squeeze(y_pred))

        pred_over_ts_np = np.squeeze(pred_over_ts).T


    # construct y 
    y_val_list = []
    for cid in full_X_val[:, 0, 0, 0]:
        if cid not in y_val.case_admission_id.values:
            cid_y = np.zeros(n_time_steps)
        else:
            cid_event_ts = y_val[y_val.case_admission_id == cid].relative_sample_date_hourly_cat.values
            if cid_event_ts < (eval_n_time_steps_before_event + 1):
                # if the event occurs before a detection window, ignore the patient
                cid_y = np.array([])
            else:
                # let y be 0s until 6 hours before the event then stop the series
                cid_y = np.zeros(int(cid_event_ts) - eval_n_time_steps_before_event - 1)
                cid_y = np.append(cid_y, 1)

        y_val_list.append(cid_y)

    
    # evaluate predictions
    # compute roc scores for each time step
    roc_scores = []
    for ts in range(n_time_steps):
        pts_idx = [i for i, y in enumerate(y_val_list) if len(y) > ts]
        y_true = np.array([y[ts] for y in y_val_list if len(y) > ts])
        y_pred = pred_over_ts_np[pts_idx, ts]
        if len(np.unique(y_true)) == 1:
            roc_scores.append(np.nan)
        else:
            roc_scores.append(roc_auc_score(y_true, y_pred))

    # compute auprc scores for each time step
    auprc_scores = []
    for ts in range(n_time_steps):
        pts_idx = [i for i, y in enumerate(y_val_list) if len(y) > ts]
        y_true = np.array([y[ts] for y in y_val_list if len(y) > ts])
        y_pred = pred_over_ts_np[pts_idx, ts]
        if len(np.unique(y_true)) == 1:
            auprc_scores.append(np.nan)
        else:
            # auprc_scores.append(binary_auprc(y_true, y_pred))
            auprc_scores.append(average_precision_score(y_true, y_pred))

    # compute MCC scores for each time step
    mcc_scores = []
    for ts in range(n_time_steps):
        pts_idx = [i for i, y in enumerate(y_val_list) if len(y) > ts]
        y_true = np.array([y[ts] for y in y_val_list if len(y) > ts])
        y_pred = pred_over_ts_np[pts_idx, ts]
        if len(np.unique(y_true)) == 1:
            mcc_scores.append(np.nan)
        else:
            mcc_scores.append(matthews_corrcoef(y_true, np.where(y_pred > 0.5, 1, 0)))

    # count number of positive samples for each time step
    n_pos_samples = []
    for ts in range(n_time_steps):
        pts_idx = [i for i, y in enumerate(y_val_list) if len(y) > ts]
        y_true = np.array([y[ts] for y in y_val_list if len(y) > ts])
        n_pos_samples.append(np.sum(y_true))
    
    # median over time for each metric
    results_df = pd.DataFrame({'roc': np.nanmedian(roc_scores),
                                    'auprc': np.nanmedian(auprc_scores),
                                    'mcc': np.nanmedian(mcc_scores)}, index=[0])


    # save results
    results_df.to_csv(os.path.join(output_path, 'validation_results.csv'))
    # save individual socres
    pd.DataFrame({'roc': roc_scores, 'auprc': auprc_scores, 'mcc': mcc_scores}).to_csv(
        os.path.join(output_path, 'validation_scores.csv'))

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

    with PdfPages(os.path.join(output_path, 'predictions_over_time.pdf')) as pdf:
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
                ax.axvline(
                    x=y_val[y_val.case_admission_id == cid].relative_sample_date_hourly_cat.values,
                    color='red'
                )

            pdf.savefig()  # Save the current figure into the PDF
            plt.close()  # Close the figure to free memory

    return results_df


if __name__ == '__main__':
    # Example usage
    # python val_evaluation.py --data_path /path/to/data --model_path /path/to/model --model_config_path /path/to/model_config
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-m', '--model_path', type=str, required=False)
    parser.add_argument('-c', '--model_config_path', type=str, required=True)
    parser.add_argument('-o', '--output_path', type=str, default=None)
    parser.add_argument('-p', '--predictions_path', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true')
    parser.add_argument('--n_time_steps', type=int, default=72)
    parser.add_argument('--eval_n_time_steps_before_event', type=int, default=6)
    args = parser.parse_args()

    validation_evaluation(data_path=args.data_path,
                          model_path=args.model_path,
                          model_config_path=args.model_config_path,
                          output_path=args.output_path,
                            predictions_path=args.predictions_path,
                            use_gpu=args.use_gpu,
                            n_time_steps=args.n_time_steps,
                            eval_n_time_steps_before_event=args.eval_n_time_steps_before_event)


