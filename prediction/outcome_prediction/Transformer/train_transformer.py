import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch as ch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from torchmetrics import AUROC

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from prediction.outcome_prediction.Transformer.utils import prepare_torch_dataset, DictLogger
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time, \
    link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table
from prediction.utils.utils import check_data, save_json, ensure_dir


def prepare_train_data(features_path: str, labels_path:str, output_dir:str, outcome:str, test_size:float, seed=0, use_cross_validation=False, n_splits=5):
    ### LOAD THE DATA
    X, y = format_to_2d_table_with_time(feature_df_path=features_path, outcome_df_path=labels_path,
                                        outcome=outcome)

    n_time_steps = X.relative_sample_date_hourly_cat.max() + 1
    n_channels = X.sample_label.unique().shape[0]

    # test if data is corrupted
    check_data(X)

    """
    SPLITTING DATA
    Splitting is done by patient id (and not admission id) as in case of the rare multiple admissions per patient there
    would be a risk of data leakage otherwise split 'pid' in TRAIN and TEST pid = unique patient_id
    """
    # Reduce every patient to a single outcome (to avoid duplicates)
    all_pids_with_outcome = link_patient_id_to_outcome(y, outcome)
    pid_train, pid_test, y_pid_train, y_pid_test = train_test_split(all_pids_with_outcome.patient_id.tolist(),
                                                                    all_pids_with_outcome.outcome.tolist(),
                                                                    stratify=all_pids_with_outcome.outcome.tolist(),
                                                                    test_size=test_size,
                                                                    random_state=seed)

    # save patient ids used for testing / training
    pd.DataFrame(pid_train, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_train.tsv'),
        sep='\t', index=False)
    pd.DataFrame(pid_test, columns=['patient_id']).to_csv(
        os.path.join(output_dir, 'pid_test.tsv'),
        sep='\t', index=False)


    if not use_cross_validation:
        train_X_df = X[X.patient_id.isin(pid_train)]
        train_y_df = y[y.patient_id.isin(pid_train)]

        train_X_np = features_to_numpy(train_X_df,
                                      ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label', 'value'])
        train_y_np = np.array([train_y_df[train_y_df.case_admission_id == cid].outcome.values[0] for cid in
                              train_X_np[:, 0, 0, 0]]).astype('float32')

        # create look-up table for case_admission_ids, sample_labels and relative_sample_date_hourly_cat
        save_json(numpy_to_lookup_table(train_X_np),
                  os.path.join(output_dir, 'test_lookup_dict.json'))

        # Remove the case_admission_id, sample_label, and time_step_label columns from the data
        train_X_np = train_X_np[:, :, :, -1].astype('float32')

        return train_X_np, train_y_np, None, None

    else:
        # define K fold
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        X_train_folds = []
        y_train_folds = []
        X_val_folds = []
        y_val_folds = []

        ### TRAIN MODEL USING K-FOLD CROSS-VALIDATION
        for fold_pid_train_idx, fold_pid_val_idx in kfold.split(pid_train, y_pid_train):
            fold_train_pidx = np.array(pid_train)[fold_pid_train_idx]
            fold_val_pidx = np.array(pid_train)[fold_pid_val_idx]

            fold_X_train_df = X.loc[X.patient_id.isin(fold_train_pidx)]
            fold_y_train_df = y.loc[y.patient_id.isin(fold_train_pidx)]
            fold_X_val_df = X.loc[X.patient_id.isin(fold_val_pidx)]
            fold_y_val_df = y.loc[y.patient_id.isin(fold_val_pidx)]

            fold_X_train = features_to_numpy(fold_X_train_df,
                                             ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label',
                                              'value'])
            fold_X_val = features_to_numpy(fold_X_val_df,
                                           ['case_admission_id', 'relative_sample_date_hourly_cat', 'sample_label',
                                            'value'])

            fold_y_train = np.array(
                [fold_y_train_df[fold_y_train_df.case_admission_id == cid].outcome.values[0] for cid in
                 fold_X_train[:, 0, 0, 0]]).astype('float32')
            fold_y_val = np.array([fold_y_val_df[fold_y_val_df.case_admission_id == cid].outcome.values[0] for cid in
                                   fold_X_val[:, 0, 0, 0]]).astype('float32')

            fold_X_train = fold_X_train[:, :, :, -1].astype('float32')
            fold_X_val = fold_X_val[:, :, :, -1].astype('float32')

            X_train_folds.append(fold_X_train)
            y_train_folds.append(fold_y_train)
            X_val_folds.append(fold_X_val)
            y_val_folds.append(fold_y_val)
        return X_train_folds, y_train_folds, X_val_folds, y_val_folds

# define the LightningModule
class LitModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.train_noise = train_noise
        self.criterion = ch.nn.BCEWithLogitsLoss()
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            x = x + ch.randn_like(x) * self.train_noise
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.train_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        self.log("train_auroc", self.train_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.val_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = ch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer



def train_transformer(features_path: str, labels_path: str, outcome: str, test_size: float, seed: int,
                      output_dir: str, num_layers: int, model_dim: int, dropout: float, ff_factor: float,
                      num_heads: int, pos_encode_factor: float, bs: int, lr: float, weight_decay: float, train_noise: float,
                      max_epochs: int = 50, max_dim: int = 500):
    saved_args = locals().copy()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # save training parameters
    training_params_filename = f'params_opsum_transformer_{timestamp}.json'
    with open(os.path.join(output_dir, training_params_filename), 'w') as fp:
        json.dump(saved_args, fp, indent=4)

    ff_dim = ff_factor * model_dim

    train_X_folds, train_y_folds, val_X_folds, val_y_folds = prepare_train_data(features_path=features_path, labels_path=labels_path, output_dir=output_dir,
                                                                outcome=outcome, test_size=test_size, seed=seed, use_cross_validation=True)

    best_scores, best_epochs, best_train_scores = [], [], []

    for i, (train_X, train_y, val_X, val_y) in enumerate(zip(train_X_folds, train_y_folds, val_X_folds, val_y_folds)):
        checkpoint_dir = os.path.join(output_dir, f'checkpoints_opsum_transformer_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)

        train_dataset, val_dataset = prepare_torch_dataset(train_X, train_y, val_X, val_y)

        train_loader = DataLoader(train_dataset, batch_size=bs)
        val_loader = DataLoader(val_dataset, batch_size=256)

        model = OPSUMTransformer(
            input_dim=84,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=1,
            max_dim=max_dim,
            pos_encode_factor=pos_encode_factor
        )

        module = LitModel(model, lr, weight_decay, train_noise)
        logger = DictLogger(0)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=2,
            monitor="val_auroc",
            mode="max",
            dirpath=checkpoint_dir,
            filename="opsum_transformer_{epoch:02d}_{val_auroc:.4f}",
        )

        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=max_epochs, logger=logger,
                             callbacks=[checkpoint_callback], default_root_dir=output_dir)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)

        best_scores.append(np.max([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x]))
        best_epochs.append(np.argmax([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x]))
        best_train_scores.append(np.max([x['train_auroc'] for x in logger.metrics if 'train_auroc' in x]))

    return best_epochs, best_scores, best_train_scores


features_path = '/home/klug/data/opsum/72h_input_data/gsu_prepro_01012023_233050/preprocessed_features_01012023_233050.csv'
labels_path = '/home/klug/data/opsum/72h_input_data/gsu_prepro_01012023_233050/preprocessed_outcomes_01012023_233050.csv'
outcome = '3M mRS 0-2'
test_size = 0.2
seed = 5
output_dir = '/home/klug/output/opsum/transformer_evaluation'
num_layers = 5
model_dim = 1024
dropout = 0.166952
ff_factor = 2
num_heads = 8
pos_encode_factor = 1.5
bs = 32
lr = 1e-3
weight_decay = 0.000496
train_noise = 2.64

best_epochs, best_scores, best_train_scores = train_transformer(features_path=features_path, labels_path=labels_path, outcome=outcome,
                                test_size=test_size, seed=seed, output_dir=output_dir, num_layers=num_layers,
                                model_dim=model_dim, dropout=dropout, ff_factor=ff_factor, num_heads=num_heads,
                                pos_encode_factor=pos_encode_factor, bs=bs, lr=lr, weight_decay=weight_decay,
                                train_noise=train_noise, max_epochs=50, max_dim=500)
print(f'Best epochs: {best_epochs}, with median {np.median(best_epochs)}')
print(f'Best scores: {best_scores}, with median {np.median(best_scores)}')
print(f'Best train scores: {best_train_scores}, with median {np.median(best_train_scores)}')


