#!/usr/bin/env python
# coding: utf-8
# %%

# %%


import optuna
from functools import partial
import torch as ch
import json
import torch as ch
import matplotlib.pyplot as plt
import os, traceback
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, matthews_corrcoef
import argparse
import sys
import os
from functools import partial
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
import optuna
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
import os
from functools import partial
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics import AUROC

sys.path.insert(0, '/home/gridsan/gleclerc/julian/OPSUM/')


from prediction.outcome_prediction.LSTM.training.utils import initiate_log_files
from prediction.outcome_prediction.data_loading.data_formatting import format_to_2d_table_with_time,     link_patient_id_to_outcome, features_to_numpy, numpy_to_lookup_table, feature_order_verification
from prediction.utils.scoring import precision, matthews, recall
from prediction.utils.utils import generate_balanced_arrays, check_data, ensure_dir, save_json
from prediction.outcome_prediction.LSTM.LSTM import lstm_generator

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer

from sklearn.preprocessing import StandardScaler

from pytorch_lightning.loggers import LightningLoggerBase
class DictLogger(LightningLoggerBase):
    """PyTorch Lightning `dict` logger."""

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.metrics = []
        self._version = version

    def log_metrics(self, metrics, step=None):
        self.metrics.append(metrics)

    @property
    def version(self):
        return self._version

    @property
    def experiment(self):
        """Return the experiment object associated with this logger."""

    def log_hyperparams(self, params):
        """
        Record hyperparameters.
        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
        """

    @property
    def name(self):
        """Return the experiment name."""
        return 'optuna'


# %%


study = optuna.create_study(direction='maximize')


# %%


def prepare_dataset(scenario):
    X_train, X_val, y_train, y_val = scenario
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 84)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, 84)).reshape(X_val.shape)
    train_dataset = TensorDataset(ch.from_numpy(X_train).cuda(), ch.from_numpy(y_train.astype(np.int32)).cuda())
    val_dataset = TensorDataset(ch.from_numpy(X_val).cuda(), ch.from_numpy(y_val.astype(np.int32)).cuda())
    return train_dataset, val_dataset


# %%


scenarios = ch.load('/home/gridsan/gleclerc/julian/data_splits_death.pth')
all_datasets = [prepare_dataset(x) for x in scenarios]


# %%


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
    
    def validation_step(self,batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.val_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        return optimizer


# %%


from uuid import uuid4
import json

def get_score(trial, all_ds):
    bs = trial.suggest_categorical("batch_size", choices=[32, 64])
    num_layers = trial.suggest_categorical("num_layers", choices=[1, 2, 4, 8, 12, 16])
    model_dim = trial.suggest_categorical("model_dim", choices=[128, 256, 512])
    train_noise = trial.suggest_loguniform("train_noise", 1e-5, 7)
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 10)
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    dropout = trial.suggest_uniform("dropout", 0, 1)
    num_heads = trial.suggest_categorical("num_head", [8])
    pos_encode_factor = trial.suggest_loguniform("pos_encode_factor", 1e-5, 10)
    lr = 1e-3
    scores = [] 
    ts = []
    
    for i, (train_dataset, val_dataset) in enumerate(all_ds):
        if i >= 3:
            break
        model = OPSUMTransformer(
            input_dim=84,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=1,
            max_dim=500,
            pos_encode_factor=pos_encode_factor
        )

        train_loader = DataLoader(train_dataset, batch_size=bs)
        val_loader = DataLoader(val_dataset, batch_size=256)
        logger = DictLogger(0)
        module = LitModel(model, lr, wd, train_noise)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=50,logger=logger,
                             callbacks=[])
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_score = np.max([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        train_score = np.max([x['train_auroc'] for x in logger.metrics if 'train_auroc' in x])
        if best_score < 0.80:
            return
        scores.append(best_score)
        ts.append(train_score)
    score = np.median(scores)
    d = dict(trial.params)
    d['score'] = score
    d['train_scores'] = np.median(ts)
    text = json.dumps(d)
    text += '\n'
    with open('/home/gridsan/gleclerc/julian/gridsearch.jsonl', 'a') as handle:
        handle.write(text)
    return score


# %%


study.optimize(partial(get_score, all_ds=all_datasets), n_trials=1000)

