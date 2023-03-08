#!/usr/bin/env python
# coding: utf-8

'''
TESTING ORIGINAL GRIDSEARCH CODE
'''


import torch as ch
import numpy as np
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import AUROC

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer
from sklearn.preprocessing import StandardScaler

try:
    from pytorch_lightning.loggers import Logger
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase
    Logger = LightningLoggerBase

class DictLogger(Logger):
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




def prepare_dataset(scenario):
    X_train, X_val, y_train, y_val = scenario
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 84)).reshape(X_train.shape)
    X_val = scaler.transform(X_val.reshape(-1, 84)).reshape(X_val.shape)
    train_dataset = TensorDataset(ch.from_numpy(X_train).cuda(), ch.from_numpy(y_train.astype(np.int32)).cuda())
    val_dataset = TensorDataset(ch.from_numpy(X_val).cuda(), ch.from_numpy(y_val.astype(np.int32)).cuda())
    return train_dataset, val_dataset


# %%


scenarios = ch.load('/home/klug/temp/data_splits.pth')
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


import json

def get_score(all_ds):
    outcome = '3M mRS 0-2'
    num_layers = 5
    model_dim = 1024
    dropout = 0.166952
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    num_heads = 8
    pos_encode_factor = 1.5
    bs = 32
    lr = 1e-3
    weight_decay = 0.000496
    train_noise = 2.64

    scores = [] 
    ts = []
    best_epochs = []
    
    for i, (train_dataset, val_dataset) in enumerate(all_ds):
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
        module = LitModel(model, lr, weight_decay, train_noise)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=50,logger=logger,
                             callbacks=[])
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        best_score = np.max([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_epoch = np.argmax([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        train_score = np.max([x['train_auroc'] for x in logger.metrics if 'train_auroc' in x])
        if best_score < 0.80:
            return
        scores.append(best_score)
        ts.append(train_score)
        best_epochs.append(best_epoch)
    score = np.median(scores)

    print(f'Best score: {score:.3f} with all scores: {scores}')
    print(f'Best epochs: {best_epochs} with median: {np.median(best_epochs)}')
    print(f'Train scores: {ts}, with median: {np.median(ts)}')

    d = dict()
    d['score'] = score
    d['train_scores'] = np.median(ts)
    text = json.dumps(d)
    text += '\n'
    with open('/home/klug/temp/gridsearch.jsonl', 'a') as handle:
        handle.write(text)
    return score



get_score(all_datasets)
