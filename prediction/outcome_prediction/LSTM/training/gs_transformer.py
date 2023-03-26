import os
from functools import partial
from datetime import datetime
import optuna
import torch as ch
from os import path
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import optim, nn, utils, Tensor
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics.classification import Accuracy
from pytorch_lightning.callbacks.callback import Callback
from sklearn.preprocessing import StandardScaler

from prediction.utils.utils import ensure_dir

INPUT_FOLDER = '/home/gl/gsu_prepro_01012023_233050/data_splits'
OUTPUT_FOLDER = '/mnt/data1/klug/output/transformer_evaluation/guillaume_v5'
OUTPUT_FOLDER = path.join(OUTPUT_FOLDER, f'transformer_gs_{datetime.now().strftime("%Y%m%d_%H%M%S")}')

from prediction.outcome_prediction.Transformer.architecture import OPSUMTransformer

ch.set_float32_matmul_precision('high')

try:
    from pytorch_lightning.loggers import LightningLoggerBase
except: 
    from pytorch_lightning.loggers.logger import Logger as LightningLoggerBase



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

    
class MyEarlyStopping(Callback):
    
    best_so_far = 0
    last_improvement = 0
    
    def __init__(self):
        super().__init__()
        
    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_auroc = logs['val_auroc'].item()
        
        if val_auroc > self.best_so_far:
            self.last_improvement = 0
        else:
            self.last_improvement += 1
            
        print(self.last_improvement)
        trainer.should_stop = val_auroc < 0.75 * self.best_so_far or self.last_improvement > 10 or (trainer.current_epoch > 10 and val_auroc < 0.55)
        
        self.best_so_far = max(val_auroc, self.best_so_far)


study = optuna.create_study(direction='maximize')

def feature_aggregation(x):
    # given an array of shape (batch_size, seq_len, feature_dim), for each feature_dim, compute the cumulative average / max / min and add them to the feature_dim
    time_avg = np.cumsum(x, axis=1) / np.arange(1, x.shape[1] + 1)[:, np.newaxis]
    time_max = np.maximum.accumulate(x, axis=1)
    time_min = np.minimum.accumulate(x, axis=1)
    return np.concatenate([x, time_avg, time_max, time_min], axis=-1).astype(np.float32)


def prepare_dataset(scenario, balanced=False, aggregate=False, rescale=True):
    X_train, X_val, y_train, y_val = scenario
    scaler = StandardScaler()

    if rescale:
        X_train = scaler.fit_transform(X_train.reshape(-1, 84)).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, 84)).reshape(X_val.shape)

    if balanced:
        X_train_neg = X_train[y_train == 0]
        X_train_pos = X_train[np.random.choice(np.where(y_train==1)[0], X_train_neg.shape[0])]
        X_train = np.concatenate([X_train_neg, X_train_pos])
        y_train = np.concatenate([np.zeros(X_train_neg.shape[0]), np.ones(X_train_pos.shape[0])])

    if aggregate:
        X_train = feature_aggregation(X_train)
        X_val = feature_aggregation(X_val)

    train_dataset = TensorDataset(ch.from_numpy(X_train).cuda(), ch.from_numpy(y_train.astype(np.int32)).cuda())
    val_dataset = TensorDataset(ch.from_numpy(X_val).cuda(), ch.from_numpy(y_val.astype(np.int32)).cuda())
    return train_dataset, val_dataset

scenarios = ch.load(path.join(INPUT_FOLDER, 'train_data_splits_3M_mRS_0-2_ts0.8_rs42_ns5.pth'))
all_datasets = [prepare_dataset(x) for x in scenarios]
all_datasets_balanced = [prepare_dataset(x, True) for x in scenarios]
all_datasets_aggregated = [prepare_dataset(x, False, True) for x in scenarios]

class LitModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise):
        super().__init__()
        self.model = model
        self.lr = lr
        self.wd = wd
        self.train_noise = train_noise
        self.criterion = ch.nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.train_accuracy_epoch = Accuracy(task='binary')
        self.val_accuracy_epoch = Accuracy(task='binary')
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            #x = x + ch.randn_like(x) * self.train_noise
            x = x + ch.randn(x.shape[0], x.shape[1], device=x.device)[:, :, None].repeat(1, 1, x.shape[2]) * self.train_noise
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.train_accuracy(predictions.ravel(), y.ravel())
        self.train_accuracy_epoch(predictions.ravel(), y.ravel())
        # self.train_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        # self.log("train_auroc", self.train_auroc, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        # self.log("train_loss_epoch", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("train_acc_epoch", self.train_accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.val_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        # self.val_accuracy_epoch(predictions.ravel(), y.ravel())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        # self.log("val_accuracy", self.val_accuracy_epoch, on_step=False, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.wd)
        
        return [optimizer], [optim.lr_scheduler.ExponentialLR(optimizer, 0.99)]


# %%


import json

def get_score(trial, all_ds):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ds_ub, ds_b, ds_a = all_ds
    bs = trial.suggest_categorical("batch_size", choices=[16, 32])
    num_layers = trial.suggest_categorical("num_layers", choices=[3, 6, 9])
    model_dim = trial.suggest_categorical("model_dim", choices=[512, 1024, 2048])
    train_noise = trial.suggest_loguniform("train_noise", 1e-5, 7)
    is_balanced = trial.suggest_categorical("balanced", [False])
    is_aggregated = trial.suggest_categorical("feature_aggregation", [False])
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 1e-2)
    ff_factor = 2
    ff_dim = ff_factor * model_dim
    dropout = trial.suggest_uniform("dropout", 0 ,0.5)
    num_heads = trial.suggest_categorical("num_head", [2, 4, 8])
    pos_encode_factor = 1
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-3)
    grad_clip = trial.suggest_loguniform('grad_clip_value', 1e-3, 2)

    val_scores = []
    best_epochs = []
    rolling_val_scores = []

    ds = ds_b if is_balanced else ds_ub
    ds = ds_a if is_aggregated else ds

    input_dim = 84 * 4 if is_aggregated else 84
    
    for i, (train_dataset, val_dataset) in enumerate(ds):
        checkpoint_dir = os.path.join(OUTPUT_FOLDER, f'checkpoints_opsum_transformer_{timestamp}_cv_{i}')
        ensure_dir(checkpoint_dir)
        model = OPSUMTransformer(
            input_dim=input_dim,
            num_layers=num_layers,
            model_dim=model_dim,
            dropout=dropout,
            ff_dim=ff_dim,
            num_heads=num_heads,
            num_classes=1,
            max_dim=500,
            pos_encode_factor=pos_encode_factor
        )

        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, drop_last=True )
        val_loader = DataLoader(val_dataset, batch_size=1024)
        logger = DictLogger(0)

        checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_auroc",
            mode="max",
            dirpath=checkpoint_dir,
            filename="opsum_transformer_{epoch:02d}_{val_auroc:.4f}",
        )

        module = LitModel(model, lr, wd, train_noise)
        trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=1000,logger=logger,
                             log_every_n_steps = 25, enable_checkpointing=True,
                             callbacks=[MyEarlyStopping(), checkpoint_callback], gradient_clip_val=grad_clip)
        trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
        val_aurocs = np.array([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_idx = np.argmax(val_aurocs)
        actual_score = np.median(val_aurocs[max(0, best_idx -1): best_idx + 2])

        best_val_score = np.max([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        best_epoch = np.argmax([x['val_auroc'] for x in logger.metrics if 'val_auroc' in x])
        val_scores.append(best_val_score)
        best_epochs.append(best_epoch)
        rolling_val_scores.append(actual_score)

    d = dict(trial.params)
    d['median_rolling_val_scores'] = float(np.median(rolling_val_scores))
    d['median_val_scores'] = float(np.median(val_scores))
    d['median_best_epochs'] = float(np.median(best_epochs))
    d['timestamp'] = timestamp
    d['best_cv_fold'] = int(np.argmax(val_scores))
    d['worst_cv_fold_val_score'] = float(np.min(val_scores))
    text = json.dumps(d)
    text += '\n'
    dest = path.join(OUTPUT_FOLDER, 'gridsearch.jsonl')
    with open(dest, 'a') as handle:
        handle.write(text)
    print("WRITTEN in ", dest)
    return np.median(rolling_val_scores)

study.optimize(partial(get_score, all_ds=(all_datasets, all_datasets_balanced, all_datasets_aggregated)), n_trials=1000)

