from sklearn.preprocessing import StandardScaler
import numpy as np
import torch as ch
from torch.utils.data import TensorDataset

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
        print(metrics)
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


def prepare_dataset(scenario, balanced=False, aggregate=False, rescale=True, use_gpu=True):
    X_train, X_val, y_train, y_val = scenario
    scaler = StandardScaler()

    if rescale:
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)

    if balanced:
        X_train_neg = X_train[y_train == 0]
        X_train_pos = X_train[np.random.choice(np.where(y_train==1)[0], X_train_neg.shape[0])]
        X_train = np.concatenate([X_train_neg, X_train_pos])
        y_train = np.concatenate([np.zeros(X_train_neg.shape[0]), np.ones(X_train_pos.shape[0])])

    if aggregate:
        X_train = feature_aggregation(X_train)
        X_val = feature_aggregation(X_val)

    if use_gpu:
        train_dataset = TensorDataset(ch.from_numpy(X_train).cuda(), ch.from_numpy(y_train.astype(np.int32)).cuda())
        val_dataset = TensorDataset(ch.from_numpy(X_val).cuda(), ch.from_numpy(y_val.astype(np.int32)).cuda())
    else:
        train_dataset = TensorDataset(ch.from_numpy(X_train), ch.from_numpy(y_train.astype(np.int32)))
        val_dataset = TensorDataset(ch.from_numpy(X_val), ch.from_numpy(y_val.astype(np.int32)))
    return train_dataset, val_dataset


def feature_aggregation(x):
    # given an array of shape (batch_size, seq_len, feature_dim), for each feature_dim, compute the cumulative average / max / min and add them to the feature_dim
    time_avg = np.cumsum(x, axis=1) / np.arange(1, x.shape[1] + 1)[:, np.newaxis]
    time_max = np.maximum.accumulate(x, axis=1)
    time_min = np.minimum.accumulate(x, axis=1)
    return np.concatenate([x, time_avg, time_max, time_min], axis=-1).astype(np.float32)

