import numpy as np

try:
    from pytorch_lightning.loggers import Logger
except ImportError:
    from pytorch_lightning.loggers import LightningLoggerBase
    Logger = LightningLoggerBase
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
import torch as ch


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

def prepare_torch_dataset(X_train, y_train, X_val = None, y_val=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.reshape(-1, 84)).reshape(X_train.shape)
    train_dataset = TensorDataset(ch.from_numpy(X_train).cuda(), ch.from_numpy(y_train.astype(np.int32)).cuda())

    if X_val is None:
        return train_dataset, None
    else:
        X_val = scaler.transform(X_val.reshape(-1, 84)).reshape(X_val.shape)
        val_dataset = TensorDataset(ch.from_numpy(X_val).cuda(), ch.from_numpy(y_val.astype(np.int32)).cuda())
        return train_dataset, val_dataset