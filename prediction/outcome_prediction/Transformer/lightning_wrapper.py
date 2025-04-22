import math
import torch as ch
from torch import optim
import pytorch_lightning as pl
from torchmetrics import AUROC, MeanAbsoluteError, AveragePrecision
from torchmetrics.classification import Accuracy
from torchmetrics.regression import CosineSimilarity, MeanAbsolutePercentageError
from flash.core.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from prediction.outcome_prediction.Transformer.architecture import OPSUM_encoder_decoder
from prediction.utils.utils import FocalLoss, APLoss, WeightedCosineSimilarity, WeightedMSELoss


class LitModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise, lr_warmup_steps=0,
                 loss_function='bce', alpha=0.25, gamma=2.0,
                 imbalance_factor=None, debug_mode=False, scheduler='exponential'):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.wd = wd
        self.train_noise = train_noise
        self.scheduler = scheduler

        if loss_function == 'bce':
            if imbalance_factor is not None:
                self.criterion = ch.nn.BCEWithLogitsLoss(pos_weight=imbalance_factor)
            else:
                self.criterion = ch.nn.BCEWithLogitsLoss()
        elif loss_function == 'focal':
            self.criterion = FocalLoss(alpha=alpha, gamma=gamma)
        elif loss_function == 'aploss':
            self.criterion = APLoss()


        self.train_accuracy = Accuracy(task='binary')
        self.train_accuracy_epoch = Accuracy(task='binary')
        self.val_accuracy_epoch = Accuracy(task='binary')
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")
        self.train_auroc_epoch = AUROC(task="binary")
        self.train_auprc = AveragePrecision(task="binary")
        self.train_auprc_epoch = AveragePrecision(task="binary")
        self.val_auprc = AveragePrecision(task="binary")


        # more logs
        self.debug_mode = debug_mode

    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            # x = x + ch.randn_like(x) * self.train_noise
            x = x + ch.randn(x.shape[0], x.shape[1], device=x.device)[:, :, None].repeat(1, 1, x.shape[2]) * self.train_noise
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.train_accuracy(predictions.ravel(), y.ravel())
        self.train_accuracy_epoch(predictions.ravel(), y.ravel())
        self.train_auroc(predictions.ravel(), y.ravel())
        self.train_auroc_epoch(predictions.ravel(), y.ravel())
        self.train_auprc(predictions.ravel(), y.ravel())
        self.train_auprc_epoch(predictions.ravel(), y.ravel())

        if self.debug_mode:
            self.log_dict(
                {
                    'train_loss': loss,
                    'train_auroc': self.train_auroc,
                    'train_auprc': self.train_auprc
                },
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            self.log_dict(
                {
                    'train_loss_epoch': loss,
                    'train_accuracy_epoch': self.train_accuracy_epoch,
                    'train_auroc_epoch': self.train_auroc_epoch,
                    'train_auprc_epoch': self.train_auprc_epoch
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )


        return loss

    def validation_step(self ,batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.val_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        self.val_auprc(ch.sigmoid(predictions.ravel()), y.ravel())

        self.log_dict({
            'val_auroc': self.val_auroc,
            'val_auprc': self.val_auprc
        }, on_step=False, on_epoch=True, prog_bar=True)

        if self.debug_mode:
            self.log_dict(
                {
                    'val_loss': loss,
                },
                on_step=False,
                on_epoch=True,
            )
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x).squeeze()
        return predictions

    def configure_optimizers(self):
        """
        Refs:
        - https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
        - https://github.com/Lightning-AI/lightning/issues/328#issuecomment-782845008
        """

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        if self.scheduler == 'exponential':
            train_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        elif self.scheduler == 'cosine':
            train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        if self.lr_warmup_steps == 0:
            return [optimizer], [train_scheduler]

        # using warmup scheduler
        def warmup(current_step: int):
            return 1 / (10 ** (float(self.lr_warmup_steps - current_step)))

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler],
                                                        [self.lr_warmup_steps])

        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]


class LitEncoderDecoderModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise, lr_warmup_steps=0, loss_function='weighted_mse'):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.wd = wd
        self.train_noise = train_noise

        if loss_function == 'mse':
            self.criterion = ch.nn.MSELoss()
        elif loss_function == 'weighted_mse':
            # using weighted mse loss with weighting of max_NIHSS
            self.criterion = WeightedMSELoss(weight_idx = 37, weight_value = 10, vector_length = 84)

        # self.train_cos_sim = CosineSimilarity(reduction='mean')
        # self.train_cos_sim_epoch = CosineSimilarity(reduction='mean')
        # self.val_cos_sim = CosineSimilarity(reduction='mean')
        self.train_cos_sim = WeightedCosineSimilarity(weight_idx = 37, weight_value = 10, vector_length = 84, reduction='mean')
        self.train_cos_sim_epoch = WeightedCosineSimilarity(weight_idx = 37, weight_value = 10, vector_length = 84, reduction='mean')
        self.val_cos_sim = WeightedCosineSimilarity(weight_idx = 37, weight_value = 10, vector_length = 84, reduction='mean')


    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            # x = x + ch.randn_like(x) * self.train_noise
            x = x + ch.randn(x.shape[0], x.shape[1], device=x.device)[:, :, None].repeat(1, 1, x.shape[2]) * self.train_noise

        # y_input is last step of x
        y_input = x[:, -1, :][:, None, :]

        predictions = self.model(x, y_input)

        loss = self.criterion(predictions, y)

        self.train_cos_sim(predictions.reshape(x.shape[0],-1), y.reshape(x.shape[0],-1))
        self.train_cos_sim_epoch(predictions.reshape(x.shape[0],-1), y.reshape(x.shape[0],-1))

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cos_sim", self.train_cos_sim_epoch, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        y_input = x[:, -1, :][:, None, :]
        predictions = self.model(x, y_input)

        loss = self.criterion(predictions, y)

        self.val_cos_sim(predictions.reshape(x.shape[0],-1), y.reshape(x.shape[0],-1))

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cos_sim", self.val_cos_sim, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_input = x[:, -1, :][:, None, :]
        predictions = self.model(x, y_input)
        return predictions

    def configure_optimizers(self):
        """
        Refs:
        - https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
        - https://github.com/Lightning-AI/lightning/issues/328#issuecomment-782845008
        """

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        train_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        if self.lr_warmup_steps == 0:
            return [optimizer], [train_scheduler]

        # using warmup scheduler
        def warmup(current_step: int):
            return 1 / (10 ** (float(self.lr_warmup_steps - current_step)))

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler],
                                                        [self.lr_warmup_steps])

        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]


class LitEncoderRegressionModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise, lr_warmup_steps=0, classification_threshold=None,
                 loss_function='rmse', scheduler='exponential', debug_mode=False):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.wd = wd
        self.train_noise = train_noise
        self.debug_mode = debug_mode
        self.scheduler = scheduler

        if loss_function == 'rmse':
            self.criterion = self.rmse_loss
        elif loss_function == 'log_cosh':
            self.criterion = self.log_cosh_loss

        # define monitoring metrics: MAE, MAPE
        self.train_mae = MeanAbsoluteError()
        self.train_mape = MeanAbsolutePercentageError()
        self.train_mae_epoch = MeanAbsoluteError()
        self.train_mape_epoch = MeanAbsolutePercentageError()
        self.val_mae = ch.nn.L1Loss()
        self.val_mape = MeanAbsolutePercentageError()
        self.val_auprc = AveragePrecision(task="binary")

        if classification_threshold is not None:
            self.classification_threshold = classification_threshold
            self.val_accuracy_epoch = Accuracy(task='binary')
            self.val_auroc = AUROC(task="binary")

    def log_cosh_loss(self, predictions, targets):
        """ Log-Cosh Loss: smoother alternative to exponential loss """
        diff = predictions - targets
        loss = ch.log(ch.cosh(diff + 1e-12))  # Small constant to avoid log(0)
        return loss.mean()

    def rmse_loss(self, predictions, targets):
        """ Computes Root Mean Squared Error (RMSE) Loss """
        loss = ch.sqrt(ch.nn.functional.mse_loss(predictions, targets))
        return loss

    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            # x = x + ch.randn_like(x) * self.train_noise
            x = x + ch.randn(x.shape[0], x.shape[1], device=x.device)[:, :, None].repeat(1, 1, x.shape[2]) * self.train_noise
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()

        # compute loss
        loss = self.criterion(predictions, y.float())

        self.train_mae(predictions.ravel(), y.ravel())
        self.train_mape(predictions.ravel(), y.ravel())
        self.train_mae_epoch(predictions.ravel(), y.ravel())
        self.train_mape_epoch(predictions.ravel(), y.ravel())

        if self.debug_mode:
            self.log_dict(
                {
                    'train_loss': loss,
                    'train_mae': self.train_mae,
                    'train_mape': self.train_mape
                },
                on_step=True,
                on_epoch=False,
                prog_bar=True
            )

            self.log_dict(
                {
                    'train_loss_epoch': loss,
                    'train_mae_epoch': self.train_mae_epoch,
                    'train_mape_epoch': self.train_mape_epoch,
                },
                on_step=False,
                on_epoch=True,
                prog_bar=True
            )
            # self.train_mae_epoch.reset()



        return loss

    def validation_step(self ,batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()

        # compute loss
        loss = self.criterion(predictions, y.float())

        val_mae = self.val_mae(predictions.ravel(), y.ravel())
        self.val_mape(predictions.ravel(), y.ravel())

        if self.classification_threshold is not None:
            # binary predictions (predicted event is within time to event threshold)
            binary_predictions = (predictions <= self.classification_threshold).float()
            # binary ground truth
            binary_y = (y <= self.classification_threshold).float()
            self.val_accuracy_epoch(binary_predictions, binary_y)
            self.val_auroc(binary_predictions, binary_y)
            self.val_auprc(binary_predictions, binary_y.int())

            # log
            self.log_dict({'val_mae': val_mae, 'val_mape': self.val_mape, 'val_accuracy': self.val_accuracy_epoch,
                           'val_loss': loss,
                           'val_auroc': self.val_auroc, 'val_auprc': self.val_auprc}, on_step=False,
                          on_epoch=True, prog_bar=True)

        else:
            # log mae and mape and loss
            self.log_dict({'val_mae': val_mae, 'val_mape': self.val_mape, 'val_loss': loss}, on_step=False,
                          on_epoch=True, prog_bar=True)
        self.val_mape.reset()


        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        predictions = self.model(x).squeeze()
        return predictions

    def configure_optimizers(self):
        """
        Refs:
        - https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
        - https://github.com/Lightning-AI/lightning/issues/328#issuecomment-782845008
        """

        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = optimizer.defaults['lr']

        if self.scheduler == 'exponential':
            train_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, 0.99)
        elif self.scheduler == 'cosine':
            # train_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs, eta_min=1e-6)
            # train_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
            #                                                                  T_0=10,
            #                                                                  T_mult=2,
            #                                                                  eta_min=1e-5
            #                                                                  )
            train_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=self.lr_warmup_steps, max_epochs=self.trainer.max_epochs)
            return [optimizer],  [
                        {
                            'scheduler': train_scheduler,
                            'interval': 'step',
                            'frequency': 1
                        }
                    ]

        if self.lr_warmup_steps == 0:
            return [optimizer], [train_scheduler]

        # using warmup scheduler
        def warmup(current_step: int):
            if current_step >= self.lr_warmup_steps:
                return 1 / math.sqrt(1 + (current_step / self.lr_warmup_steps))  # Inverse square root decay
            return max(1e-8, 1 / (10 ** float(min(self.lr_warmup_steps - current_step, 50))))  # Clamped

        warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)

        scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, train_scheduler],
                                                    [self.lr_warmup_steps])

        return [optimizer], [
            {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        ]