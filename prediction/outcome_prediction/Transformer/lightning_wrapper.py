import torch as ch
from torch import optim
import pytorch_lightning as pl
from torchmetrics import AUROC
from torchmetrics.classification import Accuracy
from torchmetrics.regression import CosineSimilarity

from prediction.outcome_prediction.Transformer.architecture import OPSUM_encoder_decoder


class LitModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise, lr_warmup_steps=0, imbalance_factor=None):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.wd = wd
        self.train_noise = train_noise
        if imbalance_factor is not None:
            self.criterion = ch.nn.BCEWithLogitsLoss(pos_weight=imbalance_factor)
        else:
            self.criterion = ch.nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='binary')
        self.train_accuracy_epoch = Accuracy(task='binary')
        self.val_accuracy_epoch = Accuracy(task='binary')
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

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
        return loss

    def validation_step(self ,batch, batch_idx, mode='train'):
        x, y = batch
        predictions = self.model(x).squeeze().ravel()
        y = y.unsqueeze(1).repeat(1, x.shape[1]).ravel()
        loss = self.criterion(predictions, y.float()).ravel()
        self.val_auroc(ch.sigmoid(predictions.ravel()), y.ravel())
        self.log("val_auroc", self.val_auroc, on_step=False, on_epoch=True, prog_bar=True)
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


class LitEncoderDecoderModel(pl.LightningModule):
    def __init__(self, model, lr, wd, train_noise, lr_warmup_steps=0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.lr_warmup_steps = lr_warmup_steps
        self.wd = wd
        self.train_noise = train_noise

        self.criterion = ch.nn.MSELoss()

        self.train_cos_sim = CosineSimilarity(reduction='mean')
        self.train_cos_sim_epoch = CosineSimilarity(reduction='mean')
        self.val_cos_sim_epoch = CosineSimilarity(reduction='mean')



    def training_step(self, batch, batch_idx, mode='train'):
        x, y = batch
        if self.train_noise != 0:
            # x = x + ch.randn_like(x) * self.train_noise
            x = x + ch.randn(x.shape[0], x.shape[1], device=x.device)[:, :, None].repeat(1, 1, x.shape[2]) * self.train_noise

        # y_input is last step of x
        y_input = x[:, -1, :][:, None, :]

        predictions = self.model(x, y_input)

        loss = self.criterion(predictions, y)

        self.train_cos_sim(predictions, y)
        self.train_cos_sim_epoch(predictions, y)

        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cos_sim", self.train_cos_sim_epoch, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx, mode='train'):
        print('Validation step', batch_idx)
        x, y = batch
        y_input = x[:, -1, :][:, None, :]
        predictions = self.model(x, y_input)

        loss = self.criterion(predictions, y)

        self.val_cos_sim_epoch(predictions, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cos_sim", self.val_cos_sim_epoch, on_step=False, on_epoch=True, prog_bar=True)

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

