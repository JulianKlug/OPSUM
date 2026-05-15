from pytorch_lightning.callbacks.callback import Callback


class MyEarlyStopping(Callback):

    last_improvement = 0

    def __init__(self, step_limit=10, metric='val_auroc', direction='max'):
        super().__init__()
        self.step_limit = step_limit
        self.metric = metric
        self.direction = direction

        if self.direction == 'max':
                   self.best_so_far = 0
        elif self.direction == 'min':
                   self.best_so_far = 1e6
        else:
            raise ValueError('direction must be "max" or "min"')

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_metric = logs[self.metric].item()

        if self.direction == 'max':
            if val_metric > self.best_so_far:
                self.last_improvement = 0
            else:
                self.last_improvement += 1

            print(self.last_improvement)

            if self.metric == 'val_auroc':
                trainer.should_stop = val_metric < 0.75 * self.best_so_far or self.last_improvement > self.step_limit or \
                            (trainer.current_epoch > 10 and val_metric < 0.55)
            else:
                trainer.should_stop = self.last_improvement > self.step_limit

            self.best_so_far = max(val_metric, self.best_so_far)

        if self.direction == 'min':
            if val_metric < self.best_so_far:
                self.last_improvement = 0
            else:
                self.last_improvement += 1

            trainer.should_stop = self.last_improvement > self.step_limit

            self.best_so_far = min(val_metric, self.best_so_far)
