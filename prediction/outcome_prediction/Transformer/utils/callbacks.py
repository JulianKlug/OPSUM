from pytorch_lightning.callbacks.callback import Callback


class MyEarlyStopping(Callback):

    best_so_far = 0
    last_improvement = 0

    def __init__(self, step_limit=10, metric='val_auroc'):
        super().__init__()
        self.step_limit = step_limit
        self.metric = metric

    def on_validation_end(self, trainer, pl_module):
        logs = trainer.callback_metrics
        val_metric = logs[self.metric].item()

        if val_metric > self.best_so_far:
            self.last_improvement = 0
        else:
            self.last_improvement += 1

        print(self.last_improvement)
        trainer.should_stop = val_metric < 0.75 * self.best_so_far or self.last_improvement > self.step_limit or \
                    (trainer.current_epoch > 10 and val_metric < 0.55)

        self.best_so_far = max(val_metric, self.best_so_far)
