from pytorch_lightning.callbacks.callback import Callback


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
        trainer.should_stop = val_auroc < 0.75 * self.best_so_far or self.last_improvement > 10 or \
                    (trainer.current_epoch > 10 and val_auroc < 0.55)

        self.best_so_far = max(val_auroc, self.best_so_far)
