import tensorflow as tf
import json
import os


def initiate_log_files(output_dir, param_path):
    #  load parameters from json file
    param_dict = json.load(open(param_path))

    # create output files
    AUCfile_path = os.path.join(output_dir, 'AUC_history_gridsearch.tsv')
    if not os.path.exists(AUCfile_path):
        AUCfile = open(AUCfile_path, 'w')

        AUCheader = ['auc_train', 'auc_val', 'matthews_train', 'matthews_val', 'cv_num'] + list(param_dict.keys())
        AUCfile.write('\t'.join(AUCheader) + '\n')

        # close files
        AUCfile.close()

    CVfile_path = os.path.join(output_dir, 'CV_history_gridsearch.tsv')
    if not os.path.exists(CVfile_path):
        CVfile = open(CVfile_path, 'w')

        CVheader = ['max epochs', 'accuracy', 'loss', 'matthews', 'precision', 'recall', 'val_accuracy', 'val_loss', 'val_matthews',
        'val_precision', 'val_recall', 'cv_num'] + list(param_dict.keys())

        CVfile.write('epoch\t' + '\t'.join(CVheader) + '\n')

        CVfile.close()

    # open log files
    error_log_path = os.path.join(output_dir,'error.log')
    if not os.path.exists(error_log_path):
        error_log = open(error_log_path, 'w')
        errorHeader = ['error', 'args']
        error_log.write('\t'.join(errorHeader) + '\n')
        error_log.close()

    progress_log_path = os.path.join(output_dir,'progress.log')
    if not os.path.exists(progress_log_path):
        progress_log = open(progress_log_path, 'w')
        progressHeader = ['completed', 'time_elapsed']
        progress_log.write('\t'.join(progressHeader) + '\n')
        progress_log.close()

    return 0

class WarmUpScheduler(tf.keras.callbacks.Callback):
    def __init__(self, final_lr, warmup_learning_rate=0.0, warmup_steps=0,
                 verbose=0):
        """Constructor for warmup learning rate scheduler.
        Args:
            learning_rate_base: base learning rate.
            warmup_learning_rate: Initial learning rate for warm up. (default:
                0.0)
            warmup_steps: Number of warmup steps. (default: 0)
            verbose: 0 -> quiet, 1 -> update messages. (default: {0})
        """

        super().__init__()
        self.final_lr = final_lr
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.verbose = verbose

        # Count global steps from 1, allows us to set warmup_steps to zero to
        # skip warmup.
        self.global_step = 1
        self._increase_per_step = \
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.global_step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.global_step <= self.warmup_steps:
            increase = \
                (self.final_lr - self.warmup_learning_rate) / self.warmup_steps
            new_lr = self.warmup_learning_rate + (increase * self.global_step)
            tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
                print(
                    f'Warmup - learning rate: '
                    f'{new_lr:.6f}/{self.final_lr:.6f}',
                    end=''
                )