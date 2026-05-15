import os
from functools import partial
from datetime import datetime
import optuna
import torch as ch
from os import path
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import json

from prediction.outcome_prediction.Transformer.architecture import OPSUM_encoder_decoder
from prediction.outcome_prediction.Transformer.lightning_wrapper import LitEncoderDecoderModel
from prediction.outcome_prediction.Transformer.utils.callbacks import MyEarlyStopping
from prediction.outcome_prediction.Transformer.utils.utils import DictLogger
from prediction.short_term_outcome_prediction.timeseries_decomposition import BucketBatchSampler, \
    prepare_subsequence_dataset
from prediction.utils.utils import ensure_dir


# data_splits_path = '/mnt/data1/klug/datasets/opsum/short_term_outcomes/gsu_Extraction_20220815_prepro_08062024_083500/early_neurological_deterioration_train_data_splits/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
data_splits_path = '/home/klug/temp/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
output_folder = '/home/klug/temp'

# data_splits_path = '/Users/jk1/Downloads/train_data_splits_early_neurological_deterioration_ts0.8_rs42_ns5.pth'
# output_folder = '/Users/jk1/Downloads'

timestamp = 'test'
use_gpu = True

splits = ch.load(path.join(data_splits_path))
ds = [prepare_subsequence_dataset(x, use_gpu=use_gpu, use_target_timeseries=True,
                                            target_timeseries_length=1) for x in splits]

batch_size = 416
num_layers = 6
num_decoder_layers = 6
model_dim = 1024
train_noise = 1e-5
wd = 1e-5
ff_factor = 2
ff_dim = ff_factor * model_dim
dropout = 0.1
num_heads = 16
# pos_encode_factor = 1
lr = 0.0001
n_lr_warm_up_steps = 0
grad_clip = 1e-3
early_stopping_step_limit = 10
max_epochs = 1

accelerator = 'gpu' if use_gpu else 'cpu'

val_scores = []
best_epochs = []

for i, (train_dataset, val_dataset) in enumerate(ds):
    checkpoint_dir = os.path.join(output_folder, f'checkpoints_short_opsum_transformer_{timestamp}_cv_{i}')
    ensure_dir(checkpoint_dir)

    input_dim = train_dataset[0][0].shape[-1]

    model = OPSUM_encoder_decoder(input_dim=input_dim, num_layers=num_layers, num_decoder_layers=num_decoder_layers,
                                    model_dim=model_dim, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout,
                                    pos_encode_factor=0.1, n_tokens=1, max_dim=5000, layer_norm_eps=1e-05)

    train_bucket_sampler = BucketBatchSampler(train_dataset.idx_to_len_map, batch_size)
    train_loader = DataLoader(train_dataset, batch_sampler=train_bucket_sampler,
                              # shuffling is done in the bucket sampler
                              shuffle=False, drop_last=False)

    val_bucket_sampler = BucketBatchSampler(val_dataset.idx_to_len_map, 1024)
    val_loader = DataLoader(val_dataset, batch_sampler=val_bucket_sampler)
    logger = DictLogger(0)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_cos_sim",
        mode="max",
        dirpath=checkpoint_dir,
        filename="short_opsum_dec_transformer_{epoch:02d}_{val_cos_sim:.4f}",
    )

    module = LitEncoderDecoderModel(model, lr, wd, train_noise, lr_warmup_steps=n_lr_warm_up_steps)
    trainer = pl.Trainer(accelerator=accelerator, devices=1, max_epochs=max_epochs,
                         logger=[logger, pl.loggers.TensorBoardLogger(output_folder, name=f'{timestamp}_cv_{i}')],
                         log_every_n_steps=50, enable_checkpointing=False,
                         callbacks=[MyEarlyStopping(step_limit=early_stopping_step_limit, metric='val_cos_sim'),
                                    # checkpoint_callback],
                                    ],
                         gradient_clip_val=grad_clip)
    trainer.fit(model=module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    # trainer.validate(model=module, dataloaders=val_loader)

    if logger.metrics == []:
        print(i, logger.metrics)
        continue
    best_val_score = np.max([x['val_cos_sim'] for x in logger.metrics if 'val_cos_sim' in x])
    best_epoch = np.argmax([x['val_cos_sim'] for x in logger.metrics if 'val_cos_sim' in x])
    val_scores.append(best_val_score)
    best_epochs.append(best_epoch)

    print(f"CV {i} Best Val Score: {best_val_score} at epoch {best_epoch}")

d = dict()
d['median_val_scores'] = float(np.median(val_scores))
d['median_best_epochs'] = float(np.median(best_epochs))
d['timestamp'] = timestamp
d['best_cv_fold'] = int(np.argmax(val_scores))
d['worst_cv_fold_val_score'] = float(np.min(val_scores))
d['split_file'] = data_splits_path
text = json.dumps(d)
text += '\n'
dest = path.join(output_folder, f'{os.path.basename(output_folder)}_gridsearch.jsonl')
with open(dest, 'a') as handle:
    handle.write(text)
print("WRITTEN in ", dest)