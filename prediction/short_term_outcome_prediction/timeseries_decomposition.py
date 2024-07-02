import pandas as pd
import numpy as np
import torch as ch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Sampler, Dataset
from torch import tensor
from collections import OrderedDict
from random import shuffle


def decompose_and_label_timeseries(timeseries: np.ndarray, y_df: pd.DataFrame, target_time_to_outcome: int = 6):
    """
    Decompose the timeseries data into individual samples (every sample is a subtimeseries) and associate each sample with a label.
    Args:
        timeseries (np.array): array of shape (num_samples, num_features, num_timesteps)
        y_df (pd.DataFrame): dataframe with case_admission_id and relative_sample_date_hourly_cat (start of the target event)
        delta (int): number of timesteps to predict in the future

    Returns:
        map (list): list of tuples (cid_idx, ts) where idx in list is the index of the sample in the flattened targets array, cid_idx is the idx of case admission id, and ts is the last timestep for this idx
        flat_labels (list): list of labels for every subsequence
    """

    # create index mapping (list of (cid, ts) in which the index in the list is the index of the sample in the flattened targets array)
    map = []
    # labels for every sub sequence
    flat_labels = []
    # maximum number of timesteps (for most patients is max of relative_sample_date_hourly_cat, but for some patients it until the occurrence of the event)
    overall_max_ts = timeseries.shape[1]
    for idx, cid in enumerate(timeseries[:, 0, 0, 0]):
        if cid in y_df.case_admission_id.values:
            max_ts = y_df[y_df.case_admission_id == cid].relative_sample_date_hourly_cat.values[0]
        else:
            max_ts = overall_max_ts
        for ts in range(int(max_ts)):
            # store idx of cid and idx of ts
            map.append((idx, ts))
            if cid in y_df.case_admission_id.values and ts + target_time_to_outcome >= max_ts:
                flat_labels.append(1)
            else:
                flat_labels.append(0)

    return map, flat_labels


class StrokeUnitBucketDataset(Dataset):

    def __init__(self, inputs, targets, idx_map):
        """
        Every sample is a sequence of timesteps with an associated label/target.
        - The index of each sample is an index in the flattened targets array
        - To retrieve the inputs for a sample, we need to know the case admission id and the last timestep for this sample (provided in idx_map)

        Args:
            inputs (np.array): array of shape (num_samples, num_features, num_timesteps)
            targets (np.array): array of shape (with targets for all idx) (flattened)
            idx_map (list): list of tuples (cid_idx, ts) where idx in list is the index of the sample in targets, cid_idx is the idx of case admission id, and ts is the last timestep for this idx
                - This is necessary to retrieve the inputs for this idx (as every patient has multiple timesteps)
        """
        self.inputs = inputs
        self.targets = targets
        self.idx_map = idx_map
        # create a mapping from overall idx to length of sequence
        # idx_to_len_map (list): list of tuples (idx, length) where idx is the index of the sample in the flattened targets array and length is the length of the sequence
        self.idx_to_len_map = [(idx, map_i[1] + 1) for idx, map_i in enumerate(self.idx_map)]

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, index):
        cid_idx = self.idx_map[index][0]
        last_ts = self.idx_map[index][1]
        if self.targets is None:
            return self.inputs[cid_idx, 0: last_ts + 1]
        else:
            return self.inputs[cid_idx, 0: last_ts + 1], self.targets[index]


class BucketBatchSampler(Sampler):
    # Ref: https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/13
    # want inputs to be an array
    def __init__(self, idx_to_len_map, batch_size):
        self.batch_size = batch_size
        self.idx_to_len_map = idx_to_len_map # list of tuples (idx, length)
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.idx_to_len_map)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.idx_to_len_map:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        batch_list = []
        for length, indices in batch_map.items():
            for group in [indices[i:(i + self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                batch_list.append(group)
        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i


def prepare_subsequence_dataset(scenario, rescale=True, target_time_to_outcome=6, use_gpu=True):
    X_train, X_val, y_train, y_val = scenario

    train_map, train_flat_labels = decompose_and_label_timeseries(X_train, y_train, target_time_to_outcome=target_time_to_outcome)
    val_map, val_flat_labels = decompose_and_label_timeseries(X_val, y_val, target_time_to_outcome=target_time_to_outcome)

    X_train = X_train[:, :, :, -1].astype('float32')
    X_val = X_val[:, :, :, -1].astype('float32')

    scaler = StandardScaler()
    if rescale:
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)


    if use_gpu:
        train_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_train).cuda(), tensor(train_flat_labels).cuda(), train_map)
        val_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_val).cuda(), tensor(val_flat_labels).cuda(), val_map)
    else:
        train_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_train), tensor(train_flat_labels), train_map)
        val_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_val), tensor(val_flat_labels), val_map)

    return train_dataset, val_dataset



