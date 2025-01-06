import pandas as pd
import numpy as np
import torch
import torch as ch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Sampler, Dataset
from torch import tensor
from collections import OrderedDict
from random import shuffle

from prediction.utils.utils import aggregate_features_over_time


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


def decompose_and_label_timeseries_time_to_event(timeseries: np.ndarray, y_df: pd.DataFrame):
    """
    Decompose the timeseries data into individual samples (every sample is a subtimeseries) and associate each sample with a label.
    The label is given by a tuple and represents the time to event or censoring (time to event|censoring, event y|n)
    Args:
        timeseries (np.array): array of shape (num_samples, num_features, num_timesteps)
        y_df (pd.DataFrame): dataframe with case_admission_id and relative_sample_date_hourly_cat (start of the target event)

    Returns:
        map (list): list of tuples (cid_idx, ts) where idx in list is the index of the sample in the flattened targets array, cid_idx is the idx of case admission id, and ts is the last timestep for this idx
        flat_labels (list): list of labels for every subsequence [label: (time to event|censoring, event y|n)]

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
            time_to_event = max_ts - ts
            if cid in y_df.case_admission_id.values:
                # flat_labels.append((time_to_event, 1))
                flat_labels.append(time_to_event)
            else:
                # data censored (no event)
                # flat_labels.append((time_to_event, 0))
                flat_labels.append(max_ts)

    return map, flat_labels


def decompose_timeseries(timeseries: np.ndarray, target_timeseries_length: int = 1):
    """
    Decompose the timeseries data into individual samples (every sample is a subtimeseries)
    Args:
        timeseries (np.array): array of shape (num_samples, num_features, num_timesteps)

    Returns:
        map (list): list of tuples (cid_idx, ts) where idx in list is the index of the sample in the flattened targets array, cid_idx is the idx of case admission id, and ts is the last timestep for this idx
    """

    # create index mapping (list of (cid, ts) in which the index in the list is the index of the sample in the flattened targets array)
    map = []
    # maximum number of timesteps (max of relative_sample_date_hourly_cat)
    overall_max_ts = timeseries.shape[1] - target_timeseries_length
    for idx, cid in enumerate(timeseries[:, 0, 0, 0]):
        for ts in range(int(overall_max_ts)):
            # store idx of cid and idx of ts
            map.append((idx, ts))

    return map


def aggregate_and_label_timeseries(timeseries, y_df, target_time_to_outcome=6, mask_after_first_positive=True):
    all_subj_labels = []
    all_subj_data = []
    n_timepoints = timeseries.shape[1]
    for idx, cid in enumerate(timeseries[:, 0, 0, 0]):
        x_data = timeseries[None, idx, :, :, -1].astype('float32')
        if cid not in y_df.case_admission_id.values:
            labels = np.zeros(n_timepoints)
        else:
            event_ts = int(y_df[y_df.case_admission_id == cid].relative_sample_date_hourly_cat.values[0])
            # let labels be 0 until 6 ts before the event then 1 until the end then 0
            n_pos_start = max(0, event_ts - target_time_to_outcome)
            n_pos_end = event_ts
            labels = np.concatenate(
                (np.zeros(n_pos_start), np.ones(n_pos_end - n_pos_start), np.zeros(n_timepoints - n_pos_end)))

            if mask_after_first_positive:
                labels = labels[:n_pos_start + 1]
                x_data = x_data[:, :n_pos_start + 1, :]
        x_data, _ = aggregate_features_over_time(x_data, np.array([None]), moving_average=False)
        all_subj_labels.append(labels)
        all_subj_data.append(x_data)

    return all_subj_data, all_subj_labels


class StrokeUnitBucketDataset(Dataset):

    def __init__(self, inputs, targets, idx_map,
                 return_target_timeseries=False, target_timeseries_indices=None, target_timeseries_length=1):
        """
        Every sample is a sequence of timesteps with an associated label/target.
        - The index of each sample is an index in the flattened targets array
        - To retrieve the inputs for a sample, we need to know the case admission id and the last timestep for this sample (provided in idx_map)

        Args:
            inputs (np.array): array of shape (num_samples, num_features, num_timesteps)
            targets (np.array): array of shape (with targets for all idx) (flattened)
            idx_map (list): list of tuples (cid_idx, ts) where idx in list is the index of the sample in targets, cid_idx is the idx of case admission id, and ts is the last timestep for this idx
                - This is necessary to retrieve the inputs for this idx (as every patient has multiple timesteps)
            return_target_timeseries (bool): whether to return the target timeseries or not (on top of the inputs and targets)
            target_timeseries_indices (list): list of indices of the target timeseries to return (if None, return all features)
            target_timeseries_length (int): length of the target timeseries to return (1 by default, only predict the next timestep)
        """
        self.inputs = inputs
        self.targets = targets
        self.idx_map = idx_map

        # only for encoder/decoder model
        self.return_target_timeseries = return_target_timeseries
        self.target_timeseries_indices = target_timeseries_indices
        self.target_timeseries_length = target_timeseries_length

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
            if self.return_target_timeseries:
                if self.target_timeseries_indices is not None:
                    # return inputs, and the target features of the rest of the timeseries (becomes the target timeseries)
                    return (self.inputs[cid_idx, 0: last_ts + 1],
                            self.inputs[cid_idx, last_ts + 1: last_ts + 1 + self.target_timeseries_length,
                                        self.target_timeseries_indices])
                else:
                    # return inputs, and the rest of the timeseries with all features (becomes the target timeseries)
                    return (self.inputs[cid_idx, 0: last_ts + 1],
                            self.inputs[cid_idx, last_ts + 1: last_ts + 1 + self.target_timeseries_length])
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


def prepare_subsequence_dataset(scenario, rescale=True, target_time_to_outcome=6, use_gpu=True,
                                use_time_to_event=False,
                                use_target_timeseries=False, target_timeseries_indices=None,
                                target_timeseries_length=1):
    """
    Prepares the dataset for the transformer model.

    Args:
        scenario (tuple): tuple of (X_train, X_val, y_train, y_val)
        rescale (bool): whether to rescale the data or not
        target_time_to_outcome (int): number of timesteps to predict in the future
        use_gpu (bool): whether to use GPU or not
        use_target_timeseries (bool): whether to return the target timeseries or not (only relevant for the encoder/decoder model)
    """
    X_train, X_val, y_train, y_val = scenario

    if not use_target_timeseries and not use_time_to_event:
        train_map, train_flat_labels = decompose_and_label_timeseries(X_train, y_train, target_time_to_outcome=target_time_to_outcome)
        val_map, val_flat_labels = decompose_and_label_timeseries(X_val, y_val, target_time_to_outcome=target_time_to_outcome)
    elif use_time_to_event:
        train_map, train_flat_labels = decompose_and_label_timeseries_time_to_event(X_train, y_train)
        val_map, val_flat_labels = decompose_and_label_timeseries_time_to_event(X_val, y_val)
    elif use_target_timeseries:
        # if we want to return the target timeseries, we need to decompose the timeseries differently (whole timeseries for each sample)
        train_map = decompose_timeseries(X_train, target_timeseries_length)
        val_map = decompose_timeseries(X_val, target_timeseries_length)
        # labels are not used (as timeseries is the target)
        train_flat_labels = torch.empty(len(train_map))
        val_flat_labels = torch.empty(len(val_map))

    X_train = X_train[:, :, :, -1].astype('float32')
    X_val = X_val[:, :, :, -1].astype('float32')

    scaler = StandardScaler()
    if rescale:
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_train.shape[-1])).reshape(X_val.shape)


    if use_gpu:
        train_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_train).cuda(), tensor(train_flat_labels).cuda(),
                                                train_map, return_target_timeseries=use_target_timeseries,
                                                target_timeseries_indices=target_timeseries_indices,
                                                target_timeseries_length=target_timeseries_length)
        val_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_val).cuda(), tensor(val_flat_labels).cuda(), val_map,
                                                return_target_timeseries=use_target_timeseries,
                                                target_timeseries_indices=target_timeseries_indices,
                                                target_timeseries_length=target_timeseries_length)
    else:
        train_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_train), tensor(train_flat_labels), train_map,
                                                return_target_timeseries=use_target_timeseries,
                                                target_timeseries_indices=target_timeseries_indices,
                                                target_timeseries_length=target_timeseries_length)
        val_dataset = StrokeUnitBucketDataset(ch.from_numpy(X_val), tensor(val_flat_labels), val_map,
                                                return_target_timeseries=use_target_timeseries,
                                                target_timeseries_indices=target_timeseries_indices,
                                                target_timeseries_length=target_timeseries_length)

    return train_dataset, val_dataset


def prepare_aggregate_dataset(scenario, rescale=True, target_time_to_outcome=6, mask_after_first_positive=True):
    """
    Prepares the dataset as an aggregate dataset (one sample per timepoint) and returns the train and validation sets.
    (only used for XGB model)

    Args:
        scenario (tuple): tuple of (X_train, X_val, y_train, y_val)
        rescale (bool): whether to rescale the data or not
        target_time_to_outcome (int): number of timesteps to predict in the future
    """
    X_train, X_val, y_train, y_val = scenario

    train_data, train_labels = aggregate_and_label_timeseries(X_train, y_train, target_time_to_outcome,
                                                              mask_after_first_positive)
    val_data, val_labels = aggregate_and_label_timeseries(X_val, y_val, target_time_to_outcome,
                                                          mask_after_first_positive)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    val_data = np.concatenate(val_data)
    val_labels = np.concatenate(val_labels)

    scaler = StandardScaler()
    if rescale:
        train_data = scaler.fit_transform(train_data)
        val_data = scaler.transform(val_data)

    return train_data, val_data, train_labels, val_labels





