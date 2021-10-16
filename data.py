# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the dlrm benchmark
# The inpts and outputs are generated according to the following three option(s)
# public data set
# Criteo Kaggle Display Advertising Challenge Dataset
# https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset


import os

import numpy as np
import torch

import data_utils


# Kaggle Display Advertising Challenge Dataset
# dataset (str): name of dataset (Kaggle or Terabyte)
# randomize (str): determines randomization scheme
#            "none": no randomization
#            "day": randomizes each day"s data (only works if split = True)
#            "total": randomizes total dataset
# split (bool) : to split into train, test, validation data-sets
class CriteoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_ind_range,
        sub_sample_rate,
        randomize,
        split="train",
        raw_path="",
        dataset_multiprocessing=False,
    ):
        days = 7
        out_file = "kaggleAdDisplayChallenge_processed"

        self.max_ind_range = max_ind_range

        # split the datafile into path and filename
        lstr = raw_path.split("/")
        self.d_path = "/".join(lstr[0:-1]) + "/"
        self.d_file = lstr[-1].split(".")[0]
        self.npzfile = self.d_path + ((self.d_file + "_day"))
        self.trafile = self.d_path + ((self.d_file + "_fea"))

        
        # check if data is already processed
        if os.path.isfile(self.d_path + out_file + ".npz"):
            print("Reading processed data")
            file = self.d_path + out_file + ".npz"
        else:
            # pre-process data if needed
            print("Reading raw data=%s" % (str(raw_path)))
            print("Pre-processing data")
            file = data_utils.getCriteoAdData(
                raw_path,
                out_file,
                max_ind_range,
                sub_sample_rate,
                days,
                split,
                randomize,
                dataset_multiprocessing,
            )
        
        # get a number of samples per day
        total_file = self.d_path + self.d_file + "_day_count.npz"
        with np.load(total_file) as data:
            total_per_file = data["total_per_file"]
        # compute offsets per file
        self.offset_per_file = np.array([0] + [x for x in total_per_file])
        for i in range(days):
            self.offset_per_file[i + 1] += self.offset_per_file[i]

        # load and preprocess data
        with np.load(file) as data:
            X_int = data["X_int"]  # continuous  feature
            X_cat = data["X_cat"]  # categorical feature
            y = data["y"]  # target
            self.counts = data["counts"]
        self.m_den = X_int.shape[1]  # den_fea
        self.n_emb = len(self.counts)
        print("Sparse fea = %d, Dense fea = %d" % (self.n_emb, self.m_den))

        # create reordering
        indices = np.arange(len(y))

        if split == "none":
            # randomize all data
            if randomize == "total":
                indices = np.random.permutation(indices)
                print("Randomized indices...")

            X_int[indices] = X_int
            X_cat[indices] = X_cat
            y[indices] = y

        else:
            indices = np.array_split(indices, self.offset_per_file[1:-1])

            # randomize train data (per day)
            if randomize == "day":  # or randomize == "total":
                for i in range(len(indices) - 1):
                    indices[i] = np.random.permutation(indices[i])
                print("Randomized indices per day ...")

            train_indices = np.concatenate(indices[:-1])
            test_indices = indices[-1]
            test_indices, val_indices = np.array_split(test_indices, 2)

            print("Defined %s indices..." % (split))

            # randomize train data (across days)
            if randomize == "total":
                train_indices = np.random.permutation(train_indices)
                print("Randomized indices across days ...")

            # create training, validation, and test sets
            if split == "train":
                self.X_int = [X_int[i] for i in train_indices]
                self.X_cat = [X_cat[i] for i in train_indices]
                self.y = [y[i] for i in train_indices]
            elif split == "val":
                self.X_int = [X_int[i] for i in val_indices]
                self.X_cat = [X_cat[i] for i in val_indices]
                self.y = [y[i] for i in val_indices]
            elif split == "test":
                self.X_int = [X_int[i] for i in test_indices]
                self.X_cat = [X_cat[i] for i in test_indices]
                self.y = [y[i] for i in test_indices]

        print("Split data according to indices...")

    def __getitem__(self, index):

        if isinstance(index, slice):
            return [
                self[idx]
                for idx in range(
                    index.start or 0, index.stop or len(self), index.step or 1
                )
            ]

        i = index

        if self.max_ind_range > 0:
            return self.X_int[i], self.X_cat[i] % self.max_ind_range, self.y[i]
        else:
            return self.X_int[i], self.X_cat[i], self.y[i]

    def __len__(self):
        return len(self.y)


def collate_wrapper_criteo_offset(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = [X_cat[:, i] for i in range(featureCnt)]
    lS_o = [torch.tensor(range(batchSize)) for _ in range(featureCnt)]
    return X_int, torch.stack(lS_o), torch.stack(lS_i), T


# Conversion from offset to length
def offset_to_length_converter(lS_o, lS_i):
    def diff(tensor):
        return tensor[1:] - tensor[:-1]

    return torch.stack(
        [
            diff(torch.cat((S_o, torch.tensor(lS_i[ind].shape))).int())
            for ind, S_o in enumerate(lS_o)
        ]
    )


def collate_wrapper_criteo_length(list_of_tuples):
    # where each tuple is (X_int, X_cat, y)
    transposed_data = list(zip(*list_of_tuples))
    X_int = torch.log(torch.tensor(transposed_data[0], dtype=torch.float) + 1)
    X_cat = torch.tensor(transposed_data[1], dtype=torch.long)
    T = torch.tensor(transposed_data[2], dtype=torch.float32).view(-1, 1)

    batchSize = X_cat.shape[0]
    featureCnt = X_cat.shape[1]

    lS_i = torch.stack([X_cat[:, i] for i in range(featureCnt)])
    lS_o = torch.stack([torch.tensor(range(batchSize)) for _ in range(featureCnt)])

    lS_l = offset_to_length_converter(lS_o, lS_i)

    return X_int, lS_l, lS_i, T


def make_criteo_data_and_loaders(args, offset_to_length_converter=False):
    train_data = CriteoDataset(
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "train",
        args.raw_data_file,
        args.dataset_multiprocessing,
    )

    test_data = CriteoDataset(
        args.max_ind_range,
        args.data_sub_sample_rate,
        args.data_randomize,
        "test",
        args.raw_data_file,
        args.dataset_multiprocessing,
    )

    collate_wrapper_criteo = collate_wrapper_criteo_offset
    if offset_to_length_converter:
        collate_wrapper_criteo = collate_wrapper_criteo_length

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.mini_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=args.test_mini_batch_size,
        shuffle=False,
        num_workers=args.test_num_workers,
        collate_fn=collate_wrapper_criteo,
        pin_memory=False,
        drop_last=False,  # True
    )

    return train_data, train_loader, test_data, test_loader
