# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the DLRM benchmark
#
# Utility function(s) to download and pre-process public data sets
# - Criteo Kaggle Display Advertising Challenge Dataset
# https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset

import os
import sys
from multiprocessing import Manager, Process

import numpy as np


def processCriteoAdData(d_path, d_file, npzfile, i, convertDicts, pre_comp_counts):
    # Process Kaggle Display Advertising Challenge or Terabyte Dataset
    # by converting unicode strings in X_cat to integers and
    # converting negative integer values in X_int.
    #
    # Loads data in the form "{kaggle|terabyte}_day_i.npz" where i is the day.
    #
    # Inputs:
    #   d_path (str): path for {kaggle|terabyte}_day_i.npz files
    #   i (int): splits in the dataset (typically 0 to 7 or 0 to 24)

    # process data if not all files exist
    filename_i = npzfile + "_{0}_processed.npz".format(i)

    if os.path.exists(filename_i):
        print("Using existing " + filename_i, end="\n")
    else:
        print("Not existing " + filename_i)
        with np.load(npzfile + "_{0}.npz".format(i)) as data:

            # Approach 2a: using pre-computed dictionaries
            X_cat_t = np.zeros(data["X_cat_t"].shape)
            for j in range(26):
                for k, x in enumerate(data["X_cat_t"][j, :]):
                    X_cat_t[j, k] = convertDicts[j][x]
            # continuous features
            X_int = data["X_int"]
            X_int[X_int < 0] = 0
            # targets
            y = data["y"]

        np.savez_compressed(
            filename_i,
            # X_cat = X_cat,
            X_cat=np.transpose(X_cat_t),  # transpose of the data
            X_int=X_int,
            y=y,
        )
        print("Processed " + filename_i, end="\n")
    # sanity check (applicable only if counts have been pre-computed & are re-computed)
    # for j in range(26):
    #    if pre_comp_counts[j] != counts[j]:
    #        sys.exit("ERROR: Sanity check on counts has failed")
    # print("\nSanity check on counts passed")

    return


def concatCriteoAdData(
        d_path,
        d_file,
        npzfile,
        trafile,
        days,
        data_split,
        randomize,
        total_per_file,
        total_count,
        o_filename
):
    # Concatenates different days and saves the result.
    #
    # Inputs:
    #   days (int): total number of days in the dataset (typically 7 or 24)
    #   d_path (str): path for {kaggle|terabyte}_day_i.npz files
    #   o_filename (str): output file name
    #
    # Output:
    #   o_file (str): output file path

    print("Concatenating multiple days into %s.npz file" % str(d_path + o_filename))

    # load and concatenate data
    for i in range(days):
        filename_i = npzfile + "_{0}_processed.npz".format(i)
        with np.load(filename_i) as data:
            if i == 0:
                X_cat = data["X_cat"]
                X_int = data["X_int"]
                y = data["y"]
            else:
                X_cat = np.concatenate((X_cat, data["X_cat"]))
                X_int = np.concatenate((X_int, data["X_int"]))
                y = np.concatenate((y, data["y"]))
        print("Loaded day:", i, "y = 1:", len(y[y == 1]), "y = 0:", len(y[y == 0]))

    with np.load(d_path + d_file + "_fea_count.npz") as data:
        counts = data["counts"]
    print("Loaded counts!")

    np.savez_compressed(
        d_path + o_filename + ".npz",
        X_cat=X_cat,
        X_int=X_int,
        y=y,
        counts=counts,
    )

    return d_path + o_filename + ".npz"



def getCriteoAdData(
        datafile,
        o_filename,
        max_ind_range=-1,
        sub_sample_rate=0.0,
        days=7,
        data_split='train',
        randomize='total',
        dataset_multiprocessing=False,
):
    # Passes through entire dataset and defines dictionaries for categorical
    # features and determines the number of total categories.
    #
    # Inputs:
    #    datafile : path to downloaded raw data file
    #    o_filename (str): saves results under o_filename if filename is not ""
    #
    # Output:
    #   o_file (str): output file path

    #split the datafile into path and filename
    lstr = datafile.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"
    d_file = lstr[-1].split(".")[0]
    npzfile = d_path + ((d_file + "_day"))
    trafile = d_path + ((d_file + "_fea"))

    # count number of datapoints in training set
    total_file = d_path + d_file + "_day_count.npz"
    if os.path.exists(total_file):
        with np.load(total_file) as data:
            total_per_file = list(data["total_per_file"])
        total_count = np.sum(total_per_file)
        print("Skipping counts per file (already exist)")
    else:
        total_count = 0
        total_per_file = []
        # WARNING: The raw data consists of a single train.txt file
        # Each line in the file is a sample, consisting of 13 continuous and
        # 26 categorical features (an extra space indicates that feature is
        # missing and will be interpreted as 0).
        if os.path.exists(datafile):
            print("Reading data from path=%s" % (datafile))
            with open(str(datafile)) as f:
                for _ in f:
                    total_count += 1
            total_per_file.append(total_count)
            # reset total per file due to split
            num_data_per_split, extras = divmod(total_count, days)
            total_per_file = [num_data_per_split] * days
            for j in range(extras):
                total_per_file[j] += 1
            # split into days (simplifies code later on)
            file_id = 0
            boundary = total_per_file[file_id]
            nf = open(npzfile + "_" + str(file_id), "w")
            with open(str(datafile)) as f:
                for j, line in enumerate(f):
                    if j == boundary:
                        nf.close()
                        file_id += 1
                        nf = open(npzfile + "_" + str(file_id), "w")
                        boundary += total_per_file[file_id]
                    nf.write(line)
            nf.close()
        else:
            sys.exit("ERROR: Criteo Kaggle Display Ad Challenge Dataset path is invalid; please download from https://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset")

    # process a file worth of data and reinitialize data
    # note that a file main contain a single or multiple splits
    def process_one_file(
            datfile,
            npzfile,
            split,
            num_data_in_split,
            dataset_multiprocessing,
            convertDictsDay=None,
            resultDay=None
    ):
        if dataset_multiprocessing:
            convertDicts_day = [{} for _ in range(26)]

        with open(str(datfile)) as f:
            y = np.zeros(num_data_in_split, dtype="i4")  # 4 byte int
            X_int = np.zeros((num_data_in_split, 13), dtype="i4")  # 4 byte int
            X_cat = np.zeros((num_data_in_split, 26), dtype="i4")  # 4 byte int
            if sub_sample_rate == 0.0:
                rand_u = 1.0
            else:
                rand_u = np.random.uniform(low=0.0, high=1.0, size=num_data_in_split)

            i = 0
            percent = 0
            for k, line in enumerate(f):
                # process a line (data point)
                line = line.split('\t')
                # set missing values to zero
                for j in range(len(line)):
                    if (line[j] == '') or (line[j] == '\n'):
                        line[j] = '0'
                # sub-sample data by dropping zero targets, if needed
                target = np.int32(line[0])
                if target == 0 and \
                   (rand_u if sub_sample_rate == 0.0 else rand_u[k]) < sub_sample_rate:
                    continue

                y[i] = target
                X_int[i] = np.array(line[1:14], dtype=np.int32)
                if max_ind_range > 0:
                    X_cat[i] = np.array(
                        list(map(lambda x: int(x, 16) % max_ind_range, line[14:])),
                        dtype=np.int32
                    )
                else:
                    X_cat[i] = np.array(
                        list(map(lambda x: int(x, 16), line[14:])),
                        dtype=np.int32
                    )

                # count uniques
                if dataset_multiprocessing:
                    for j in range(26):
                        convertDicts_day[j][X_cat[i][j]] = 1
                    # debug prints
                    if float(i)/num_data_in_split*100 > percent+1:
                        percent = int(float(i)/num_data_in_split*100)
                        print(
                            "Load %d/%d (%d%%) Split: %d  Label True: %d  Stored: %d"
                            % (
                                i,
                                num_data_in_split,
                                percent,
                                split,
                                target,
                                y[i],
                            ),
                            end="\n",
                        )
                else:
                    for j in range(26):
                        convertDicts[j][X_cat[i][j]] = 1
                    # debug prints
                    print(
                        "Load %d/%d  Split: %d  Label True: %d  Stored: %d"
                        % (
                            i,
                            num_data_in_split,
                            split,
                            target,
                            y[i],
                        ),
                        end="\r",
                    )
                i += 1

            # store num_data_in_split samples or extras at the end of file
            # count uniques
            # X_cat_t  = np.transpose(X_cat)
            # for j in range(26):
            #     for x in X_cat_t[j,:]:
            #         convertDicts[j][x] = 1
            # store parsed
            filename_s = npzfile + "_{0}.npz".format(split)
            if os.path.exists(filename_s):
                print("\nSkip existing " + filename_s)
            else:
                np.savez_compressed(
                    filename_s,
                    X_int=X_int[0:i, :],
                    # X_cat=X_cat[0:i, :],
                    X_cat_t=np.transpose(X_cat[0:i, :]),  # transpose of the data
                    y=y[0:i],
                )
                print("\nSaved " + npzfile + "_{0}.npz!".format(split))

        if dataset_multiprocessing:
            resultDay[split] = i
            convertDictsDay[split] = convertDicts_day
            return
        else:
            return i

    # create all splits (reuse existing files if possible)
    recreate_flag = False
    convertDicts = [{} for _ in range(26)]
    # WARNING: to get reproducable sub-sampling results you must reset the seed below
    # np.random.seed(123)
    # in this case there is a single split in each day
    for i in range(days):
        npzfile_i = npzfile + "_{0}.npz".format(i)
        npzfile_p = npzfile + "_{0}_processed.npz".format(i)
        if os.path.exists(npzfile_i):
            print("Skip existing " + npzfile_i)
        elif os.path.exists(npzfile_p):
            print("Skip existing " + npzfile_p)
        else:
            recreate_flag = True

    if recreate_flag:
        if dataset_multiprocessing:
            resultDay = Manager().dict()
            convertDictsDay = Manager().dict()
            processes = [Process(target=process_one_file,
                                 name="process_one_file:%i" % i,
                                 args=(npzfile + "_{0}".format(i),
                                       npzfile,
                                       i,
                                       total_per_file[i],
                                       dataset_multiprocessing,
                                       convertDictsDay,
                                       resultDay,
                                       )
                                 ) for i in range(0, days)]
            for process in processes:
                process.start()
            for process in processes:
                process.join()
            for day in range(days):
                total_per_file[day] = resultDay[day]
                print("Constructing convertDicts Split: {}".format(day))
                convertDicts_tmp = convertDictsDay[day]
                for i in range(26):
                    for j in convertDicts_tmp[i]:
                        convertDicts[i][j] = 1
        else:
            for i in range(days):
                total_per_file[i] = process_one_file(
                    npzfile + "_{0}".format(i),
                    npzfile,
                    i,
                    total_per_file[i],
                    dataset_multiprocessing,
                )

    # report and save total into a file
    total_count = np.sum(total_per_file)
    if not os.path.exists(total_file):
        np.savez_compressed(total_file, total_per_file=total_per_file)
    print("Total number of samples:", total_count)
    print("Divided into days/splits:\n", total_per_file)

    # dictionary files
    counts = np.zeros(26, dtype=np.int32)
    if recreate_flag:
        # create dictionaries
        for j in range(26):
            for i, x in enumerate(convertDicts[j]):
                convertDicts[j][x] = i
            dict_file_j = d_path + d_file + "_fea_dict_{0}.npz".format(j)
            if not os.path.exists(dict_file_j):
                np.savez_compressed(
                    dict_file_j,
                    unique=np.array(list(convertDicts[j]), dtype=np.int32)
                )
            counts[j] = len(convertDicts[j])
        # store (uniques and) counts
        count_file = d_path + d_file + "_fea_count.npz"
        if not os.path.exists(count_file):
            np.savez_compressed(count_file, counts=counts)
    else:
        # create dictionaries (from existing files)
        for j in range(26):
            with np.load(d_path + d_file + "_fea_dict_{0}.npz".format(j)) as data:
                unique = data["unique"]
            for i, x in enumerate(unique):
                convertDicts[j][x] = i
        # load (uniques and) counts
        with np.load(d_path + d_file + "_fea_count.npz") as data:
            counts = data["counts"]

    # process all splits
    if dataset_multiprocessing:
        processes = [Process(target=processCriteoAdData,
                           name="processCriteoAdData:%i" % i,
                           args=(d_path,
                                 d_file,
                                 npzfile,
                                 i,
                                 convertDicts,
                                 counts,
                                 )
                           ) for i in range(0, days)]
        for process in processes:
            process.start()
        for process in processes:
            process.join()

    else:
        for i in range(days):
            processCriteoAdData(d_path, d_file, npzfile, i, convertDicts, counts)

    o_file = concatCriteoAdData(
        d_path,
        d_file,
        npzfile,
        trafile,
        days,
        data_split,
        randomize,
        total_per_file,
        total_count,
        o_filename
    )

    return o_file

