"""compute_stats.py. Calculates the statistical measurements for the CoNIC Challenge.

This code supports binary panoptic quality for binary segmentation, multiclass panoptic quality for 
simultaneous segmentation and classification and multiclass coefficient of determination (R2) for
multiclass regression. Binary panoptic quality is calculated per image and the results are averaged.
For multiclass panoptic quality, stats are calculated over the entire dataset for each class before taking
the average over the classes.

Usage:
    compute_stats.py [--mode=<str>] [--pred=<path>] [--true=<path>]
    compute_stats.py (-h | --help)
    compute_stats.py --version

Options:
    -h --help                   Show this string.
    --version                   Show version.
    --mode=<str>                Choose either `regression` or `seg_class`.
    --pred=<path>               Path to the results directory.
    --true=<path>               Path to the ground truth directory.

"""

from docopt import docopt
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm
import argparse
from nms import overwrite_ensemble_two_maps


def get_regression_results(pred_array, OUT_DIR, FOLD_IDX):
    # Recalculate Counts
    middle_seg = pred_array[:, 16:240, 26:240, :]

    all_counts = []
    # print(middle_seg.shape[0])
    for i in range(middle_seg.shape[0]):
        cell_counts = [0,0,0,0,0,0]

        instance_and_type = middle_seg[i]
        instance_map = instance_and_type[..., 0]
        type_map = instance_and_type[..., 1]
            
        instance_ids = np.unique(instance_map)
        for instance_id in instance_ids:
            if instance_id == 0:
                continue

            # category_id
            instance_part = (instance_map == instance_id)
            category_ids_in_instance = np.unique(type_map[instance_part])
            if len(category_ids_in_instance) != 1:
                type_map[instance_part] = np.argmax(np.bincount(type_map[instance_part].astype(np.int64)))
                category_ids_in_instance = np.unique(type_map[instance_part])
            assert len(category_ids_in_instance) == 1
            category_id = int(category_ids_in_instance[0])
            if category_id > 6 or category_id == 0:
                continue
                # raise Exception("Only 6 types")

            cell_counts[category_id-1] += 1
        all_counts.append(cell_counts)
        
    all_counts_2_np = np.asarray(all_counts)
    df = pd.DataFrame(data=all_counts_2_np, columns=["neutrophil", "epithelial", "lymphocyte", "plasma", "eosinophil", "connective"])
    
    .pdf.to_csv(f"{OUT_DIR}/pred_fold_{FOLD_IDX}.csv", index=False)
    print(f"save to {OUT_DIR}/pred_fold_{FOLD_IDX}.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--instance_pred', type=str)
    parser.add_argument('--semantic_pred', type=str)
    parser.add_argument('--out_dir_path', type=str)
    parser.add_argument('--fold', type=str)
    args = parser.parse_args()

    instance_pred_path = args.instance_pred
    semantic_pred_path = args.semantic_pred
    OUT_DIR = args.out_dir_path
    FOLD_IDX = args.fold
    
    instance_pred_format = instance_pred_path.split(".")[-1]
    semantic_pred_format = semantic_pred_path.split(".")[-1]
    if instance_pred_format != "npy" or semantic_pred_format != "npy":
        raise ValueError("pred and true must be in npy format.")

    instance_pred_array = np.load(instance_pred_path)
    senmatic_pred_array = np.load(semantic_pred_path)

    ensembled_pred_array = overwrite_ensemble_two_maps(senmatic_pred_array, instance_pred_array)
    np.save(f'{OUT_DIR}/pred_{FOLD_IDX}.npy', ensembled_pred_array)

    get_regression_results(ensembled_pred_array, OUT_DIR=OUT_DIR, FOLD_IDX=FOLD_IDX)
