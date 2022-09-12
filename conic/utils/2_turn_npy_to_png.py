import os
import numpy as np
import joblib
import cv2
import pandas as pd
from PIL import Image


# # ------------------ Get Splitting Information and Get all image and all labels ------------------ 
splits = joblib.load("../../dataset/splits.dat")
train_len = [3963, 3910, 4006, 4012, 4033]
val_len = [1018, 1071, 975, 969, 948]
imgs = np.load("../../dataset/images.npy")
labels = np.load("../../dataset/labels.npy")

# thess images are black, 
error_list = [88, 93, 99, 425, 431, 687, 688, 689, 691, 692, 693, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 709, 710, 711, 752, 757, 762, 767, 772, 775, 776, 777, 778, 779, 780, 781, 782, 783, 806, 807, 808, 811, 817, 818, 1410, 1414, 1415, 1419, 1420, 1424, 1425, 1428, 1429, 1430, 1431, 1688, 1715, 1898, 1899, 1900, 1901, 1902, 1903, 1904, 1905, 1910, 1915, 1930, 2188, 2189, 2190, 2191, 2192, 2193, 2194, 2195, 2198, 2199, 2218, 2252, 2290, 2291, 2448, 2520, 2521, 2522, 2528, 2529, 2536, 3315, 3364, 3919, 4598, 4601, 4603, 4604, 4938, 4940, \
        1036, 1041, 1046, 1047, 1048, 1051, 1052, 1053, 1054, 1056, 1057, 1058, 1059, 1061, 1062, 1063, 1064, 1065, 1071, 1144, 1149, 1828, 1833, 1838, 1841, 1842, 1843, 1847, 1848, 1849, 1850, 1851, 1852, 1853, 1854, 1855, 1856, 1862, 1863, 3084]

# Generate npy for hovernet 
if not os.path.exists('../../dataset/hovernet/npy/'):
    os.makedirs('../../dataset/hovernet/npy/') 

if not os.path.exists('../../dataset/hovernet/csv/'):
    os.makedirs('../../dataset/hovernet/csv/') 

for FOLD_IDX in [0, 1, 2, 3, 4]:
    train_indices = splits[FOLD_IDX]['train']
    # train_indices = [x for x in train_indices if x not in error_list]
    valid_indices = splits[FOLD_IDX]['valid']
    # valid_indices = [x for x in valid_indices if x not in error_list]
    assert len(valid_indices)==val_len[FOLD_IDX] and len(train_indices)==train_len[FOLD_IDX]

    # ------------------ Splitted .npy Files------------------ 
    train_imgs = imgs[train_indices]
    np.save(f'../../dataset/hovernet/npy/train_image_fold_{FOLD_IDX}.npy', train_imgs)

    train_labels = labels[train_indices]
    np.save(f'../../dataset/hovernet/npy/train_true_fold_{FOLD_IDX}.npy', train_labels)

    valid_imgs = imgs[valid_indices]
    np.save(f'../../dataset/hovernet/npy/valid_image_fold_{FOLD_IDX}.npy', valid_imgs)

    valid_labels = labels[valid_indices]
    np.save(f'../../dataset/hovernet/npy/valid_true_fold_{FOLD_IDX}.npy', valid_labels)

    # ------------------ Counts ------------------ 
    gt_counts = pd.read_csv("../../dataset/hovernet/csv/counts.csv")

    train_gt_counts = gt_counts.loc[train_indices]
    train_gt_counts.to_csv(f"../../dataset/hovernet/csv/counts_train_fold_{FOLD_IDX}.csv", index=False)

    val_gt_counts = gt_counts.loc[valid_indices]
    val_gt_counts.to_csv(f"../../dataset/hovernet/csv/counts_valid_fold_{FOLD_IDX}.csv", index=False)

# Generate image for detectron2
if not os.path.exists('../../dataset/detectron2/resized_images/'):
    os.makedirs('../../dataset/detectron2/resized_images/') 

for idx in range(imgs.shape[0]):
    print(idx)
    img = imgs[idx]
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(f'../../dataset/detectron2/resized_images/{idx:04d}.jpg', img_resized)
