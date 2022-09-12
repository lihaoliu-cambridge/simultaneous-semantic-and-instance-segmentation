import copy
import numpy as np


def remove_zeros(pred_array_list):
    nr_patches = pred_array_list.shape[0]
    for idx in range(nr_patches):
        pred_instance_and_type = pred_array_list[idx]
        pred_instance_map = pred_instance_and_type[..., 0]
        pred_type_map = pred_instance_and_type[..., 1]
        for instance_id in np.unique(pred_instance_map):
            if instance_id == 0:
                continue
            pred_2_instance_part = (pred_instance_map == instance_id)
            pred_2_category_ids_in_instance = np.unique(pred_type_map[pred_2_instance_part])
            assert len(pred_2_category_ids_in_instance) == 1
            pred_2_category_id = int(pred_2_category_ids_in_instance[0])
            if pred_2_category_id > 6 or pred_2_category_id == 0:
                pred_instance_map[pred_2_instance_part] = 0
    return pred_array_list


def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos)) 
    return (x1, y1, x2, y2) 


def bb_intersection_over_union(A, B):
    xA = max(A[0], B[0])
    yA = max(A[1], B[1])
    xB = min(A[2], B[2])
    yB = min(A[3], B[3])

    # compute the area of intersection rectangle
    interArea = max(1, xB - xA) * max(1, yB - yA)

    if interArea <= 1.:
        return 0.0

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (A[2] - A[0] + 1) * (A[3] - A[1] + 1)
    boxBArea = (B[2] - B[0] + 1) * (B[3] - B[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

    
def overwrite_ensemble_two_maps(semantic_pred_arrat_list, instance_pred_array_list, semantic_flags=[1,4,5]):
    # Semantic is 1
    # Instance is 2
    semantic_pred_arrat_list = remove_zeros(semantic_pred_arrat_list)
    instance_pred_array_list = remove_zeros(instance_pred_array_list)

    nr_patches = instance_pred_array_list.shape[0]
    for idx in range(nr_patches):
        pred_1_instance_and_type = semantic_pred_arrat_list[idx]
        pred_1_instance_map = pred_1_instance_and_type[..., 0]
        pred_1_type_map = pred_1_instance_and_type[..., 1]
        
        pred_2_instance_and_type = instance_pred_array_list[idx]
        pred_2_instance_map = pred_2_instance_and_type[..., 0]
        pred_2_type_map = pred_2_instance_and_type[..., 1]
        
        pred_1_instance_ids = np.unique(pred_1_instance_map)
        for instance_id in pred_1_instance_ids:
            if instance_id == 0:
                continue
                
            # best for segmentation -> semantic segmentation results Overwrites instance segmentation results
            pred_1_instance_part = (pred_1_instance_map == instance_id)
            pred_1_category_ids_in_instance = np.unique(pred_1_type_map[pred_1_instance_part])
            assert len(pred_1_category_ids_in_instance) == 1
            pred_1_category_id = int(pred_1_category_ids_in_instance[0])
            if pred_1_category_id > 6 or pred_1_category_id == 0:
                raise Exception

            pred_2_potential_instance_id = np.argmax(np.bincount(pred_2_instance_map[pred_1_instance_part].astype(np.int64)))
            if pred_2_potential_instance_id == 0:
                pred_2_type_map[pred_1_instance_part] = pred_1_category_id
                pred_2_instance_map[pred_1_instance_part] = instance_id + 2000
            else:
                pred_2_instance_part = (pred_2_instance_map == pred_2_potential_instance_id)
                pred_2_category_ids_in_instance = np.unique(pred_2_type_map[pred_2_instance_part])
                assert len(pred_2_category_ids_in_instance) == 1
                pred_2_category_id = int(pred_2_category_ids_in_instance[0])
                if pred_2_category_id > 6 or pred_2_category_id == 0:
                    raise Exception

                if pred_1_category_id in semantic_flags:
                    pred_2_instance_map[pred_1_instance_part] = pred_2_potential_instance_id
                    pred_2_type_map[pred_1_instance_part] = pred_1_category_id
                    pred_2_type_map[pred_2_instance_part] = pred_1_category_id
                else:
                    pred_2_instance_map[pred_1_instance_part] = pred_2_potential_instance_id
                    pred_2_type_map[pred_1_instance_part] = pred_2_category_id
                    pred_2_type_map[pred_2_instance_part] = pred_2_category_id

    return instance_pred_array_list
