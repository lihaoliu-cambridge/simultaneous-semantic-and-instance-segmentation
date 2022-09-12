import os
import json
import numpy as np
import joblib
import pandas as pd
import cv2


def mask2box(mask):
    index = np.argwhere(mask == 1)
    rows = index[:, 0]
    clos = index[:, 1]
    y1 = int(np.min(rows))  # y
    x1 = int(np.min(clos))  # x
    y2 = int(np.max(rows))
    x2 = int(np.max(clos))
    return (x1, y1, x2, y2)


def gen_coco_format_dataset(phase, imgs, labels, indices, fold=0):
    result = {
        "info": {"description": "CoNIC dataset."},
        "categories": [
            {'id': 1, 'name': 'neutrophil'},
            {'id': 2, 'name': 'epithelial'},
            {'id': 3, 'name': 'lymphocyte'},
            {'id': 4, 'name': 'plasma'},
            {'id': 5, 'name': 'eosinophil'},
            {'id': 6, 'name': 'connective'}
        ]
    }

    images_info = []
    labels_info = []
    obj_count = 0
    for idx in indices:
        # Images
        image_name = f"{idx:04d}.jpg"
        print(phase, idx, image_name)

        images_info.append(
            {
                "file_name": image_name,
                "height": 512,
                "width": 512,
                "id": idx
            }
        )

        # Instance Segmentations 1
        instance_and_type = labels[idx]
        instance_map = instance_and_type[..., 0]
        type_map = instance_and_type[..., 1]

        instance_ids = np.unique(instance_map)

        for instance_id in instance_ids:
            if instance_id == 0:
                continue

            # category_id
            instance_part = (instance_map == instance_id)
            category_ids_in_instance = np.unique(type_map[instance_part])
            assert len(category_ids_in_instance) == 1
            category_id = int(category_ids_in_instance[0])
            if category_id > 6 or category_id == 0:
                raise Exception("Only 6 types")

            instance_part = (cv2.resize((instance_part*255).astype('uint8'), (512, 512), interpolation=cv2.INTER_NEAREST) / 255).astype(np.bool)
            
            # area
            area = int(instance_part.sum())

            # bbox
            x1, y1, x2, y2 = mask2box(instance_part)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # segmentation (polygon, which means contour)
            segmentation = []
            contours, _ = cv2.findContours((instance_part * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            assert len(category_ids_in_instance) == 1
            contour = contours[0].flatten().tolist()
            segmentation.append(contour)
            if len(segmentation) == 0:
                raise Exception("Error: no segmentations.")

            # add all label information for one instance
            labels_info.append(
                {
                    "segmentation": segmentation,  # poly
                    "area": area,  # segmentation area
                    "iscrowd": 0,
                    "image_id": idx,
                    "bbox": [x1, y1, w, h],
                    "category_id": category_id,
                    "id": obj_count
                },
            )
            obj_count += 1

    result["images"] = images_info
    result["annotations"] = labels_info
    with open('../../dataset/detectron2/annotations/instances_{}2017_resized_fold_{}.json'.format(phase, fold), 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    if not os.path.exists('../../dataset/detectron2/annotations/'):
        os.makedirs('../../dataset/detectron2/annotations/')

    # ------------------ Get Splitting Information and Get all image and all labels ------------------ 
    splits = joblib.load("../../dataset/splits.dat")
    assert(len(splits)==6)
    imgs = np.load("../../dataset/images.npy")
    labels = np.load("../../dataset/labels.npy")

    train_len = [3963, 3910, 4006, 4012, 4033]
    val_len = [1018, 1071, 975, 969, 948]

    for FOLD_IDX in [0, 1, 2, 3, 4]:
        train_indices = splits[FOLD_IDX]['train']
        valid_indices = splits[FOLD_IDX]['valid']
        assert len(valid_indices)==val_len[FOLD_IDX] and len(train_indices)==train_len[FOLD_IDX]

        gen_coco_format_dataset("train", imgs, labels, train_indices, fold=FOLD_IDX)
        gen_coco_format_dataset("val", imgs, labels, valid_indices, fold=FOLD_IDX)
