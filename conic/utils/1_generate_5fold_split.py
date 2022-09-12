import numpy as np
import joblib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

SEED = 5
info = pd.read_csv('../../dataset/patch_info.csv')
file_names = np.squeeze(info.to_numpy()).tolist()

img_sources = [v.split('-')[0] for v in file_names]
img_sources = np.unique(img_sources)

cohort_sources = [v.split('_')[0] for v in img_sources]
_, cohort_sources = np.unique(cohort_sources, return_inverse=True)

num_trials = 10
splitter = StratifiedShuffleSplit(
    n_splits=num_trials,
    train_size=0.8,
    test_size=0.2,
    random_state=SEED
)

splits = []
split_generator = splitter.split(img_sources, cohort_sources)
for train_indices, valid_indices in split_generator:
    train_cohorts = img_sources[train_indices]
    valid_cohorts = img_sources[valid_indices]

    train_cohort_sources = cohort_sources[train_indices]
    valid_cohort_sources = cohort_sources[valid_indices]
    assert np.intersect1d(train_cohorts, valid_cohorts).size == 0
    train_names = [
        file_name
        for file_name in file_names
        for source in train_cohorts
        if source == file_name.split('-')[0]
    ]
    valid_names = [
        file_name
        for file_name in file_names
        for source in valid_cohorts
        if source == file_name.split('-')[0]
    ]
    train_names = np.unique(train_names)
    valid_names = np.unique(valid_names)
    print(f'Train: {len(train_names):04d} - Valid: {len(valid_names):04d}')
    assert np.intersect1d(train_names, valid_names).size == 0
    train_indices = [file_names.index(v) for v in train_names]
    valid_indices = [file_names.index(v) for v in valid_names]
    splits.append({
        'train': train_indices,
        'valid': valid_indices
    })

    # train indice 4-folds
    splitter2 = StratifiedKFold(n_splits=4, random_state=SEED, shuffle=True)
    split_generator2 = splitter2.split(train_cohorts, train_cohort_sources)
    for train_indices_4fold, valid_indices_4fold in split_generator2:
        train_cohorts_4fold = train_cohorts[train_indices_4fold]
        valid_cohorts_4fold = train_cohorts[valid_indices_4fold]
        
        assert np.intersect1d(train_cohorts_4fold, valid_cohorts_4fold).size == 0
        train_names = [
            file_name
            for file_name in file_names
            for source in train_cohorts_4fold
            if source == file_name.split('-')[0]
        ]
        valid_names = [
            file_name
            for file_name in file_names
            for source in valid_cohorts_4fold
            if source == file_name.split('-')[0]
        ]
        train_names = np.unique(train_names)
        valid_names = np.unique(valid_names)
        assert np.intersect1d(train_names, valid_names).size == 0
        train_indices_4fold = [file_names.index(v) for v in train_names] + valid_indices
        valid_indices_4fold = [file_names.index(v) for v in valid_names] 
        print(f'Train: {len(train_indices_4fold):04d} - Valid: {len(valid_indices_4fold):04d}')
        splits.append({
            'train': train_indices_4fold,
            'valid': valid_indices_4fold
    })

    splits.append({
        'train': train_indices + valid_indices,
        'valid': valid_indices
    })
    

    break

for i in range(5):
    for j in range(5):
        if i != j:
            assert np.intersect1d(splits[i]["valid"], splits[j]["valid"]).size == 0
assert np.intersect1d(splits[5]["train"], splits[5]["valid"]).size == 1018

joblib.dump(splits, '../../dataset/splits.dat')
