
import os
import gc
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List
from collections import Counter


def show_info(dic: Dict[int, List]):
    for k in dic:
        length = len(dic[k])
        print(f"{k}, {length}")

    
def show_label(labels):
    labels_dic = Counter(list(labels))
    print('labels: ', labels_dic)


def split_data(data_root, source_file, des_path, num_per_class, ratio=0.9, class_nums=4):
    f = open(source_file, 'rb')
    gc.disable()
    features, labels, patients_idx, patient_num_beats = joblib.load(f)
    gc.enable()
    f.close()
    print('features: ', features.shape)
    print('labels: ', labels.shape)
    print('patients: ', len(patients_idx))
    show_label(labels)

    patients_idx_set = list(set(patients_idx))
    patients_idx_sorted = sorted(patients_idx_set)
    patients_dic = {idx: patients_idx_sorted.index(idx) for idx in patients_idx_sorted}
    patients_idx = [patients_dic[idx] for idx in patients_idx]

    # print(features, labels, patients_idx, patient_num_beats)
    features_dic: Dict[int, List] = {}
    patients_idx_dic: Dict[int, List] = {}
    for feature, label, patient_idx in zip(features, labels, patients_idx):
        # feature: (261,) 4 RR 257 raw;  label、patient_idx is a number
        # print(feature, label, patient_idx)
        if label in features_dic:
            features_dic[label].append(feature)
            patients_idx_dic[label].append(patient_idx)
        else:
            features_dic[label] = [feature]
            patients_idx_dic[label] = [patient_idx]
    show_info(features_dic)

    train_features_list, train_labels_list, train_patients_list = [], [], [] # train_features: (N_i, D)
    test_features_list, test_labels_list, test_patients_list = [], [], []
    valid_features_list, valid_labels_list, valid_patients_list = [], [], []
    test_fake_features_list, test_fake_labels_list, test_fake_patients_list = [], [], []
    for class_idx in features_dic:
        class_features: np.ndarray = np.array(features_dic[class_idx])
        print(f'class {class_idx}, features shape {class_features.shape}')
        class_patients: np.ndarray = np.array(patients_idx_dic[class_idx])
        print(f'class {class_idx}, patients shape {class_patients.shape}')
        length = len(class_patients)
        # train_len = int(num_per_class*ratio) # 5000*0.8=4000 per class
        # test_len = num_per_class - train_len # 5000-4000=1000
        train_len = int(num_per_class*ratio) # 5000*0.9=4500 2000
        test_len = num_per_class - train_len # 5000-4500=500 2000
        valid_len = 500
        train_len = (train_len - valid_len) // 2 # (4500-500) // 2 = 2000
        test_fake_len = train_len

        permutation = np.random.permutation(length)

        class_features, class_patients = class_features[permutation], class_patients[permutation]
        print('after shuffle: ', class_features.shape, class_patients.shape)

        train_features_list.append(class_features[:train_len, :])
        train_labels_list += [class_idx] * train_len
        train_patients_list.append(class_patients[:train_len])

        valid_features_list.append(class_features[train_len:train_len+valid_len, :])
        valid_labels_list += [class_idx] * valid_len
        valid_patients_list.append(class_patients[train_len:train_len+valid_len])

        test_features_list.append(class_features[train_len+valid_len:train_len+valid_len+test_len, :])
        test_labels_list += [class_idx] * test_len
        test_patients_list.append(class_patients[train_len+valid_len:train_len+valid_len+test_len])

        test_fake_features_list.append(class_features[train_len+valid_len+test_len:train_len+valid_len+test_len+test_fake_len, :])
        test_fake_labels_list += [class_idx] * test_fake_len
        test_fake_patients_list.append(class_patients[train_len+valid_len+test_len:train_len+valid_len+test_len+test_fake_len])

    train_features = np.concatenate(train_features_list, axis=0) # (N_0+N_1+N_2+N_3, D)
    test_features = np.concatenate(test_features_list, axis=0)
    valid_features = np.concatenate(valid_features_list, axis=0)
    test_fake_features = np.concatenate(test_fake_features_list, axis=0)
    train_patients = np.concatenate(train_patients_list, axis=0)
    test_patients = np.concatenate(test_patients_list, axis=0)
    valid_patients = np.concatenate(valid_patients_list, axis=0)
    test_fake_patients = np.concatenate(test_fake_patients_list, axis=0)
    print("after intra-mode data split according to each class...")
    print(f"train features shape {train_features.shape}")
    print(f"valid features shape {valid_features.shape}")
    print(f"test features shape {test_features.shape}")
    print(f"test in fake-train/test features shape {test_fake_features.shape}")
    print(f"train patients shape {train_patients.shape}")
    print(f"valid patients shape {valid_patients.shape}")
    print(f"test patients shape {test_patients.shape}")
    print(f"test in fake-train/test patients shape {test_fake_patients.shape}")
    
    if not os.path.exists(des_path):
        os.makedirs(des_path)
    with open(os.path.join(des_path, "train.p"), "wb") as f:
        joblib.dump([train_features, np.array(train_labels_list), train_patients], f)
    with open(os.path.join(des_path, "test_aug.p"), "wb") as f:
        joblib.dump([test_features, np.array(test_labels_list), test_patients], f)
    with open(os.path.join(des_path, "valid.p"), "wb") as f:
        joblib.dump([valid_features, np.array(valid_labels_list), valid_patients], f)
    with open(os.path.join(des_path, "test_fake.p"), "wb") as f:
        joblib.dump([test_fake_features, np.array(test_fake_labels_list), test_fake_patients], f)
    print('data saved!')


if __name__ == "__main__":
    data_root = "../dataset/"
    features_labels_file = data_root + "archive/intra_features/intra_w_128_129_DS1_rm_bsline_maxRR_RR_raw_MLII_None.p"
    split_data(data_root, source_file=features_labels_file, des_path=data_root+"data/", num_per_class=5000)