import joblib
import numpy as np
import torch
from typing import Dict, Tuple
import random
import os


def statistics_ecgs(all_features, all_labels, all_patients) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
    labels_features_dic: Dict[int, torch.Tensor] = {}
    labels_patients_dic: Dict[int, torch.Tensor] = {}
    counter = 1
    for feature, label, patient in zip(all_features, all_labels, all_patients):
        label = int(label)
        # feature
        print(f"[{counter}/{len(all_features)}], label {label}.")
        counter += 1
        if label not in labels_features_dic:
            temp_1 = [feature]
            temp_2 = [patient]
            labels_features_dic[label] = temp_1
            labels_patients_dic[label] = temp_2
        else:
            labels_features_dic[label].append(feature)
            labels_patients_dic[label].append(patient)
    # print(type(all_features), type(all_labels), type(all_patients))
    for k in labels_features_dic:
        labels_features_dic[k] = np.vstack(labels_features_dic[k]) # (N_i, 261)
        labels_patients_dic[k] = np.array(labels_patients_dic[k]) # (N_i, )
        print(f"label {k}, features shape {labels_features_dic[k].shape}")
    return labels_features_dic, labels_patients_dic


def make_dataset(data_root, p_file, save_file, nums=5):
    all_features, all_labels, all_patients = joblib.load(os.path.join(data_root, p_file))
    labels_features_dic, labels_patients_dic = statistics_ecgs(all_features, all_labels, all_patients)
    selected_pair_features, selected_pair_labels, selected_pair_patients = [], [], []
    for label in labels_features_dic:
        size = len(labels_features_dic[label])
        for k in range(size):
            print(f"coping [{k+1}/{size}] of label {label}...")
            for other_label in labels_features_dic:
                selected_idx = random.sample(torch.arange(size).tolist(), nums) # (nums,)
                # sub_features = labels_features_dic[other_label][selected_idx].unsqueeze(1) # (nums, 1, dim)
                sub_features = np.expand_dims(labels_features_dic[other_label][selected_idx], axis=1) # (nums, 1, dim)
                # sub_labels = torch.LongTensor([other_label] * nums) # (nums,)
                sub_labels = np.expand_dims(np.array([other_label] * nums), axis=1) # (nums, 1)
                # sub_patients = labels_patients_dic[other_label][selected_idx].unsqueeze(0) # (1, nums)
                sub_patients = np.expand_dims(labels_patients_dic[other_label][selected_idx], axis=1) # (nums, 1)
                # repeat_features = labels_features_dic[label][k].unsqueeze(0).repeat(nums, 1).unsqueeze(1) # (nums, 1, dim)
                repeat_features = np.expand_dims(np.tile(np.expand_dims(labels_features_dic[label][k], axis=0), (nums, 1)), axis=1) # (nums, 1, dim)
                # repeat_labels = torch.LongTensor([label] * nums).unsqueeze(0) # (1, nums,)
                repeat_labels = np.expand_dims(np.array([label] * nums), axis=1) # (nums, 1)
                # repeat_patients = torch.LongTensor([labels_patients_dic[label][k]] * nums).unsqueeze(0) # (1, nums)
                repeat_patients = np.expand_dims(np.array([labels_patients_dic[label][k]] * nums), axis=1) # (nums, 1)
                selected_pair_features.append(np.concatenate([repeat_features, sub_features], axis=1)) # (nums, 2, dim)
                selected_pair_labels.append(np.concatenate([repeat_labels, sub_labels], axis=1)) # (nums, 2)
                selected_pair_patients.append(np.concatenate([repeat_patients, sub_patients], axis=1)) # (nums, 2)
    selected_pair_features = np.concatenate(selected_pair_features, axis=0) # (N, 2, dim)
    selected_pair_labels = np.concatenate(selected_pair_labels, axis=0) # (N, 2)
    selected_pair_patients = np.concatenate(selected_pair_patients, axis=0) # (N, 2)
    with open(os.path.join(data_root, save_file), "wb") as f:
        joblib.dump([selected_pair_features, selected_pair_labels, selected_pair_patients], f)
    print(f"Pair dataset {save_file} has been saved! Size: {len(selected_pair_features)}")


if __name__ == "__main__":
    frame_len = 257
    data_root = "./dataset/inter_data_{}/".format(frame_len)
    p_files = ["standard_train.p", "standard_valid.p"]
    save_files = ["standard_train_pair.p", "standard_valid_pair.p"]
    k = 1
    for p_file, save_file in zip(p_files, save_files):
        make_dataset(data_root, p_file, save_file, nums=k)