import joblib
import os
import numpy as np


use_rr = True
frame_len = 257
data_path = "./dataset/inter_data_{}/".format(frame_len)
files = ['train.p', 'valid.p', 'test.p']
standard_files = ['standard_train.p', 'standard_valid.p', 'standard_test.p']
for file_name in files:
    file_path = os.path.join(data_path, file_name)
    save_path = os.path.join(data_path, "standard_"+file_name)
    all_features, labels, patients = joblib.load(file_path) # [N, 261]
    print(f"load successfully! {file_path}")
    features: np.ndarray = all_features[:, 4:] if use_rr else all_features
    v_min = features.min(axis=1, keepdims=True)
    v_max = features.max(axis=1, keepdims=True)
    features = (features - v_min) / (v_max - v_min)
    if use_rr:
        all_features[:, 4:] = features
    else:
        all_features = features
    with open(save_path, 'wb') as f:
        joblib.dump([all_features, labels, patients], f)
    print(f"saved in {save_path} successfully!")