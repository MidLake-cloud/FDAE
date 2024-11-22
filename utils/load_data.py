import numpy as np
import torch
from torch.utils.data import Dataset
import joblib


def load_ecg(p_file_name):
    return joblib.load(p_file_name)


def scale_signal(signal: np.ndarray, min_val=-0.01563, max_val=0.042557):
    """
    :param min:
    :param max:
    :return:
    """
    scaled = np.interp(signal, (signal.min(), signal.max()), (min_val, max_val))
    return scaled
 

class EcgDataset(Dataset):
    '''
        For Pair Dataset, features shape (N=160000, 2, 220)
        For Other Dataset, features shape (N=160000, 220)
    '''
    def __init__(self, p_file, frame_len=257, use_ar=False, use_patients=True, transform=None) -> None:
        super().__init__()
        self.features, self.labels, self.patients_idx = load_ecg(p_file) # [N, 4+frame_len]
        self.use_patients = use_patients
        self.transform = transform
        self.frame_len = frame_len
        self.use_ar = use_ar
        if self.use_ar:
            self.ar_indices = [112, 119, 120, 124, 132, 142, 146, 149, 150, 171] # 被选中的特征索引
        if transform:
            self.features = self.scale()
            print("Scaled over! Dataset scale: {}".format(self.features.shape))
        print("Dataset scale: {}".format(self.features.shape))
    
    def scale(self):
        size = len(self.features)
        scaled_features = []
        for i in range(size):
            print("Scaled ECG...({}/{})".format(i, size))
            sample = np.concatenate([self.features[i, :4], self.transform(self.features[i, -self.frame_len:])], axis=0) # [4+frame_len]
            scaled_features.append(sample)
        return np.array(scaled_features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        if self.use_patients:
            # if self.transform:
            #     return self.transform(self.features[index, -self.frame_len:]), self.labels[index], self.patients_idx[index]
            if self.use_ar:
                return self.features[index, :], self.labels[index], self.patients_idx[index], self.features[index, self.ar_indices]
            return self.features[index, :], self.labels[index], self.patients_idx[index]
        else:
            # if self.transform:
            #     return self.transform(self.features[index, -self.frame_len:]), self.labels
            if self.use_ar:
                return self.features[index, :], self.labels[index], self.features[index, self.ar_indices]
            return self.features[index, :], self.labels[index]


class EcgPairDataset(Dataset):
    def __init__(self, p_file) -> None:
        super().__init__()
        self.features_pair, self.labels_pair, self.patients_pair = load_ecg(p_file)
    
    def __len__(self):
        return len(self.features_pair)
    
    def __getitem__(self, index):
        return torch.FloatTensor(self.features_pair[index]), torch.LongTensor(self.labels_pair[index]), torch.LongTensor(self.patients_pair[index])


class Scale(object):
    def __call__(self, ecg):
        heartbeat = scale_signal(ecg)
        return heartbeat


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        heartbeat, label = sample['cardiac_cycle'], sample['label']
        return {'cardiac_cycle': (torch.from_numpy(heartbeat)).double(),
                'label': torch.from_numpy(label),
                'beat_type': sample['beat_type'],
                }