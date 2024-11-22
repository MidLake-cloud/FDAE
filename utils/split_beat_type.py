
import joblib
import os


label_list = ['N', 'L', 'R', 'V']


def split_beats(data_root, file_name, data_mode='train'):
    features, labels, patients = joblib.load(os.path.join(data_root, file_name))
    size = len(features)
    print(type(features), type(labels), type(patients), size)
    class_nums = 4
    each_class_samples = size // 4
    for i in range(class_nums):
        start, end = i*each_class_samples, (i+1)*each_class_samples
        temp_features, temp_labels, temp_patients = features[start:end], labels[start:end], patients[start:end]
        label_idx = temp_labels[0]
        print("label idx {}, type {}".format(label_idx, label_list[label_idx]))
        with open(os.path.join(data_root, 'standard_{}_{}.p'.format(label_list[label_idx], data_mode)), 'wb') as f:
            joblib.dump([temp_features, temp_labels, temp_patients], f)


if __name__ == '__main__':
    frame_len = 216
    data_root = "dataset/data_{}/".format(frame_len)
    file_list = ['standard_train.p', 'standard_valid.p', 'standard_test_fake.p', 'standard_test_aug.p']
    mode_list = ['train', 'valid', 'test_fake', 'test_aug']
    for file_name, data_mode in zip(file_list, mode_list):
        split_beats(data_root=data_root, file_name=file_name, data_mode=data_mode)
        print('File ({}) has been read and split by beat type, save into new file!'.format(file_name))