import joblib
import os
import torch
from torch.utils.data import DataLoader
from models.fsae import fSAE, LinearfSAE
from drivers.fsae_driver import fSAEDriver
from models.disnet import ConvDisNet, LinearDisnet
from drivers.disnet_driver import DisNetDriver
from utils.load_data import EcgDataset


if __name__ == "__main__":
    batch_size = 256
    epochs = 50
    class_nums = 4
    lr = 2e-3
    loss_epoch = 50
    vision_epoch = 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    patients_num = 44
    frame_len = 257
    residual_blocks = 5
    # alpha = 1
    beta = 5
    lambda_m = 0.1
    lambda_p = 1
    lambda_dis = 1
    lambda_cl = 0.1
    lambda_r = 0.8
    model_type = 'conv' # linear or conv
    assert model_type in ['linear', 'conv'], "model type must be linear or conv!"
    train_file = "./dataset/data_{}/standard_train_pair.p".format(frame_len)
    test_file = "./dataset/data_{}/standard_test_fake_pair.p".format(frame_len)
    raw_file = "./dataset/data_{}/standard_test_fake.p".format(frame_len)
    valid_file = "./dataset/data_{}/standard_valid_pair.p".format(frame_len)
    data_file = './dataset/data_{}/standard_train.p'.format(frame_len)
    train_data = EcgDataset(p_file=train_file, frame_len=frame_len)
    valid_data = EcgDataset(p_file=valid_file, frame_len=frame_len)
    test_data = EcgDataset(p_file=test_file, frame_len=frame_len)
    raw_data = EcgDataset(p_file=raw_file, frame_len=frame_len)
    data_set = EcgDataset(p_file=data_file, frame_len=frame_len)
    raw_size = raw_data.__len__()
    # train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    raw_dataloader = DataLoader(dataset=raw_data, batch_size=raw_size//class_nums, shuffle=False)
    # disnet
    kernel_chs = 32
    d_kernel_chs = 32
    
    if model_type == 'linear':
        # Linear FSAE
        model_name = 'linear disnet'
        stack_config = [(frame_len, 100), (100, 100), ]
        d_stack_config = [(frame_len, 100), (100, 32), (32, 1)]
        model = LinearDisnet(stack_spec=stack_config, d_stack_spec=d_stack_config, inp_dim=frame_len)
    elif model_type == 'conv':
        # Conv FSAE
        model_name = 'conv disnet_{}'.format(residual_blocks)
        if frame_len == 216:
            stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, 2*kernel_chs, 7, 7, 1), (2*kernel_chs, kernel_chs, 8, 8, 0)]
            d_stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, kernel_chs, 7, 7, 1), (kernel_chs, 1, 8, 8, 0)]
            model = ConvDisNet(stack_spec=stack_config, d_stack_spec=d_stack_config, linear_dim=kernel_chs)
        elif frame_len == 257:
            if residual_blocks == 5:
                stack_config = [(1, kernel_chs, 5, 4, 0), (kernel_chs, 2*kernel_chs, 4, 4, 0), (2*kernel_chs, 2*kernel_chs, 4, 4, 0), (2*kernel_chs, 2*kernel_chs, 2, 2, 0), (2*kernel_chs, kernel_chs, 2, 2, 0)]
                d_stack_config = [(1, d_kernel_chs, 5, 4, 0), (d_kernel_chs, 2*d_kernel_chs, 4, 4, 0), (2*d_kernel_chs, 2*d_kernel_chs, 4, 4, 0), (2*kernel_chs, kernel_chs, 2, 2, 0), (kernel_chs, 1, 2, 2, 0)]
            elif residual_blocks == 4:
                stack_config = [(1, kernel_chs, 5, 4, 0), (kernel_chs, 2*kernel_chs, 4, 4, 0), (2*kernel_chs, 2*kernel_chs, 4, 4, 0), (2*kernel_chs, kernel_chs, 4, 4, 0)]
                d_stack_config = [(1, d_kernel_chs, 5, 4, 0), (d_kernel_chs, 2*d_kernel_chs, 4, 4, 0), (2*d_kernel_chs, d_kernel_chs, 4, 4, 0), (d_kernel_chs, 1, 4, 4, 0)]
            elif residual_blocks == 3:
                stack_config = [(1, kernel_chs, 5, 4, 0), (kernel_chs, 2*kernel_chs, 8, 8, 0), (2*kernel_chs, kernel_chs, 8, 8, 0)]
                d_stack_config = [(1, kernel_chs, 5, 4, 0), (kernel_chs, kernel_chs, 8, 8, 0), (kernel_chs, 1, 8, 8, 0)]
            elif residual_blocks == 2:
                stack_config = [(1, kernel_chs, 9, 8, 0), (kernel_chs, kernel_chs, 32, 32, 0)]
                d_stack_config = [(1, kernel_chs, 9, 8, 0), (kernel_chs, 1, 32, 32, 0)]
            model = ConvDisNet(stack_spec=stack_config, d_stack_spec=d_stack_config, linear_dim=kernel_chs)
    driver = DisNetDriver(model, frame_len, batch_size, epochs, lr, device, beta, lambda_m, lambda_p, lambda_dis, lambda_cl, lambda_r, vision_epoch, loss_epoch, save_root="./out/{}/".format(frame_len), model_name=model_name, model_file='model_best.pt', use_rr=True, use_ablation=False)
    driver.extract_identity(data_loader, save_file='identity {}'.format(model_name))
    print("over!")