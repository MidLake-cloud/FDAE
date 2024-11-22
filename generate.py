
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from drivers.FDAe_driver import FDAeDriver
from drivers.cvae_driver import CVAEDriver
from drivers.fsae_driver import fSAEDriver
from drivers.gan_driver import GANDriver
from drivers.simgan_driver import SimGANDriver
from drivers.disnet_driver import DisNetDriver
from utils.load_data import EcgDataset
from models.FDAe import FDAe
from models.cvae import CVAE, ConvCVAE, LinearCVAE
from models.fsae import fSAE, LinearfSAE
from models.gan import DCGenerator, DCDiscriminator
from models.disnet import ConvDisNet, LinearDisnet
from models.simulator import Simulator
import time
import joblib
import os


if __name__ == '__main__':
    use_ablation = False
    generative_model = 'disnet' # ['cvae', 'dcgan']
    model_type = 'conv' # ['linear', 'conv', None]
    ablation = '' # ['pl', 'sl', 'cl']
    assert generative_model in ['cvae', 'dcgan', 'simdcgan', 'disnet'], 'Generative model must be FDAE, CVAE, DCGAN or SimDCGAN!'
    assert model_type in [None, 'linear', 'conv'], 'Model type must be None, linear or conv!'
    beat_types = ['N', 'R', 'L', 'V']
    patients_num = 44
    class_nums = 4
    fake_nums = 10
    latent_dim = 32
    frame_len = 257
    residual_blocks = 2
    lr = 4e-3
    assert frame_len in [216, 257], 'ECG length must be 216 or 257!'
    num_classes = 4
    batch_size = 64
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    mode = 'inter'
    raw_file = "./dataset/{}_data_{}/standard_train.p".format(mode, frame_len)
    raw_data = EcgDataset(p_file=raw_file)
    raw_size = raw_data.__len__()
    raw_dataloader = DataLoader(dataset=raw_data, batch_size=raw_size//class_nums, shuffle=False)
    save_fake_path="./dataset/{}_data_{}/".format(mode, frame_len)
    if model_type == 'conv':
        model_name = 'conv disnet_{}'.format(residual_blocks)
        if frame_len == 257:
            kernel_chs = 32
            d_kernel_chs = 32
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
        elif frame_len == 216:
            kernel_chs = 32
            d_kernel_chs = 32
            stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, 2*kernel_chs, 7, 7, 1), (2*kernel_chs, kernel_chs, 8, 8, 0)]
            d_stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, kernel_chs, 7, 7, 1), (kernel_chs, 1, 8, 8, 0)]
        if ablation != '':
            model_name += ' wo {}'.format(ablation)
        model = ConvDisNet(stack_spec=stack_config, d_stack_spec=d_stack_config, linear_dim=kernel_chs)
    elif model_type == 'linear':
        model_name = 'linear disnet'
        stack_config = [(frame_len, 100), (100, 100), ]
        d_stack_config = [(frame_len, 100), (100, 32), (32, 1)]
        model = LinearDisnet(stack_spec=stack_config, d_stack_spec=d_stack_config, inp_dim=frame_len)
    driver = DisNetDriver(model, frame_len, batch_size, epochs=50, lr=lr, device=device, beta=1, lambda_m=1, lambda_p=1, lambda_dis=1, lambda_cl=1, lambda_r=1, vision_epoch=50, loss_epoch=50, save_root="./{}_out/{}/".format(mode, frame_len), model_name=model_name, model_file='model_best.pt', use_rr=True, use_ablation=False)
    driver.generate(raw_dataloader, save_fake_path="./dataset/{}_data_{}/".format(mode, frame_len), each_nums=350, class_nums=4, use_params=False)