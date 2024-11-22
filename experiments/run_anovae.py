import joblib
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from models.vae import ConvVAE, LinearVAE, AnoVAE
from models.disnet import ConvDisNet, LinearDisnet
from drivers.vae_driver import VAEDriver, ARVAEDriver
from drivers.anovae_driver import AnoVAEDriver
from utils.load_data import EcgDataset


parser = argparse.ArgumentParser(description='Train an VAE of type Linear VAE or Conv VAE.', )
parser.add_argument('--VAE_TYPE', type=str, default='linear_vae', help='Type of vae, either linearvae or convvae.',
                    required=False, choices=['linear_vae', 'conv_vae'])
parser.add_argument('--BATCH_SIZE', type=int, default=64, help='batch size.',
                    required=False)
parser.add_argument('--EPOCHS', type=int, default=100, help='Number of epochs.', required=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print(classname)
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    args = parser.parse_args()
    vae_type = args.VAE_TYPE
    batch_size = args.BATCH_SIZE
    epochs = args.EPOCHS
    class_nums = 4
    lr = 2e-4
    noise_dim = 10
    loss_epoch = 50
    vision_epoch = 5
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    patients_num = 44
    frame_len = 257
    save_root = "/out/{}/".format(frame_len)
    residual_blocks = 5
    disnet_model_type = 'conv'
    
    kernel_chs = 32 # 100
    d_kernel_chs = 32
    disnet_lr = 2e-3
    if disnet_model_type == 'conv':
        if frame_len == 257:
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
            stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, 2*kernel_chs, 7, 7, 1), (2*kernel_chs, kernel_chs, 8, 8, 0)]
            d_stack_config = [(1, kernel_chs, 4, 4, 0), (kernel_chs, kernel_chs, 7, 7, 1), (kernel_chs, 1, 8, 8, 0)]
        disnet_model_name = 'conv disnet_{}'.format(residual_blocks)
        model_name = 'anovae for {}'.format(disnet_model_name)
        disnet = ConvDisNet(stack_spec=stack_config, d_stack_spec=d_stack_config, linear_dim=kernel_chs)
    elif disnet_model_type == 'linear':
        model_name = 'linear disnet'
        stack_config = [(frame_len, 100), (100, 100), ]
        d_stack_config = [(frame_len, 100), (100, 32), (32, 1)]
        disnet_model_name = 'linear disnet'
        model_name = 'anovae for {}'.format(disnet_model_name)
        disnet = LinearDisnet(stack_spec=stack_config, d_stack_spec=d_stack_config, inp_dim=frame_len)
    disnet_model_path = f"./out/{frame_len}/{disnet_model_name}/{disnet_lr}/model_best.pt"
    disnet.load_state_dict(torch.load(disnet_model_path, map_location=device))
    print("Load DisNet model file {} successfully!".format(disnet_model_path))

    data_file = './dataset/data_{}/identity {}.npy'.format(frame_len, disnet_model_name)
    raw_identity = np.load(data_file)
    print(raw_identity.shape)
    np.random.shuffle(raw_identity) # 打乱样本
    raw_identity = torch.from_numpy(raw_identity)
    data_size = len(raw_identity)
    train_size = int(data_size * 0.9)
    valid_size = data_size - train_size
    train_data = TensorDataset(raw_identity[:train_size, :])
    valid_data = TensorDataset(raw_identity[-valid_size:, :])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=False)
 
    # vae
    if vae_type == 'linear_vae':
        model = AnoVAE(identity_dim=20, latent_dim=10, noise_dim=10)
        driver = AnoVAEDriver(model, disnet.identity_classifier, frame_len, batch_size, epochs, lr, noise_dim, device, vision_epoch, loss_epoch, save_root, model_name=model_name)
        driver.train(train_loader, valid_loader)
        driver.generate(100)
    elif vae_type == 'conv_vae':
        model = AnoVAE(identity_dim=20, latent_dim=10, noise_dim=10)
        driver = AnoVAEDriver(model, disnet.identity_classifier, frame_len, batch_size, epochs, lr, noise_dim, device, vision_epoch, loss_epoch, save_root, model_name=model_name)
        driver.train(train_loader, valid_loader)
        driver.generate(100)
    print("over!")