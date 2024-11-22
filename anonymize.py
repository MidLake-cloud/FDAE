
import torch
import joblib
import numpy as np
from models.disnet import ConvDisNet, LinearDisnet
from models.vae import AnoVAE
from drivers.disnet_driver import DisNetDriver


if __name__ == '__main__':
    frame_len = 257
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)
    raw_data_file = './dataset/data_{}/standard_train.p'.format(frame_len)
    raw_data, raw_labels, raw_patients = joblib.load(raw_data_file)
    raw_data = torch.from_numpy(raw_data).float().to(device)
    print('raw ECG dataset shape: ', raw_data.shape)
    k = 5

    residual_blocks = 5
    disnet_model_type = 'conv'
    disnet_lr = 2e-3
    kernel_chs = 32
    d_kernel_chs = 32
    if disnet_model_type == 'conv':
        disnet_model_name = 'conv disnet_{}'.format(residual_blocks)
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
        disnet = ConvDisNet(stack_spec=stack_config, d_stack_spec=d_stack_config, linear_dim=kernel_chs)
    elif disnet_model_type == 'linear':
        disnet_model_name = 'linear disnet'
        stack_config = [(frame_len, 100), (100, 100), ]
        d_stack_config = [(frame_len, 100), (100, 32), (32, 1)]
        disnet = LinearDisnet(stack_spec=stack_config, d_stack_spec=d_stack_config, inp_dim=frame_len)
    disnet_model_path = f"./out/{frame_len}/{disnet_model_name}/{disnet_lr}/model_best.pt"
    disnet.load_state_dict(torch.load(disnet_model_path, map_location=device))
    disnet = disnet.to(device)
    disnet.eval()

    noise_dim = 10
    ano_vae_name = 'anovae for {}'.format(disnet_model_name)
    ano_vae_lr = 2e-4
    ano_vae = AnoVAE(identity_dim=20, latent_dim=10, noise_dim=10)
    ano_vae_path = './out/{}/{}/{}/model_best.pt'.format(frame_len, ano_vae_name, ano_vae_lr)
    ano_vae.load_state_dict(torch.load(ano_vae_path, map_location=device))
    ano_vae = ano_vae.to(device)
    ano_vae.eval()

    ano_ecgs = []
    ano_ecgs_label = []
    with torch.no_grad():
        for idx, (raw_ecg, raw_label, raw_patient) in enumerate(zip(raw_data, raw_labels, raw_patients)):
            raw_ecg = raw_ecg.reshape(1, -1).float().to(device).repeat(k, 1) # [k, 4+frame_len]
            # print(raw_ecg.device)
            z_m, z_o, z_p = disnet.disentangle(raw_ecg[:, 4:])
            z = torch.randn(k, noise_dim).to(device).to(device) # (num_samples, noise_dim)
            generated_identity: torch.Tensor = ano_vae.sample(z)
            generated_ecg = disnet.decode(z_m, z_o, generated_identity) # [k, frame_len]
            fake_samples = torch.concat([raw_ecg[:, :4], generated_ecg], dim=1)
            ano_ecgs.append(fake_samples)
            ano_ecgs_label += [raw_label] * k
            print('Generating anonymized ECGs... batch {}'.format(idx))
        ano_ecgs = torch.concat(ano_ecgs, dim=0).cpu().numpy()
    ano_ecgs_label = np.array(ano_ecgs_label)
    print('generated ECG dataset shape: {}, label shape: {}'.format(ano_ecgs.shape, ano_ecgs_label.shape))
    generative_model = '{} ano'.format(disnet_model_name)
    save_file = './dataset/data_{}/{}_fake.p'.format(frame_len, generative_model)
    with open(save_file, 'wb') as f:
        joblib.dump([ano_ecgs, ano_ecgs_label, []], f)
    print('Anonymized ECG saved in path: {}'.format(save_file))