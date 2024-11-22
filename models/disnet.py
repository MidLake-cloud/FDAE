import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResidualEncoder, ResidualDecoder


class MLP(nn.Module):
    def __init__(self, dim, latent_dim, class_nums) -> None:
        super().__init__()
        self.latent_layer = nn.Linear(dim, latent_dim)
        self.classifier = nn.Linear(latent_dim, class_nums)
        self.dropout = nn.Dropout()

    def forward(self, z):
        out = self.dropout(self.latent_layer(z))
        out = self.classifier(out)
        return F.softmax(out, dim=1)


class Discriminator(nn.Module):
    def __init__(self, stack_spec) -> None:
        super().__init__()
        ops = []
        for i, (in_c, out_c, kernel, stride, padding) in enumerate(stack_spec):
            # (1, 100, 2049,  2048)
            last = i == (len(stack_spec)-1)
            ops.append(ResidualEncoder(in_c, out_c, kernel, stride, padding, activation=nn.LeakyReLU(),
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass
        ops.append(nn.Sigmoid())
        self.net = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor):
        out = self.net(x.unsqueeze(1)) # [B, 1]
        return out


class LinearDiscriminator(nn.Module):
    def __init__(self, stack_spec) -> None:
        super().__init__()
        ops = []
        for i, (in_feat, out_feat) in enumerate(stack_spec):
            last = i == (len(stack_spec)-1)
            ops.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if not last:
                ops.append(nn.ReLU())
        ops.append(nn.Sigmoid())
        self.net = nn.Sequential(*ops)

    def forward(self, x: torch.Tensor):
        out = self.net(x)
        return out


class ConvDisNet(nn.Module):
    def __init__(self, stack_spec, d_stack_spec, linear_dim=100, latent_dim=20, disease_latent=50, identity_latent=50, medical_labels=4, identity_labels=44) -> None:
        super().__init__()
        encode_ops = []
        decode_ops = []
        for i, (in_c, out_c, kernel, stride, padding) in enumerate(stack_spec):
            # (1, 100, 2049,  2048)
            last = i == (len(stack_spec)-1)
            encode_ops.append(ResidualEncoder(in_c, out_c, kernel, stride, padding, activation=nn.LeakyReLU(),
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass
        for i, (out_c, in_c, kernel, stride, padding) in enumerate(stack_spec[::-1]):
            # (1, 100, 2049, 2048)
            last = (i == len(stack_spec)-1)
            decode_ops.append(ResidualDecoder(in_c, out_c, kernel, stride, padding, activation=nn.LeakyReLU(),
                                              dropout=0.1,
                                              last=last))
            if not last:
                pass
        self.encoder = nn.Sequential(*encode_ops)
        self.decoder = nn.Sequential(*decode_ops)
        self.patient_linear = nn.Linear(linear_dim, latent_dim)
        self.medical_linear = nn.Linear(linear_dim, latent_dim)
        self.other_linear = nn.Linear(linear_dim, latent_dim)
        self.fusion_linear = nn.Linear(latent_dim*3, linear_dim)
        self.discriminator = Discriminator(d_stack_spec)
        self.disease_classifier = MLP(dim=latent_dim, latent_dim=disease_latent, class_nums=medical_labels)
        self.identity_classifier = MLP(dim=latent_dim, latent_dim=identity_latent, class_nums=identity_labels)

    def disentangle(self, ecg: torch.Tensor):
        # ecg: [B, frame_len]
        h = self.encoder(ecg.unsqueeze(1))
        h = h.squeeze(-1) # [B, linear_dim]
        return self.medical_linear(h), self.other_linear(h), self.patient_linear(h)

    def decode(self, z_m, z_o, z_p):
        fusion_z = self.fusion_linear(torch.concat([z_m, z_o, z_p], dim=1)).unsqueeze(-1) # [B, latent_dim, 1]
        return self.decoder(fusion_z).squeeze(1)

    def forward(self, ecg):
        z_m, z_o, z_p = self.disentangle(ecg)
        recon = self.decode(z_m, z_o, z_p)
        disease_out = self.disease_classifier(z_m)
        identity_out = self.identity_classifier(z_p)
        discrim_out = self.discriminator(recon)
        return recon.squeeze(1), z_m, z_o, z_p, disease_out, identity_out, discrim_out.squeeze(-1)

    def swap_generate(self, ecg_i, ecg_j):
        z_m_i, z_o_i, z_p_i = self.disentangle(ecg_i)
        z_m_j, z_o_j, z_p_j = self.disentangle(ecg_j)
        fake_ecg_ij, fake_ecg_ji = self.decode(z_m_i, z_o_j, z_p_j), self.decode(z_m_j, z_o_i, z_p_i)
        return fake_ecg_ij, fake_ecg_ji


class LinearDisnet(nn.Module):
    def __init__(self, stack_spec, d_stack_spec, inp_dim=257, linear_dim=100, latent_dim=20, disease_latent=50, identity_latent=50, medical_labels=4, identity_labels=44) -> None:
        super().__init__()
        encode_ops = []
        decode_ops = []
        self.frame_len = inp_dim
        for i, (in_feat, out_feat) in enumerate(stack_spec):
            # (1, 100, 2049,  2048)
            last = i == (len(stack_spec)-1)
            encode_ops.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if not last:
                encode_ops.append(nn.ReLU())
        for i, (out_feat, in_feat) in enumerate(stack_spec[::-1]):
            # (1, 100, 2049, 2048)
            last = i == (len(stack_spec)-1)
            decode_ops.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            if not last:
                decode_ops.append(nn.ReLU())
        self.encoder = nn.Sequential(*encode_ops)
        self.decoder = nn.Sequential(*decode_ops)
        self.patient_linear = nn.Linear(linear_dim, latent_dim)
        self.medical_linear = nn.Linear(linear_dim, latent_dim)
        self.other_linear = nn.Linear(linear_dim, latent_dim)
        self.fusion_linear = nn.Linear(latent_dim*3, linear_dim)
        self.discriminator = LinearDiscriminator(d_stack_spec)
        self.disease_classifier = MLP(dim=latent_dim, latent_dim=disease_latent, class_nums=medical_labels)
        self.identity_classifier = MLP(dim=latent_dim, latent_dim=identity_latent, class_nums=identity_labels)

    def disentangle(self, ecg: torch.Tensor):
        # ecg: [B, frame_len]
        h = self.encoder(ecg)
        h = h.squeeze(-1) # [B, linear_dim]
        return self.medical_linear(h), self.other_linear(h), self.patient_linear(h)

    def decode(self, z_m, z_o, z_p):
        fusion_z = self.fusion_linear(torch.concat([z_m, z_o, z_p], dim=1)) # [B, latent_dim, 1]
        return self.decoder(fusion_z)

    def forward(self, ecg):
        z_m, z_o, z_p = self.disentangle(ecg)
        recon = self.decode(z_m, z_o, z_p)
        disease_out = self.disease_classifier(z_m)
        identity_out = self.identity_classifier(z_p)
        discrim_out = self.discriminator(recon)
        return recon, z_m, z_o, z_p, disease_out, identity_out, discrim_out

    def swap_generate(self, ecg_i, ecg_j):
        z_m_i, z_o_i, z_p_i = self.disentangle(ecg_i)
        z_m_j, z_o_j, z_p_j = self.disentangle(ecg_j)
        fake_ecg_ij, fake_ecg_ji = self.decode(z_m_i, z_o_j, z_p_j), self.decode(z_m_j, z_o_i, z_p_i)
        return fake_ecg_ij, fake_ecg_ji


