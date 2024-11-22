import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import joblib
import time
# from memory_profiler import profile
from .driver import Driver
from models.fsae import fSAE
from models.disnet import ConvDisNet, LinearDisnet
from utils.vision import visualization_ecgs

 
class DisNetDriver(Driver):
    def __init__(self, model: ConvDisNet, frame_len, batch, epochs, lr, device, beta, lambda_m, lambda_p, lambda_dis, lambda_cl, lambda_r, vision_epoch, loss_epoch, save_root, model_name, model_file=None, use_rr=True, use_ablation=False):
        super().__init__(model, frame_len, batch, epochs, lr, device, vision_epoch, loss_epoch, save_root, model_name, model_file, use_rr)
        self.model_name = model_name
        self.loss_func = nn.MSELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p
        self.lambda_dis = lambda_dis
        self.beta = beta
        self.lambda_cl = lambda_cl
        self.lambda_r = lambda_r
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.use_abaltion = use_ablation
        if use_ablation:
            self.save_path = f"{save_root}/{model_name}/{lr}/alpha_{lambda_m}_beta_{beta}/"
        else:
            self.save_path = f"{save_root}/{model_name}/{lr}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if model_file != None:
            # load existed model
            self.load_model(model_file)

    def load_model(self, model_file):
        model_path = "{}/{}".format(self.save_path, model_file)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def cal_loss(self, x, y):
        return torch.mean(self.loss_func(x, y), dim=1)

    def contrastive_loss(self, z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label: torch.Tensor, same_patient: torch.Tensor):
        # (z_m_i, z_o_j, z_p_j) -> fake_ecg_ij -> (z_m_ij, z_o_ij, z_p_ij)
        # (z_m_j, z_o_i, z_p_i) -> fake_ecg_ji -> (z_m_ji, z_o_ji, z_p_ji)
        part = self.beta - self.cal_loss(z_m_i, z_m_j)
        medical_loss = same_label * 0.5 * self.cal_loss(z_m_i, z_m_j) + (1 - same_label) * 0.5 * torch.where(0 > part, torch.zeros_like(part), part)
        patient_loss = same_patient * 0.5 * self.cal_loss(z_p_i, z_p_j)
        disen_loss = self.cal_loss(z_m_ij, z_m_i) + self.cal_loss(z_o_ij, z_o_j) + self.cal_loss(z_p_ij, z_p_j) + self.cal_loss(z_m_ji, z_m_j) + self.cal_loss(z_o_ji, z_o_i) + self.cal_loss(z_p_ji, z_p_i)
        return medical_loss.mean(), patient_loss.sum(), disen_loss.mean()

    def train_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
        true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
        ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
        same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
        same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
        # ecgs_1, ecgs_2: (b, frame_len)
        ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
        self.optimizer.zero_grad()
        recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
        recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)
        fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
        fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)
        z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
        z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
        medical_loss, patient_loss, disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
        recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
        classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
        realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
        loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_dis * disen_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
        loss.backward()
        self.optimizer.step()
        output_dict = {
            "recon_1": recon_i,
            "recon_2": recon_j
        }
        return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict

    def train(self, trainloader: DataLoader, validloader: DataLoader):
        epoch_train_losses = []
        epoch_rec_losses = []
        epoch_medical_losses = []
        epoch_patient_losses = []
        epoch_disen_losses = []
        epoch_classifi_losses = []
        epoch_realism_losses = []
        valid_losses = [np.inf]
        # valid_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            train_losses = []
            # valid_losses = []
            rec_losses = []
            patient_losses = []
            medical_losses = []
            disen_losses = []
            classifi_losses = []
            realism_losses = []
            start_time = time.time()
            for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(trainloader):
                # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
                loss, rec_loss, medical_loss, patient_loss, disen_loss, classification_loss, realism_loss, out_dict = self.train_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
                train_losses.append(loss)
                rec_losses.append(rec_loss)
                medical_losses.append(medical_loss)
                patient_losses.append(patient_loss)
                disen_losses.append(disen_loss)
                classifi_losses.append(classification_loss)
                realism_losses.append(realism_loss)
                message = 'Epoch [{}/{}] ({}) train loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, disen loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(epoch+1, self.epochs, idx,
                                                                    np.mean(train_losses),
                                                                    np.mean(rec_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
                print(message)
            end_time = time.time()
            epoch_train_losses.append(np.mean(train_losses))
            epoch_rec_losses.append(np.mean(rec_losses))
            epoch_medical_losses.append(np.mean(medical_losses))
            epoch_patient_losses.append(np.mean(patient_losses))
            epoch_disen_losses.append(np.mean(disen_losses))
            epoch_classifi_losses.append(np.mean(classifi_losses))
            epoch_realism_losses.append(np.mean(realism_losses))
            print("Epoch {} complete! Start evaluate! Cost time {:.4f}".format(epoch+1, end_time-start_time))
            
            valid_loss, _, _, _, _, _, _ = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.epochs,
                                np.mean(train_losses), valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)

        self.save_model("model_final.pt")
        self.draw_losses([epoch_train_losses], ["train loss"], mode="train")
        self.draw_losses([epoch_medical_losses], ['medical loss'], mode='train')
        self.draw_losses([epoch_patient_losses], ['patient loss'], mode='train')
        self.draw_losses([epoch_disen_losses], ['disentanlement loss'], mode='train')
        self.draw_losses([epoch_classifi_losses], ['classification loss'], mode='train')
        self.draw_losses([epoch_realism_losses], ['realism loss'], mode='train')
        self.draw_losses([valid_losses], ['valid loss'], mode='valid')


    def test(self, testloader: DataLoader):
        self.model.eval()
        test_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(testloader):
            loss, recon_loss, medical_loss, patient_loss, disen_loss, classifi_loss, realism_loss, out_dict = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            test_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        
        return np.mean(test_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(disen_losses), np.mean(classifi_losses), np.mean(realism_losses)
    

    def test_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        with torch.no_grad():
            # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
            true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
            ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
            same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
            same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
            # ecgs_1, ecgs_2: (b, frame_len)
            ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
            # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
            recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
            recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)
            fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
            fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)
            z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
            z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
            medical_loss, patient_loss, disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
            recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
            classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
            realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
            loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_dis * disen_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
            output_dict = {
                "recon_1": recon_i,
                "recon_2": recon_j
            }
            return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict


    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(validloader):
            loss, recon_loss, medical_loss, patient_loss, disen_loss, classifi_loss, realism_loss, _ = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            valid_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        message = 'Evaluate complete... valid loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, disen loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(
                                                                    np.mean(valid_losses),
                                                                    np.mean(recon_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
        print(message)
        return np.mean(valid_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(disen_losses), np.mean(classifi_losses), np.mean(realism_losses)
    
    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))

    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve") 
        plt.savefig(os.path.join(self.save_path, f"{'_'.join(titles)} loss curve.png"))
        plt.close()

    def extract_identity(self, data_loader, save_file='identity.npy'):
        identity_list = []
        with torch.no_grad():
            self.model.eval()
            for idx, (ecgs, labels, patients) in enumerate(data_loader):
                # ecgs: [B, frame_len+4]
                ecgs = ecgs[:, 4:] if self.use_rr else ecgs
                ecgs = ecgs.float().to(self.device)
                recon, z_m, z_o, z_p, _, _, _ = self.model(ecgs) # [B, latent_dim]
                identity_list.append(z_p)
                print('({}) extract patient-independent representations(identity)... tensor shape: {}'.format(idx, z_p.shape))
            identity_array = torch.concat(identity_list, dim=0).cpu().numpy()
            save_path = os.path.join('./dataset/data_{}/'.format(self.frame_len), save_file)
            np.save(save_path, identity_array)
            print('Identity saved in {}... tensor shape: {}'.format(save_path, identity_array.shape))

        
    def generate(self, dataloader, save_fake_path, each_nums=2000, class_nums=4, use_params=False):
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to('cpu')
            ecgs_list: list[torch.Tensor] = []
            labels_list = [] # [0, 1, 2, 3]
            patients_list: list[torch.Tensor] = []
            for idx, (ecgs, labels, patients) in enumerate(dataloader):
                ecgs_list.append(ecgs)
                labels_list.append(labels[0])
                patients_list.append(patients)
            fake_contents = []
            fake_labels = []
            fake_patients = []
            start_time = time.time()
            for left_idx, left_label in enumerate(labels_list):
                # 选中ecgs_list[left_idx] --- left_label
                part_one = ecgs_list[left_idx] # [2000, L]
                part_one_patients = patients_list[left_idx]
                for right_idx, right_label in enumerate(labels_list):
                    if right_idx < left_idx:
                        continue
                    # 选中ecgs_list[right_idx] --- right_label
                    part_two = ecgs_list[right_idx] # [2000, L]
                    part_two_patients = patients_list[right_idx]
                    part_one, part_two = part_one.float(), part_two.float()
                    fake_part_two, fake_part_one = self.model.swap_generate(part_one[:, 4:], part_two[:, 4:]) # 交叉生成
                    visualization_ecgs(k="{} to {}".format(left_idx, right_idx), mode="generate", content_list=[
                        part_one[0, 4:].cpu().detach().numpy(), part_two[0, 4:].cpu().detach().numpy(), fake_part_one[0].cpu().detach().numpy(), fake_part_two[0].cpu().detach().numpy()
                    ], label_list=[
                        left_label, right_label, right_label, left_label
                    ], patient_list=[
                        part_one_patients[0], part_two_patients[0], part_one_patients[0], part_two_patients[0],
                    ], intro_list=[
                        "initial ECG 1", "initial ECG 2", "fake ECG 1 generated by FSAE", "fake ECG 2 generated by FSAE"
                    ], save_path=self.save_path)
                    print("select {}(label {}) and {}(label {}) to swap generation... generated ecgs shape: {}".format(left_idx, left_label, right_idx, right_label, fake_part_one.shape))
                    fake_contents.append(torch.concat([part_two[:, :4], fake_part_one], dim=1))
                    fake_contents.append(torch.concat([part_one[:, :4], fake_part_two], dim=1))
                    fake_labels += [right_label] * each_nums
                    fake_labels += [left_label] * each_nums
                    fake_patients.append(part_one_patients)
                    fake_patients.append(part_two_patients)
            fake_contents = torch.concat(fake_contents, dim=0).detach().numpy()
            fake_labels = np.array(fake_labels)
            fake_patients = torch.concat(fake_patients, dim=0).detach().numpy()
            print(f"[{self.model_name}] generate all fake ecgs cost time: {time.time()-start_time}")
            print(f"[{self.model_name}] fake ecgs features shape {fake_contents.shape}")
            print(f"[{self.model_name}] fake ecgs labels shape {fake_labels.shape}")
            print(f"[{self.model_name}] fake ecgs patients shape {fake_patients.shape}")
            if not os.path.exists(save_fake_path):
                os.makedirs(save_fake_path)
            if not use_params:
                save_file = "{}_fake.p".format(self.model_name)
            else:
                save_file = "{}_alpha_{}_beta_{}_fake.p".format(self.model_name, self.alpha, self.beta)
            file_path = os.path.join(save_fake_path, save_file)
            with open(file_path, 'wb') as f:
                joblib.dump([fake_contents, fake_labels, fake_patients], f)
            print("Save fake ecgs over! File has been saved into {}...".format(file_path))


class DisNetDriverWOSl(Driver):
    def __init__(self, model: ConvDisNet, frame_len, batch, epochs, lr, device, beta, lambda_m, lambda_p, lambda_dis, lambda_cl, lambda_r, vision_epoch, loss_epoch, save_root, model_name, model_file=None, use_rr=True, use_ablation=False):
        super().__init__(model, frame_len, batch, epochs, lr, device, vision_epoch, loss_epoch, save_root, model_name, model_file, use_rr)
        self.model_name = model_name
        self.loss_func = nn.MSELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p
        self.lambda_dis = lambda_dis
        self.beta = beta
        self.lambda_cl = lambda_cl
        self.lambda_r = lambda_r
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.use_abaltion = use_ablation
        if use_ablation:
            self.save_path = f"{save_root}/{model_name}/{lr}/alpha_{lambda_m}_beta_{beta}/"
        else:
            self.save_path = f"{save_root}/{model_name}/{lr}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if model_file != None:
            # load existed model
            self.load_model(model_file)

    def load_model(self, model_file):
        # model_path = f"./out_{self.frame_len}/{self.model_name}/{self.lr}/alpha_{self.alpha}_beta_{self.beta}/{model_file}"
        model_path = "{}/{}".format(self.save_path, model_file)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def cal_loss(self, x, y):
        return torch.mean(self.loss_func(x, y), dim=1)

    def contrastive_loss(self, z_m_i, z_p_i, z_m_j, z_p_j, same_label: torch.Tensor, same_patient: torch.Tensor):
        # (z_m_i, z_o_j, z_p_j) -> fake_ecg_ij -> (z_m_ij, z_o_ij, z_p_ij)
        # (z_m_j, z_o_i, z_p_i) -> fake_ecg_ji -> (z_m_ji, z_o_ji, z_p_ji)
        part = self.beta - self.cal_loss(z_m_i, z_m_j)
        medical_loss = same_label * 0.5 * self.cal_loss(z_m_i, z_m_j) + (1 - same_label) * 0.5 * torch.where(0 > part, torch.zeros_like(part), part)
        patient_loss = same_patient * 0.5 * self.cal_loss(z_p_i, z_p_j)
        # disen_loss = self.cal_loss(z_m_ij, z_m_i) + self.cal_loss(z_o_ij, z_o_j) + self.cal_loss(z_p_ij, z_p_j) + self.cal_loss(z_m_ji, z_m_j) + self.cal_loss(z_o_ji, z_o_i) + self.cal_loss(z_p_ji, z_p_i)
        return medical_loss.mean(), patient_loss.sum()

    def train_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
        true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
        ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
        same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
        same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
        # ecgs_1, ecgs_2: (b, frame_len)
        ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
        # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
        self.optimizer.zero_grad()
        recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
        recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)
        medical_loss, patient_loss = self.contrastive_loss(z_m_i, z_p_i, z_m_j, z_p_j, same_label.unsqueeze(1), same_patient.unsqueeze(1))
        recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
        classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
        realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
        loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
        loss.backward()
        self.optimizer.step()
        output_dict = {
            "recon_1": recon_i,
            "recon_2": recon_j
        }
        return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict

    def train(self, trainloader: DataLoader, validloader: DataLoader):
        epoch_train_losses = []
        epoch_rec_losses = []
        epoch_medical_losses = []
        epoch_patient_losses = []
        epoch_disen_losses = []
        epoch_classifi_losses = []
        epoch_realism_losses = []
        valid_losses = [np.inf]
        # valid_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            train_losses = []
            # valid_losses = []
            rec_losses = []
            patient_losses = []
            medical_losses = []
            # disen_losses = []
            classifi_losses = []
            realism_losses = []
            start_time = time.time()
            for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(trainloader):
                # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
                loss, rec_loss, medical_loss, patient_loss, classification_loss, realism_loss, out_dict = self.train_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
                train_losses.append(loss)
                rec_losses.append(rec_loss)
                medical_losses.append(medical_loss)
                patient_losses.append(patient_loss)
                # disen_losses.append(disen_loss)
                classifi_losses.append(classification_loss)
                realism_losses.append(realism_loss)
                message = 'Epoch [{}/{}] ({}) train loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(epoch+1, self.epochs, idx,
                                                                    np.mean(train_losses),
                                                                    np.mean(rec_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    # np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
                print(message)
            end_time = time.time()
            epoch_train_losses.append(np.mean(train_losses))
            epoch_rec_losses.append(np.mean(rec_losses))
            epoch_medical_losses.append(np.mean(medical_losses))
            epoch_patient_losses.append(np.mean(patient_losses))
            # epoch_disen_losses.append(np.mean(disen_losses))
            epoch_classifi_losses.append(np.mean(classifi_losses))
            epoch_realism_losses.append(np.mean(realism_losses))
            print("Epoch {} complete! Start evaluate! Cost time {:.4f}".format(epoch+1, end_time-start_time))
            
            valid_loss, _, _, _, _, _ = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.epochs,
                                np.mean(train_losses), valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)

        self.save_model("model_final.pt")
        # self.draw_losses([epoch_train_losses, epoch_rec_losses, epoch_medical_losses, epoch_patient_losses], ["train loss", "recon loss", "medical loss", "patient loss"], mode="train")
        self.draw_losses([epoch_train_losses], ["train loss"], mode="train")
        self.draw_losses([epoch_medical_losses], ['medical loss'], mode='train')
        self.draw_losses([epoch_patient_losses], ['patient loss'], mode='train')
        # self.draw_losses([epoch_disen_losses], ['disentanlement loss'], mode='train')
        self.draw_losses([epoch_classifi_losses], ['classification loss'], mode='train')
        self.draw_losses([epoch_realism_losses], ['realism loss'], mode='train')
        self.draw_losses([valid_losses], ['valid loss'], mode='valid')


    def test(self, testloader: DataLoader):
        self.model.eval()
        test_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(testloader):
            loss, recon_loss, medical_loss, patient_loss, classifi_loss, realism_loss, out_dict = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            test_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            # disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        
        return np.mean(test_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(classifi_losses), np.mean(realism_losses)
    

    def test_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        with torch.no_grad():
            # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
            true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
            ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
            same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
            same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
            # ecgs_1, ecgs_2: (b, frame_len)
            ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
            # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
            recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
            recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)
            medical_loss, patient_loss = self.contrastive_loss(z_m_i, z_p_i, z_m_j, z_p_j, same_label.unsqueeze(1), same_patient.unsqueeze(1))
            recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
            classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
            realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
            loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
            output_dict = {
                "recon_1": recon_i,
                "recon_2": recon_j
            }
            return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict


    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(validloader):
            loss, recon_loss, medical_loss, patient_loss, classifi_loss, realism_loss, _ = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            valid_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            # disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        message = 'Evaluate complete... valid loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(
                                                                    np.mean(valid_losses),
                                                                    np.mean(recon_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    # np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
        print(message)
        return np.mean(valid_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(classifi_losses), np.mean(realism_losses)
    
    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))

    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve") 
        plt.savefig(os.path.join(self.save_path, f"{'_'.join(titles)} loss curve.png"))
        plt.close()

    def extract_identity(self, data_loader, save_file='identity.npy'):
        identity_list = []
        with torch.no_grad():
            self.model.eval()
            for idx, (ecgs, labels, patients) in enumerate(data_loader):
                # ecgs: [B, frame_len+4]
                ecgs = ecgs[:, 4:] if self.use_rr else ecgs
                ecgs = ecgs.float().to(self.device)
                recon, z_m, z_o, z_p, _, _, _ = self.model(ecgs) # [B, latent_dim]
                identity_list.append(z_p)
                print('({}) extract patient-independent representations(identity)... tensor shape: {}'.format(idx, z_p.shape))
            identity_array = torch.concat(identity_list, dim=0).cpu().numpy()
            save_path = os.path.join('./dataset/data_{}/'.format(self.frame_len), save_file)
            np.save(save_path, identity_array)
            print('Identity saved in {}... tensor shape: {}'.format(save_path, identity_array.shape))


    def generate(self, dataloader, save_fake_path, each_nums=2000, class_nums=4, use_params=False):
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to('cpu')
            ecgs_list: list[torch.Tensor] = []
            labels_list = [] # [0, 1, 2, 3]
            patients_list: list[torch.Tensor] = []
            for idx, (ecgs, labels, patients) in enumerate(dataloader):
                ecgs_list.append(ecgs)
                labels_list.append(labels[0])
                patients_list.append(patients)
            fake_contents = []
            fake_labels = []
            fake_patients = []
            start_time = time.time()
            for left_idx, left_label in enumerate(labels_list):
                part_one = ecgs_list[left_idx]
                part_one_patients = patients_list[left_idx]
                for right_idx, right_label in enumerate(labels_list):
                    if right_idx < left_idx:
                        continue
                    part_two = ecgs_list[right_idx]
                    part_two_patients = patients_list[right_idx]
                    part_one, part_two = part_one.float(), part_two.float()
                    fake_part_two, fake_part_one = self.model.swap_generate(part_one[:, 4:], part_two[:, 4:])
                    visualization_ecgs(k="{} to {}".format(left_idx, right_idx), mode="generate", content_list=[
                        part_one[0, 4:].cpu().detach().numpy(), part_two[0, 4:].cpu().detach().numpy(), fake_part_one[0].cpu().detach().numpy(), fake_part_two[0].cpu().detach().numpy()
                    ], label_list=[
                        left_label, right_label, right_label, left_label
                    ], patient_list=[
                        part_one_patients[0], part_two_patients[0], part_one_patients[0], part_two_patients[0],
                    ], intro_list=[
                        "initial ECG 1", "initial ECG 2", "fake ECG 1 generated by FSAE", "fake ECG 2 generated by FSAE"
                    ], save_path=self.save_path)
                    print("select {}(label {}) and {}(label {}) to swap generation... generated ecgs shape: {}".format(left_idx, left_label, right_idx, right_label, fake_part_one.shape))
                    fake_contents.append(torch.concat([part_two[:, :4], fake_part_one], dim=1))
                    fake_contents.append(torch.concat([part_one[:, :4], fake_part_two], dim=1))
                    fake_labels += [right_label] * each_nums
                    fake_labels += [left_label] * each_nums
                    fake_patients.append(part_one_patients)
                    fake_patients.append(part_two_patients)
            fake_contents = torch.concat(fake_contents, dim=0).detach().numpy()
            fake_labels = np.array(fake_labels)
            fake_patients = torch.concat(fake_patients, dim=0).detach().numpy()
            print(f"[{self.model_name}] generate all fake ecgs cost time: {time.time()-start_time}")
            print(f"[{self.model_name}] fake ecgs features shape {fake_contents.shape}")
            print(f"[{self.model_name}] fake ecgs labels shape {fake_labels.shape}")
            print(f"[{self.model_name}] fake ecgs patients shape {fake_patients.shape}")
            if not os.path.exists(save_fake_path):
                os.makedirs(save_fake_path)
            if not use_params:
                save_file = "{}_fake.p".format(self.model_name)
            else:
                save_file = "{}_alpha_{}_beta_{}_fake.p".format(self.model_name, self.alpha, self.beta)
            file_path = os.path.join(save_fake_path, save_file)
            with open(file_path, 'wb') as f:
                joblib.dump([fake_contents, fake_labels, fake_patients], f)
            print("Save fake ecgs over! File has been saved into {}...".format(file_path))


class DisNetDriverWOPl(Driver):
    def __init__(self, model: fSAE, frame_len, batch, epochs, lr, device, beta, lambda_m, lambda_p, lambda_dis, lambda_cl, lambda_r, vision_epoch, loss_epoch, save_root, model_name, model_file=None, use_rr=True, use_ablation=False):
        super().__init__(model, frame_len, batch, epochs, lr, device, vision_epoch, loss_epoch, save_root, model_name, model_file, use_rr)
        self.model_name = model_name
        self.loss_func = nn.MSELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p
        self.lambda_dis = lambda_dis
        self.beta = beta
        self.lambda_cl = lambda_cl
        self.lambda_r = lambda_r
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.use_abaltion = use_ablation
        if use_ablation:
            self.save_path = f"{save_root}/{model_name}/{lr}/alpha_{lambda_m}_beta_{beta}/"
        else:
            self.save_path = f"{save_root}/{model_name}/{lr}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if model_file != None:
            # load existed model
            self.load_model(model_file)

    def load_model(self, model_file):
        # model_path = f"./out_{self.frame_len}/{self.model_name}/{self.lr}/alpha_{self.alpha}_beta_{self.beta}/{model_file}"
        model_path = "{}/{}".format(self.save_path, model_file)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def cal_loss(self, x, y):
        return torch.mean(self.loss_func(x, y), dim=1)

    def contrastive_loss(self, z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label: torch.Tensor, same_patient: torch.Tensor):
        disen_loss = self.cal_loss(z_m_ij, z_m_i) + self.cal_loss(z_o_ij, z_o_j) + self.cal_loss(z_p_ij, z_p_j) + self.cal_loss(z_m_ji, z_m_j) + self.cal_loss(z_o_ji, z_o_i) + self.cal_loss(z_p_ji, z_p_i)
        return disen_loss.mean()

    def train_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
        true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
        ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
        same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
        same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
        # ecgs_1, ecgs_2: (b, frame_len)
        ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
        # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
        self.optimizer.zero_grad()
        recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
        recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)

        fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
        fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)

        z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
        z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
        disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
        recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
        classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
        realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
        loss: torch.Tensor = recon_loss + self.lambda_dis * disen_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
        loss.backward()
        self.optimizer.step()
        output_dict = {
            "recon_1": recon_i,
            "recon_2": recon_j
        }
        return loss.item(), recon_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict

    def train(self, trainloader: DataLoader, validloader: DataLoader):
        epoch_train_losses = []
        epoch_rec_losses = []
        epoch_medical_losses = []
        epoch_patient_losses = []
        epoch_disen_losses = []
        epoch_classifi_losses = []
        epoch_realism_losses = []
        valid_losses = [np.inf]
        # valid_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            train_losses = []
            # valid_losses = []
            rec_losses = []
            patient_losses = []
            medical_losses = []
            disen_losses = []
            classifi_losses = []
            realism_losses = []
            start_time = time.time()
            for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(trainloader):
                # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
                loss, rec_loss, disen_loss, classification_loss, realism_loss, out_dict = self.train_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
                train_losses.append(loss)
                rec_losses.append(rec_loss)
                # medical_losses.append(medical_loss)
                # patient_losses.append(patient_loss)
                disen_losses.append(disen_loss)
                classifi_losses.append(classification_loss)
                realism_losses.append(realism_loss)
                message = 'Epoch [{}/{}] ({}) train loss: {:.4f}, rec loss: {:.4f}, disen loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(epoch+1, self.epochs, idx,
                                                                    np.mean(train_losses),
                                                                    np.mean(rec_losses),
                                                                    np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
                print(message)
            end_time = time.time()
            epoch_train_losses.append(np.mean(train_losses))
            epoch_rec_losses.append(np.mean(rec_losses))
            epoch_disen_losses.append(np.mean(disen_losses))
            epoch_classifi_losses.append(np.mean(classifi_losses))
            epoch_realism_losses.append(np.mean(realism_losses))
            print("Epoch {} complete! Start evaluate! Cost time {:.4f}".format(epoch+1, end_time-start_time))
            
            valid_loss, _, _, _, _ = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.epochs,
                                np.mean(train_losses), valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)

        self.save_model("model_final.pt")
        # self.draw_losses([epoch_train_losses, epoch_rec_losses, epoch_medical_losses, epoch_patient_losses], ["train loss", "recon loss", "medical loss", "patient loss"], mode="train")
        self.draw_losses([epoch_train_losses], ["train loss"], mode="train")
        # self.draw_losses([epoch_medical_losses], ['medical loss'], mode='train')
        # self.draw_losses([epoch_patient_losses], ['patient loss'], mode='train')
        self.draw_losses([epoch_disen_losses], ['disentanlement loss'], mode='train')
        self.draw_losses([epoch_classifi_losses], ['classification loss'], mode='train')
        self.draw_losses([epoch_realism_losses], ['realism loss'], mode='train')
        self.draw_losses([valid_losses], ['valid loss'], mode='valid')


    def test(self, testloader: DataLoader):
        self.model.eval()
        test_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(testloader):
            loss, recon_loss, disen_loss, classifi_loss, realism_loss, out_dict = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            test_losses.append(loss)
            recon_losses.append(recon_loss)
            # medical_losses.append(medical_loss)
            # patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        
        return np.mean(test_losses), np.mean(recon_losses), np.mean(disen_losses), np.mean(classifi_losses), np.mean(realism_losses)
    

    def test_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        with torch.no_grad():
            # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
            true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
            ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
            same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
            same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
            # ecgs_1, ecgs_2: (b, frame_len)
            ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
            # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)

            recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
            recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)

            fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
            fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)

            z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
            z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
            disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
            recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
            classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
            realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
            loss: torch.Tensor = recon_loss + self.lambda_dis * disen_loss + self.lambda_cl * classification_loss + self.lambda_r * realism_loss
            output_dict = {
                "recon_1": recon_i,
                "recon_2": recon_j
            }
            return loss.item(), recon_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_cl * classification_loss.item(), self.lambda_r * realism_loss.item(), output_dict


    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(validloader):
            loss, recon_loss, disen_loss, classifi_loss, realism_loss, _ = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            valid_losses.append(loss)
            recon_losses.append(recon_loss)
            # medical_losses.append(medical_loss)
            # patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        message = 'Evaluate complete... valid loss: {:.4f}, rec loss: {:.4f}, disen loss: {:.4f}, classification loss: {:.4f}, realism loss: {:.4f}'.format(
                                                                    np.mean(valid_losses),
                                                                    np.mean(recon_losses),
                                                                    np.mean(disen_losses),
                                                                    np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
        print(message)
        return np.mean(valid_losses), np.mean(recon_losses), np.mean(disen_losses), np.mean(classifi_losses), np.mean(realism_losses)
    
    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))

    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve") 
        plt.savefig(os.path.join(self.save_path, f"{'_'.join(titles)} loss curve.png"))
        plt.close()

    def extract_identity(self, data_loader, save_file='identity.npy'):
        identity_list = []
        with torch.no_grad():
            self.model.eval()
            for idx, (ecgs, labels, patients) in enumerate(data_loader):
                # ecgs: [B, frame_len+4]
                ecgs = ecgs[:, 4:] if self.use_rr else ecgs
                ecgs = ecgs.float().to(self.device)
                recon, z_m, z_o, z_p, _, _, _ = self.model(ecgs) # [B, latent_dim]
                identity_list.append(z_p)
                print('({}) extract patient-independent representations(identity)... tensor shape: {}'.format(idx, z_p.shape))
            identity_array = torch.concat(identity_list, dim=0).cpu().numpy()
            save_path = os.path.join('./dataset/data_{}/'.format(self.frame_len), save_file)
            np.save(save_path, identity_array)
            print('Identity saved in {}... tensor shape: {}'.format(save_path, identity_array.shape))
        
    def generate(self, dataloader, save_fake_path, each_nums=2000, class_nums=4, use_params=False):
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to('cpu')
            ecgs_list: list[torch.Tensor] = []
            labels_list = [] # [0, 1, 2, 3]
            patients_list: list[torch.Tensor] = []
            for idx, (ecgs, labels, patients) in enumerate(dataloader):
                ecgs_list.append(ecgs)
                labels_list.append(labels[0])
                patients_list.append(patients)
            fake_contents = []
            fake_labels = []
            fake_patients = []
            start_time = time.time()
            for left_idx, left_label in enumerate(labels_list):
                part_one = ecgs_list[left_idx] # [2000, L]
                part_one_patients = patients_list[left_idx]
                for right_idx, right_label in enumerate(labels_list):
                    if right_idx < left_idx:
                        continue
                    part_two = ecgs_list[right_idx] # [2000, L]
                    part_two_patients = patients_list[right_idx]
                    part_one, part_two = part_one.float(), part_two.float()
                    fake_part_two, fake_part_one = self.model.swap_generate(part_one[:, 4:], part_two[:, 4:])
                    visualization_ecgs(k="{} to {}".format(left_idx, right_idx), mode="generate", content_list=[
                        part_one[0, 4:].cpu().detach().numpy(), part_two[0, 4:].cpu().detach().numpy(), fake_part_one[0].cpu().detach().numpy(), fake_part_two[0].cpu().detach().numpy()
                    ], label_list=[
                        left_label, right_label, right_label, left_label
                    ], patient_list=[
                        part_one_patients[0], part_two_patients[0], part_one_patients[0], part_two_patients[0],
                    ], intro_list=[
                        "initial ECG 1", "initial ECG 2", "fake ECG 1 generated by FSAE", "fake ECG 2 generated by FSAE"
                    ], save_path=self.save_path)
                    print("select {}(label {}) and {}(label {}) to swap generation... generated ecgs shape: {}".format(left_idx, left_label, right_idx, right_label, fake_part_one.shape))
                    fake_contents.append(torch.concat([part_two[:, :4], fake_part_one], dim=1))
                    fake_contents.append(torch.concat([part_one[:, :4], fake_part_two], dim=1))
                    fake_labels += [right_label] * each_nums
                    fake_labels += [left_label] * each_nums
                    fake_patients.append(part_one_patients)
                    fake_patients.append(part_two_patients)
            fake_contents = torch.concat(fake_contents, dim=0).detach().numpy()
            fake_labels = np.array(fake_labels)
            fake_patients = torch.concat(fake_patients, dim=0).detach().numpy()
            print(f"[{self.model_name}] generate all fake ecgs cost time: {time.time()-start_time}")
            print(f"[{self.model_name}] fake ecgs features shape {fake_contents.shape}")
            print(f"[{self.model_name}] fake ecgs labels shape {fake_labels.shape}")
            print(f"[{self.model_name}] fake ecgs patients shape {fake_patients.shape}")
            if not os.path.exists(save_fake_path):
                os.makedirs(save_fake_path)
            if not use_params:
                save_file = "{}_fake.p".format(self.model_name)
            else:
                save_file = "{}_alpha_{}_beta_{}_fake.p".format(self.model_name, self.alpha, self.beta)
            file_path = os.path.join(save_fake_path, save_file)
            with open(file_path, 'wb') as f:
                joblib.dump([fake_contents, fake_labels, fake_patients], f)
            print("Save fake ecgs over! File has been saved into {}...".format(file_path))


class DisNetDriverWOCl(Driver):
    def __init__(self, model: fSAE, frame_len, batch, epochs, lr, device, beta, lambda_m, lambda_p, lambda_dis, lambda_cl, lambda_r, vision_epoch, loss_epoch, save_root, model_name, model_file=None, use_rr=True, use_ablation=False):
        super().__init__(model, frame_len, batch, epochs, lr, device, vision_epoch, loss_epoch, save_root, model_name, model_file, use_rr)
        self.model_name = model_name
        self.loss_func = nn.MSELoss(reduction='none')
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.adversarial_loss = torch.nn.BCELoss()
        self.lambda_m = lambda_m
        self.lambda_p = lambda_p
        self.lambda_dis = lambda_dis
        self.beta = beta
        self.lambda_cl = lambda_cl
        self.lambda_r = lambda_r
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.use_abaltion = use_ablation
        if use_ablation:
            self.save_path = f"{save_root}/{model_name}/{lr}/alpha_{lambda_m}_beta_{beta}/"
        else:
            self.save_path = f"{save_root}/{model_name}/{lr}/"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        if model_file != None:
            # load existed model
            self.load_model(model_file)

    def load_model(self, model_file):
        # model_path = f"./out_{self.frame_len}/{self.model_name}/{self.lr}/alpha_{self.alpha}_beta_{self.beta}/{model_file}"
        model_path = "{}/{}".format(self.save_path, model_file)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
    
    def cal_loss(self, x, y):
        return torch.mean(self.loss_func(x, y), dim=1)


    def contrastive_loss(self, z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label: torch.Tensor, same_patient: torch.Tensor):
        # (z_m_i, z_o_j, z_p_j) -> fake_ecg_ij -> (z_m_ij, z_o_ij, z_p_ij)
        # (z_m_j, z_o_i, z_p_i) -> fake_ecg_ji -> (z_m_ji, z_o_ji, z_p_ji)
        part = self.beta - self.cal_loss(z_m_i, z_m_j)
        medical_loss = same_label * 0.5 * self.cal_loss(z_m_i, z_m_j) + (1 - same_label) * 0.5 * torch.where(0 > part, torch.zeros_like(part), part)
        patient_loss = same_patient * 0.5 * self.cal_loss(z_p_i, z_p_j)
        disen_loss = self.cal_loss(z_m_ij, z_m_i) + self.cal_loss(z_o_ij, z_o_j) + self.cal_loss(z_p_ij, z_p_j) + self.cal_loss(z_m_ji, z_m_j) + self.cal_loss(z_o_ji, z_o_i) + self.cal_loss(z_p_ji, z_p_i)
        return medical_loss.mean(), patient_loss.sum(), disen_loss.mean()

    def train_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
        true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
        ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
        same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
        same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
        # ecgs_1, ecgs_2: (b, frame_len)
        ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
        # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
        self.optimizer.zero_grad()
        recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
        recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)
        fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
        fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)
        z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
        z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
        medical_loss, patient_loss, disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
        recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
        # classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
        realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
        loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_dis * disen_loss + self.lambda_r * realism_loss
        loss.backward()
        self.optimizer.step()
        output_dict = {
            "recon_1": recon_i,
            "recon_2": recon_j
        }
        return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_r * realism_loss.item(), output_dict

    def train(self, trainloader: DataLoader, validloader: DataLoader):
        epoch_train_losses = []
        epoch_rec_losses = []
        epoch_medical_losses = []
        epoch_patient_losses = []
        epoch_disen_losses = []
        epoch_classifi_losses = []
        epoch_realism_losses = []
        valid_losses = [np.inf]
        # valid_loss = 1e9
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            train_losses = []
            # valid_losses = []
            rec_losses = []
            patient_losses = []
            medical_losses = []
            disen_losses = []
            # classifi_losses = []
            realism_losses = []
            start_time = time.time()
            for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(trainloader):
                # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
                loss, rec_loss, medical_loss, patient_loss, disen_loss, realism_loss, out_dict = self.train_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
                train_losses.append(loss)
                rec_losses.append(rec_loss)
                medical_losses.append(medical_loss)
                patient_losses.append(patient_loss)
                disen_losses.append(disen_loss)
                # classifi_losses.append(classification_loss)
                realism_losses.append(realism_loss)
                message = 'Epoch [{}/{}] ({}) train loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, disen loss: {:.4f}, realism loss: {:.4f}'.format(epoch+1, self.epochs, idx,
                                                                    np.mean(train_losses),
                                                                    np.mean(rec_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    np.mean(disen_losses),
                                                                    # np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
                print(message)
            end_time = time.time()
            epoch_train_losses.append(np.mean(train_losses))
            epoch_rec_losses.append(np.mean(rec_losses))
            epoch_medical_losses.append(np.mean(medical_losses))
            epoch_patient_losses.append(np.mean(patient_losses))
            epoch_disen_losses.append(np.mean(disen_losses))
            # epoch_classifi_losses.append(np.mean(classifi_losses))
            epoch_realism_losses.append(np.mean(realism_losses))
            print("Epoch {} complete! Start evaluate! Cost time {:.4f}".format(epoch+1, end_time-start_time))
            
            valid_loss, _, _, _, _, _ = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.epochs,
                                np.mean(train_losses), valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)

        self.save_model("model_final.pt")
        self.draw_losses([epoch_train_losses], ["train loss"], mode="train")
        self.draw_losses([epoch_medical_losses], ['medical loss'], mode='train')
        self.draw_losses([epoch_patient_losses], ['patient loss'], mode='train')
        self.draw_losses([epoch_disen_losses], ['disentanlement loss'], mode='train')
        self.draw_losses([epoch_realism_losses], ['realism loss'], mode='train')
        self.draw_losses([valid_losses], ['valid loss'], mode='valid')


    def test(self, testloader: DataLoader):
        self.model.eval()
        test_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(testloader):
            loss, recon_loss, medical_loss, patient_loss, disen_loss, realism_loss, out_dict = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            test_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            # classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        
        return np.mean(test_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(disen_losses), np.mean(realism_losses)
    

    def test_epoch(self, ecg_pairs: torch.Tensor, label_pairs: torch.Tensor, patient_pairs: torch.Tensor):
        with torch.no_grad():
            # ecg_pairs: (b, 2, dim)    label_pairs: (b, 2)     patient_pairs: (b, 2)
            true_labels = torch.ones(ecg_pairs.shape[0], 1).float().to(self.device) # [b, 1]
            ecgs_i, ecgs_j = (ecg_pairs[:, 0, 4:], ecg_pairs[:, 1, 4:]) if self.use_rr else (ecg_pairs[:, 0, :], ecg_pairs[:, 1, :])
            same_label = (label_pairs[:, 0] == label_pairs[:, 1]).long()
            same_patient = (patient_pairs[:, 0] == patient_pairs[:, 1]).long()
            # ecgs_1, ecgs_2: (b, frame_len)
            ecgs_i, ecgs_j = ecgs_i.float(), ecgs_j.float()
            # ecgs_i, ecgs_j, same_label, same_patient = ecgs_i.to(self.device).float(), ecgs_j.to(self.device).float(), same_label.to(self.device), same_patient.to(self.device)
            recon_i, z_m_i, z_o_i, z_p_i, disease_out_i, identity_out_i, discrim_out_i = self.model(ecgs_i)
            recon_j, z_m_j, z_o_j, z_p_j, disease_out_j, identity_out_j, discrim_out_j = self.model(ecgs_j)

            fake_ecg_ij = self.model.decode(z_m_i, z_o_j, z_p_j)
            fake_ecg_ji = self.model.decode(z_m_j, z_o_i, z_p_i)

            z_m_ij, z_o_ij, z_p_ij = self.model.disentangle(fake_ecg_ij)
            z_m_ji, z_o_ji, z_p_ji = self.model.disentangle(fake_ecg_ji)
            medical_loss, patient_loss, disen_loss = self.contrastive_loss(z_m_i, z_o_i, z_p_i, z_m_j, z_o_j, z_p_j, z_m_ij, z_o_ij, z_p_ij, z_m_ji, z_o_ji, z_p_ji, same_label.unsqueeze(1), same_patient.unsqueeze(1))
            recon_loss = torch.sqrt(torch.mean((recon_i - ecgs_i) ** 2)) + torch.sqrt(torch.mean((recon_j - ecgs_j) ** 2))
            # classification_loss = self.cross_entropy_loss(disease_out_i, label_pairs[:, 0]) + self.cross_entropy_loss(disease_out_j, label_pairs[:, 1]) + self.cross_entropy_loss(identity_out_i, patient_pairs[:, 0]) + self.cross_entropy_loss(identity_out_j, patient_pairs[:, 1])
            realism_loss = self.adversarial_loss(discrim_out_i, true_labels) + self.adversarial_loss(discrim_out_j, true_labels)
            loss: torch.Tensor = recon_loss + self.lambda_m * medical_loss + self.lambda_p * patient_loss + self.lambda_dis * disen_loss + self.lambda_r * realism_loss
            output_dict = {
                "recon_1": recon_i,
                "recon_2": recon_j
            }
            return loss.item(), recon_loss.item(), self.lambda_m * medical_loss.item(), self.lambda_p * patient_loss.item(), self.lambda_dis * disen_loss.item(), self.lambda_r * realism_loss.item(), output_dict


    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        recon_losses = []
        medical_losses = []
        patient_losses = []
        disen_losses = []
        classifi_losses = []
        realism_losses = []
        for idx, (ecg_pairs, label_pairs, patient_pairs) in enumerate(validloader):
            loss, recon_loss, medical_loss, patient_loss, disen_loss, realism_loss, _ = self.test_epoch(ecg_pairs.to(self.device), label_pairs.to(self.device), patient_pairs.to(self.device))
            valid_losses.append(loss)
            recon_losses.append(recon_loss)
            medical_losses.append(medical_loss)
            patient_losses.append(patient_loss)
            disen_losses.append(disen_loss)
            # classifi_losses.append(classifi_loss)
            realism_losses.append(realism_loss)
        message = 'Evaluate complete... valid loss: {:.4f}, rec loss: {:.4f}, medical loss: {:.4f}, patient loss: {:.4f}, disen loss: {:.4f}, realism loss: {:.4f}'.format(
                                                                    np.mean(valid_losses),
                                                                    np.mean(recon_losses),
                                                                    np.mean(medical_losses),
                                                                    np.mean(patient_losses),
                                                                    np.mean(disen_losses),
                                                                    # np.mean(classifi_losses),
                                                                    np.mean(realism_losses)
                                                                )
        print(message)
        return np.mean(valid_losses), np.mean(recon_losses), np.mean(medical_losses), np.mean(patient_losses), np.mean(disen_losses), np.mean(realism_losses)
    
    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))

    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve") 
        plt.savefig(os.path.join(self.save_path, f"{'_'.join(titles)} loss curve.png"))
        plt.close()

    def extract_identity(self, data_loader, save_file='identity.npy'):
        identity_list = []
        with torch.no_grad():
            self.model.eval()
            for idx, (ecgs, labels, patients) in enumerate(data_loader):
                # ecgs: [B, frame_len+4]
                ecgs = ecgs[:, 4:] if self.use_rr else ecgs
                ecgs = ecgs.float().to(self.device)
                recon, z_m, z_o, z_p, _, _, _ = self.model(ecgs) # [B, latent_dim]
                identity_list.append(z_p)
                print('({}) extract patient-independent representations(identity)... tensor shape: {}'.format(idx, z_p.shape))
            identity_array = torch.concat(identity_list, dim=0).cpu().numpy()
            save_path = os.path.join('./dataset/data_{}/'.format(self.frame_len), save_file)
            np.save(save_path, identity_array)
            print('Identity saved in {}... tensor shape: {}'.format(save_path, identity_array.shape))

        
    def generate(self, dataloader, save_fake_path, each_nums=2000, class_nums=4, use_params=False):
        with torch.no_grad():
            self.model.eval()
            self.model = self.model.to('cpu')
            ecgs_list: list[torch.Tensor] = []
            labels_list = [] # [0, 1, 2, 3]
            patients_list: list[torch.Tensor] = []
            for idx, (ecgs, labels, patients) in enumerate(dataloader):
                ecgs_list.append(ecgs)
                labels_list.append(labels[0])
                patients_list.append(patients)
            fake_contents = []
            fake_labels = []
            fake_patients = []
            start_time = time.time()
            for left_idx, left_label in enumerate(labels_list):
                part_one = ecgs_list[left_idx] # [2000, L]
                part_one_patients = patients_list[left_idx]
                for right_idx, right_label in enumerate(labels_list):
                    if right_idx < left_idx:
                        continue
                    part_two = ecgs_list[right_idx] # [2000, L]
                    part_two_patients = patients_list[right_idx]
                    part_one, part_two = part_one.float(), part_two.float()
                    fake_part_two, fake_part_one = self.model.swap_generate(part_one[:, 4:], part_two[:, 4:])
                    visualization_ecgs(k="{} to {}".format(left_idx, right_idx), mode="generate", content_list=[
                        part_one[0, 4:].cpu().detach().numpy(), part_two[0, 4:].cpu().detach().numpy(), fake_part_one[0].cpu().detach().numpy(), fake_part_two[0].cpu().detach().numpy()
                    ], label_list=[
                        left_label, right_label, right_label, left_label
                    ], patient_list=[
                        part_one_patients[0], part_two_patients[0], part_one_patients[0], part_two_patients[0],
                    ], intro_list=[
                        "initial ECG 1", "initial ECG 2", "fake ECG 1 generated by FSAE", "fake ECG 2 generated by FSAE"
                    ], save_path=self.save_path)
                    print("select {}(label {}) and {}(label {}) to swap generation... generated ecgs shape: {}".format(left_idx, left_label, right_idx, right_label, fake_part_one.shape))
                    fake_contents.append(torch.concat([part_two[:, :4], fake_part_one], dim=1))
                    fake_contents.append(torch.concat([part_one[:, :4], fake_part_two], dim=1))
                    fake_labels += [right_label] * each_nums
                    fake_labels += [left_label] * each_nums
                    fake_patients.append(part_one_patients)
                    fake_patients.append(part_two_patients)
            fake_contents = torch.concat(fake_contents, dim=0).detach().numpy()
            fake_labels = np.array(fake_labels)
            fake_patients = torch.concat(fake_patients, dim=0).detach().numpy()
            print(f"[{self.model_name}] generate all fake ecgs cost time: {time.time()-start_time}")
            print(f"[{self.model_name}] fake ecgs features shape {fake_contents.shape}")
            print(f"[{self.model_name}] fake ecgs labels shape {fake_labels.shape}")
            print(f"[{self.model_name}] fake ecgs patients shape {fake_patients.shape}")
            if not os.path.exists(save_fake_path):
                os.makedirs(save_fake_path)
            if not use_params:
                save_file = "{}_fake.p".format(self.model_name)
            else:
                save_file = "{}_alpha_{}_beta_{}_fake.p".format(self.model_name, self.alpha, self.beta)
            file_path = os.path.join(save_fake_path, save_file)
            with open(file_path, 'wb') as f:
                joblib.dump([fake_contents, fake_labels, fake_patients], f)
            print("Save fake ecgs over! File has been saved into {}...".format(file_path))