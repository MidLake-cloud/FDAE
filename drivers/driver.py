import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import time
from typing import List
from utils.vision import process_image, draw_loss

 
class Driver:
    def __init__(self, model: nn.Module, frame_len, batch, epochs, lr, device, vision_epoch, loss_epoch, save_root, model_name, model_file=None, use_rr=True):
        self.device = device
        self.frame_len = frame_len
        self.model = model.to(device)
        self.use_rr = use_rr
        self.save_root = save_root
        self.model_name = model_name
        self.batch = batch
        self.epochs = epochs
        self.vision_epoch = vision_epoch
        self.loss_epoch = loss_epoch
        self.valid_loss_list: List = []
        self.train_loss_list: List = []
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.save_path = f"{save_root}/{model_name}/{lr}/" if lr != None else f"{save_root}/{model_name}/"


    def train_epoch(self, features, labels, patients_idx=None):
        # features: (b, dim)    labels: (b, )
        self.optimizer.zero_grad()
        out = self.model(features, labels)
        # recon_loss = self.loss_func(ecgs_1, recon_1).mean() + self.loss_func(ecgs_2, recon_2).mean()
        recon_loss = torch.sqrt(torch.mean((out - features) ** 2)) + torch.sqrt(torch.mean((out - features) ** 2))
        recon_loss.backward()
        self.optimizer.step()
        return out, recon_loss.item()
        # return self.model(features, patients_idx, None)


    def train(self, trainloader: DataLoader, validloader: DataLoader):
        counter = 0
        epoch_train_losses = []
        valid_losses = [np.inf]
        for epoch in range(self.epochs):
            self.model.train()
            train_losses = []
            start_time = time.time()
            for idx, (features, labels, patients_idx) in enumerate(trainloader):
                features, labels, patients_idx = features.to(self.device), labels.to(self.device), patients_idx.to(self.device)
                out, loss = self.train_epoch(features, labels, patients_idx)
                # torch.nn.utils.clip_grad_norm_(parameters, 10.)
                train_losses.append(loss)
                message = 'Epoch [{}/{}] ({}) train loss: {:.4f}'.format(epoch+1, self.epochs, idx,
                                                                    np.mean(train_losses),
                                                                )
                print(message)
            end_time = time.time()
            epoch_train_loss = np.mean(train_losses)
            epoch_train_losses.append(epoch_train_loss)
            print(f"Epoch {epoch} complete! Start evaluate!")
            valid_loss = self.valid(validloader)
            message = 'Epoch [{}/{}] train loss: {:.4f}, valid loss: {:.4f}, {:.1f}s'.format(epoch+1, self.epochs,
                                 epoch_train_loss, valid_loss, (end_time - start_time))
            print(message)
            if valid_loss < np.min(valid_losses):
                self.save_model("model_best.pt")
                print('Valid loss decrease from {:.4f} to {:.4f}, saving to {}'.format(np.min(valid_losses), valid_loss, os.path.join(self.save_path, "model_best.pt")))
            valid_losses.append(valid_loss)
            # scheduler.step(valid_loss)
        self.draw_losses(losses=[epoch_train_losses, valid_losses], titles=["train loss", "valid loss"], mode="train")


    def valid(self, validloader):
        self.model.eval()
        valid_losses = []
        for idx, (features, labels, patients_idx) in enumerate(validloader):
            features, labels, patients_idx = features.to(self.device), labels.to(self.device), patients_idx.to(self.device)
            out, loss = self.train_epoch(features, labels, patients_idx)
            valid_losses.append(loss)
            # self.valid_loss_list.append(loss.item())
        valid_losses.append(loss)
        print(f"valid over. mean loss: {loss}")
        valid_loss = np.mean(valid_losses)
        message = 'Evaluate complete... valid loss: {:.4f}'.format(valid_loss)
        print(message)
        return valid_loss


    def test_epoch(self, features, labels, patients_idx=None):
        out = self.model(features, labels)
        loss = torch.sqrt(torch.mean((out - features) ** 2)) + torch.sqrt(torch.mean((out - features) ** 2))
        return out, loss
    

    def test(self, testloader: DataLoader):
        print("testing!")
        self.model.eval()
        test_losses = []
        for idx, (features, labels, patients_idx) in enumerate(testloader):
            features, labels, patients_idx = features.to(self.device).float(), labels.to(self.device), patients_idx.to(self.device)
            out, loss = self.test_epoch(features, labels, patients_idx)
            test_losses.append(loss)
        return np.mean(test_losses)


    def load_model(self, model_file):
        model_path = f"./out_{self.frame_len}/{self.model_name}/{self.lr}/{model_file}"
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("Load model file {} successfully!".format(model_path))


    def save_model(self, model_file):
        torch.save(self.model.state_dict(), os.path.join(self.save_path, model_file))


    def draw_losses(self, losses: list, titles: list, mode: str="train"):
        for loss, title in zip(losses, titles):
            plt.plot(loss, label=title)
        plt.legend()
        plt.title("loss curve")
        plt.savefig(os.path.join(self.save_path, f"{'_'.join(titles)} loss.png"))
        plt.close()