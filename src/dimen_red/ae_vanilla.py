"""
Vanilla Autoencoder to reduce dimensionality of images 
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))


import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
from math import ceil
import torch.optim as optim

import numpy as np

from typing import Optional, Tuple, Union, Dict, Any
import pandas as pd
import json


# Initialize weights with LeCun normal intialization
def initialize_weights(net_l, scale=1) -> None:
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


class vanilla_autoencoder(nn.Module):
    def __init__(self, device, hp: Dict[str, Any], snapshot_dir=None) -> None:
        r"""Vanilla Autoencoder class for dimensionality reduction

        Args:
            device :
            hp (Dict[str, Any]) : Dictionary of training hyperparameters.
            snapshot_dir (str) : Directory to store the training result.
            
        Returns:
            None

        """
        super().__init__()

        self.device = device

        self._hp = hp
        self._snapshot_dir = snapshot_dir
        img_shape = self._hp["model_params"]["img_shape"]

        self._scaling_factor = 1.0

        if "scaling_factor" in self._hp["model_params"].keys():
            self._scaling_factor = self._hp["model_params"]["scaling_factor"]
        self._hp["model_params"]["M"] = ceil(img_shape[1] / 8)
        self._hp["model_params"]["output_padding"] = 0

        self.best_valid_loss = 9999

        self.early_stopping_limit = 20
        self.epochs_no_improve = 0

        if img_shape[1] % 8 == 0:
            self._hp["model_params"]["output_padding"] = 1

        self.recons_loss_function = nn.MSELoss(reduction="mean")

        self.enc_conv, self.enc_fc = self.construct_encoder()
        self.dec_deconv, self.dec_fc = self.construct_decoder()

        initialize_weights(self)
        self.instantiate_optimizer()
        if self._snapshot_dir is not None:
            self.export_hyperparameters(self._snapshot_dir)

        self.train_loss = []
        self.valid_loss = []

    def construct_encoder(self) -> Tuple[nn.Sequential]:
        r"""Construct encoder based on the model hyperparameters.

        Args:
            None

        Returns:
            enc_conv (nn.Sequential): Convolutional layers in the encoder.
            enc_fc (nn.Sequential): Fully connected layers in the encoder.
        """
        in_ch = self._hp["model_params"]["img_shape"][0]
        nz = self._hp["model_params"]["nz"]
        M = self._hp["model_params"]["M"]

        enc_conv = nn.Sequential(
            nn.Conv2d(in_ch, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        enc_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * M * M, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, nz),
            nn.Sigmoid(),
        )

        return enc_conv, enc_fc

    def construct_decoder(self) -> Tuple[nn.Sequential]:
        r"""Construct decoder based on the model hyperparameters.

        Args:
            None

        Returns:
            dec_conv (nn.Sequential): Convolutional layers in the encoder.
            dec_fc (nn.Sequential): Fully connected layers in the encoder.
        """

        nz = self._hp["model_params"]["nz"]
        M = self._hp["model_params"]["M"]
        out_ch = self._hp["model_params"]["img_shape"][0]
        output_padding = self._hp["model_params"]["output_padding"]

        dec_fc = nn.Sequential(
            nn.Linear(nz, 1024, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512 * M * M, bias=False),
            nn.BatchNorm1d(512 * M * M),
            nn.ReLU(),
        )

        dec_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                512,
                128,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=2, bias=False),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_ch, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

        return dec_deconv, dec_fc

    def instantiate_optimizer(self) -> None:
        r"""Set the autoencoder optimizer with the given hyperparameters.

        Args:
            None
        Returns:
            None
        """
        self = self.to(self.device)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=self._hp["optim_params"]["lr"],
            betas=self._hp["optim_params"]["betas"],
        )

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        r"""Encode the input image into a latent space

        Args:
            z (torch.Tensor) : Input image.

        Returns:
            latent_feature (torch.Tensor) : Latent feature of the image.
        """

        out = self.enc_conv(z)
        return self.enc_fc(out) * self._scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        r"""Reconstruct image from a input latent feature.

        Args:
            z (torch.Tensor) : Input latent feature.

        Returns:
            out (torch.Tensor) : Reconstructed image.
        """

        out = self.dec_fc(z)
        out = out.view(
            -1, 512, self._hp["model_params"]["M"], self._hp["model_params"]["M"]
        )
        out = self.dec_deconv(out)

        return out

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward pass of the autoencoder.

        Args:
            z (torch.Tensor): Original input image.

        Returns:
            x (torch.Tensor): Latent feauture extracted from the original image.
            recons_image (torch.Tensor): Image reconstructed with the autoencoder.
        """
        x = self.encode(z)
        return x, self.decode(x)

    def compute_loss(self, x_batch: Union[torch.Tensor, np.ndarray], y_batch=None):
        if type(x_batch) is np.ndarray:
            x_batch = torch.from_numpy(x_batch).to(self.device)

        latent, recons = self.forward(x_batch.float())

        return self.recons_loss_function(recons, x_batch.float())

    def train_batch(self, x_batch: torch.Tensor, y_batch: torch.Tensor = None) -> float:
        loss = self.compute_loss(x_batch, y_batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train_all_batches(self, trainloader):
        loss = []
        for img, label in trainloader:
            img = img.to(self.device)
            label = label.to(self.device)

            loss.append(self.train_batch(img, label))

        return np.mean(np.array(loss))

    def train_model(
        self,
        num_epoch: int,
        trainloader: torch.utils.data.DataLoader,
        validloader: torch.utils.data.DataLoader,
    ) -> None:
        for epoch in range(1, num_epoch + 1):
            train_loss = self.train_all_batches(trainloader)
            valid_loss = self.valid(validloader)

            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            self.display_loss(epoch, num_epoch, train_loss, valid_loss)

            self.save_best_loss_model(valid_loss)
            self.save_loss(epoch)

            if self.early_stopping():
                break

    def early_stopping(self) -> bool:
        if self.epochs_no_improve >= self.early_stopping_limit:
            return True
        return False

    @torch.no_grad()
    def predict(self, dataloader):
        img_shape = self._hp["model_params"]["img_shape"]
        latent_features = np.zeros(
            (len(dataloader.dataset), self._hp["model_params"]["nz"])
        )
        recons_images = np.zeros(
            (len(dataloader.dataset), img_shape[0], img_shape[1], img_shape[2])
        )
        labels = np.zeros((len(dataloader.dataset),))
        self.eval()

        idx = 0
        for img, label in dataloader:
            latent, recons = self.forward(img.to(self.device))
            latent_features[idx : idx + len(img), :] = latent.cpu().detach().numpy()
            recons_images[idx : idx + len(img), :] = recons.cpu().detach().numpy()
            labels[idx : idx + len(img)] = label.cpu().detach().numpy().flatten()

            idx = idx + len(img)
        return latent_features, recons_images, labels

    @torch.no_grad()
    def valid(self, validloader: torch.utils.data.DataLoader) -> float:
        r"""Evaluate the validation loss for the model and save the model if a
            new minimum loss is found.

        Args:
            validloader (torch.utils.data.DataLoader) : Pytorch data loader with
                the validation data.

        Returns:
            loss (float) : Validation loss.
        """
        self.eval()
        loss = []
        for img, label in validloader:
            img = img.to(self.device)
            label = label.to(self.device)

            loss.append(self.compute_loss(img, label).item())

        loss = np.mean(np.array(loss))

        self.save_best_loss_model(loss)

        return loss

    @staticmethod
    def display_loss(
        epoch: int, num_epoch: int, train_loss: torch.Tensor, valid_loss: torch.Tensor
    ) -> None:
        print(
            f"Epoch : {epoch}/{num_epoch}, "
            f"Train loss (average) = {train_loss.item():.8f}"
        )
        print(f"Epoch : {epoch}/{num_epoch}, " f"Valid loss = {valid_loss.item():.8f}")

    def save_loss(self, epoch: int) -> None:
        epochs = np.arange(1, epoch + 1)

        df = pd.DataFrame(
            {
                "epochs": epochs,
                "train_loss": self.train_loss,
                "valid_loss": self.valid_loss,
            }
        )

        df.to_csv(os.path.join(self._snapshot_dir, "output.csv"))

    def save_best_loss_model(
        self,
        valid_loss: float,
        trainloader: torch.utils.data.DataLoader = None,
        validloader: torch.utils.data.DataLoader = None,
    ) -> None:
        r"""Stores the current model if it achieves the lowest validation loss;
        otherwise, it increases ``self.epochs_no_improve by 1``.


        Args:
            valid_loss (float): Validation loss at the current epoch.

        Returns:
            None
        """
        if self.best_valid_loss > valid_loss:
            self.epochs_no_improve = 0
            print(f"New min: {self.best_valid_loss:.2e}")

            self.best_valid_loss = valid_loss
            if self._snapshot_dir is not None:
                torch.save(
                    self.state_dict(), os.path.join(self._snapshot_dir, "best_model.pt")
                )
        else:
            self.epochs_no_improve += 1

    def load_model(self, model_path: str) -> None:
        r"""Load PyTorch Autoencoder model stored in the given path.

        Args:
            model_path (str): Path of the model to be loaded.
        Returns:
            None
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError("No path to load model.")
        self.load_state_dict(
            torch.load(model_path, map_location=torch.device(self.device))
        )

    def export_hyperparameters(self, snapshot_dir: str) -> None:
        r"""Store the hyperparameter used for the autoencoder training in the given path.

        Args:
            snapshot_dir (str): Directory to store the hyperparameters.

        Returns:
            None
        """
        file_path = os.path.join(snapshot_dir, "ae_hyperparameters.json")
        with open(file_path, "w") as file:
            json.dump(self._hp, file)
