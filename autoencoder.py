import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt


import os
from torch import optim, nn, utils, Tensor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L




# define any number of nn.Modules (or use your current ones)
encoder = nn.Sequential(
    nn.Linear(1000, 500), 
    nn.ReLU(), 
    nn.Linear(500, 250), 
    nn.ReLU(),
    nn.Linear(250, 100), 
    nn.ReLU(),
    nn.Linear(100, 50), 
    nn.ReLU(),
    nn.Linear(50, 20), 
    nn.ReLU(),
    nn.Linear(20, 10),
    )
decoder = nn.Sequential(
    nn.Linear(10, 20), 
    nn.ReLU(),
    nn.Linear(20, 50), 
    nn.ReLU(),
    nn.Linear(50, 100), 
    nn.ReLU(),
    nn.Linear(100, 250), 
    nn.ReLU(),
    nn.Linear(250, 500), 
    nn.ReLU(),
    nn.Linear(500, 1000), 
    nn.Tanh(),
    )

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("test_loss", test_loss)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        test_loss = nn.functional.mse_loss(x_hat, x)
        self.log("val_loss", test_loss)
    

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder
autoencoder = LitAutoEncoder(encoder, decoder)