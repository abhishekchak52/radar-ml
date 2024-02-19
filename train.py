from torch.utils.data import Dataset, DataLoader, random_split

import os
import torch
from torch import optim, nn, utils, Tensor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L
from autoencoder import LitAutoEncoder, encoder, decoder
from datasets import SincDataset
dataset = SincDataset('.data/sinc_data.txt')

generator2 = torch.Generator().manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(dataset, [0.6, 0.2, 0.2], generator=generator2)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=11)
test_loader = DataLoader(dataset=test_dataset,
                          batch_size=4,
                          shuffle=False,
                          num_workers=11)
val_loader = DataLoader(dataset=val_dataset,
                          batch_size=4,
                          shuffle=False,
                          num_workers=11)

autoencoder = LitAutoEncoder(encoder, decoder)


# train the model (hint: here are some helpful Trainer arguments for rapid idea iteration)
trainer = L.Trainer(max_epochs=100, callbacks=[EarlyStopping(monitor="val_loss", mode="min")])
trainer.fit(model=autoencoder, train_dataloaders=train_loader, val_dataloaders=val_loader)