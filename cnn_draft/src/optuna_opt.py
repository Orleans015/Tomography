import os
import gc
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, random_split, Subset
from torchmetrics import Accuracy
from torchvision.datasets import MNIST
from torchvision import transforms
import torch.optim as optim
import torchmetrics
import lightning as L
import create_db as c_db
import optuna
from optuna.integration import PyTorchLightningPruningCallback

# Training hyperparameters
INPUTSIZE = 92
OUTPUTSIZE = (110, 110)
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Dataset
DATA_DIR = "/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/"
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 7

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'

default_config = {
  "batch_size": 128,
  "learning_rate": 3e-4,
  "linear_size_1": 256,
  "tcnn_size_1": 16,
  "tcnn_size_2": 8,
  "tcnn_size_3": 4,
  "tcnn_size_4": 4,
}

class Reshape(nn.Module):
  def __init__(self, shape):
    super(Reshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)
  
class TomoModel(L.LightningModule):
  def __init__(self, inputsize, config=default_config):
    super().__init__()
    self.lr = config["learning_rate"]
    self.linear = nn.Sequential(
        nn.Linear(inputsize, config["linear_size_1"]),
        nn.LeakyReLU(),
        nn.Linear(config["linear_size_1"], config["tcnn_size_1"]*11*11),
        nn.LeakyReLU(),
        Reshape([-1, config["tcnn_size_1"], 11, 11]),
    )
    self.anti_conv = nn.Sequential(
        nn.ConvTranspose2d(config["tcnn_size_1"], config["tcnn_size_1"], kernel_size=2, stride=2, padding=0),
        nn.Conv2d(config["tcnn_size_1"], config["tcnn_size_1"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(config["tcnn_size_1"], config["tcnn_size_1"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(config["tcnn_size_1"], config["tcnn_size_2"], kernel_size=2, stride=2, padding=0),
        nn.Conv2d(config["tcnn_size_2"], config["tcnn_size_2"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(config["tcnn_size_2"], config["tcnn_size_2"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(config["tcnn_size_2"], config["tcnn_size_3"], kernel_size=2, stride=2, padding=0),
        nn.Conv2d(config["tcnn_size_3"], config["tcnn_size_3"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(config["tcnn_size_3"], config["tcnn_size_3"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(config["tcnn_size_3"], config["tcnn_size_4"], kernel_size=2, stride=2, padding=0),
        nn.Conv2d(config["tcnn_size_4"], config["tcnn_size_4"], kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(config["tcnn_size_4"], config["tcnn_size_4"], kernel_size=5, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(config["tcnn_size_4"], 1, kernel_size=5, stride=1, padding=0),
    )

    self.net = nn.Sequential(
        self.linear,
        self.anti_conv,
    )

    self.loss_rate = 0.2
    self.loss_fn = nn.MSELoss()
    self.best_val_loss = torch.tensor(float('inf'))
    self.mse = torchmetrics.MeanSquaredError()
    self.mae = torchmetrics.MeanAbsoluteError()
    self.r2 = torchmetrics.R2Score()
    self.training_step_outputs = []
      
  def forward(self, x):
    x = self.net(x)
    return x
  
  def training_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mae = self.mae(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.training_step_outputs.append(loss.detach().cpu().numpy())
    self.log_dict({'train_loss': loss,
                   'train_mae': mae,
                   'train_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mae = self.mae(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.log_dict({'val_loss': loss,
                   'val_mae': mae,
                   'val_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mae = self.mae(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.log_dict({'test_loss': loss,
                   'test_mae': mae,
                   'test_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)
    y_hat = self(x)
    preds = torch.argmax(y_hat, dim=1)
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)
  
class TomographyDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, file_name, n_samples=None):
    self.data_dir = data_dir
    self.file_name = file_name
    bloated_dataset = np.load(
      os.path.join(self.data_dir, self.file_name),
      allow_pickle=True
      )
    bloated_dataset = bloated_dataset[:n_samples] if n_samples is not None else bloated_dataset
    self.data = bloated_dataset['data']
    self.target = bloated_dataset['emiss'].reshape(-1, 1, 110, 110)
    self.labels = bloated_dataset['label']
    self.shots = bloated_dataset['shot']
    self.time = bloated_dataset['time']
    self.dataerr = bloated_dataset['data_err']
    self.coeff = bloated_dataset['target']
    self.emiss = bloated_dataset['emiss'].reshape(-1, 1, 110, 110)
    self.x_emiss = bloated_dataset['x_emiss'][0]
    self.y_emiss = bloated_dataset['y_emiss'][0]
    self.majr = bloated_dataset['majr'][0]
    self.minr = bloated_dataset['minr'][0]
    self.b_tor = bloated_dataset['b_tor']
    self.b_rad = bloated_dataset['b_rad']
    self.phi_tor = bloated_dataset['phi_tor']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    return self.data[idx], self.target[idx]

class TomographyDataModule(L.LightningDataModule):
  def __init__(self, data_dir, file_name, batch_size, num_workers=4):
    super().__init__()
    self.data_dir = data_dir
    self.file_name = file_name
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.mean = 0.
    self.std = 0.

  def prepare_data(self):
    c_db.create_db()

  def setup(self, stage=None):
    entire_dataset = TomographyDataset(self.data_dir, self.file_name, n_samples=None)
    mask_pos = entire_dataset.data > 0
    mask_neg = entire_dataset.data < 0
    entire_dataset.data[mask_pos] = (entire_dataset.data[mask_pos] - entire_dataset.data[mask_pos].mean()) / entire_dataset.data[mask_pos].std()
    entire_dataset.data[mask_neg] = -10
    self.mean = entire_dataset.target.mean()
    self.std = entire_dataset.target.std()
    entire_dataset.target = (entire_dataset.target - self.mean) / self.std
    generator = torch.Generator().manual_seed(42)
    self.train_ds, self.val_ds, self.test_ds = random_split(entire_dataset,
                                              [0.8, 0.1, 0.1],
                                              generator=generator)

  def train_dataloader(self):
    return DataLoader(self.train_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      pin_memory=True,
                      shuffle=True)

  def val_dataloader(self):
    return DataLoader(self.val_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      pin_memory=True,
                      shuffle=False)

  def test_dataloader(self):
    return DataLoader(self.test_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      pin_memory=True,
                      shuffle=False)

def objective(trial):
    config = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "learning_rate": trial.suggest_float("learning_rate", 5e-5, 1e-2),
        "linear_size_1": trial.suggest_categorical("linear_size_1", [32, 64, 128, 256, 512]),
        "tcnn_size_1": trial.suggest_categorical("tcnn_size_1", [2, 4, 8, 16, 32]),
        "tcnn_size_2": trial.suggest_categorical("tcnn_size_2", [2, 4, 8, 16, 32]),
        "tcnn_size_3": trial.suggest_categorical("tcnn_size_3", [2, 4, 8, 16, 32]),
        "tcnn_size_4": trial.suggest_categorical("tcnn_size_4", [2, 4, 8, 16, 32]),
    }

    dm = TomographyDataModule(
        data_dir=DATA_DIR,
        file_name=FILE_NAME,
        batch_size=config["batch_size"],
        num_workers=NUM_WORKERS,
    )

    model = TomoModel(
        inputsize=INPUTSIZE,
        config=config,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        max_epochs=NUM_EPOCHS,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val_loss"),
        ],
        enable_progress_bar=True,
    )

    trainer.fit(model, datamodule=dm)
    val_loss = trainer.callback_metrics["val_loss"].item()
    return val_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=200)

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
