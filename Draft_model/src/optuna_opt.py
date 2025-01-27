"""
This script trains a neural network model for tomography using PyTorch Lightning and Optuna for hyperparameter tuning.
Constants:
  INPUTSIZE (int): Size of the input layer.
  OUTPUTSIZE (int): Size of the output layer.
  LEARNING_RATE (float): Learning rate for the optimizer.
  BATCH_SIZE (int): Batch size for training.
  NUM_EPOCHS (int): Number of epochs for training.
  DATA_DIR (str): Directory where the dataset is stored.
  FILE_NAME (str): Name of the dataset file.
  NUM_WORKERS (int): Number of workers for data loading.
  ACCELERATOR (str): Type of accelerator to use ('gpu' or 'cpu').
  DEVICES (list): List of device IDs to use for training.
  PRECISION (str): Precision type for training ('16-mixed' or '32').
Classes:
  TomoModel(L.LightningModule):
    A PyTorch Lightning module defining the neural network model for tomography.
    Methods:
      __init__(self, inputsize, outputsize, config): Initializes the model with given input size, output size, and configuration.
      forward(self, x): Defines the forward pass of the model.
      training_step(self, batch, batch_idx): Defines the training step.
      validation_step(self, batch, batch_idx): Defines the validation step.
      test_step(self, batch, batch_idx): Defines the test step.
      _common_step(self, batch, batch_idx): Common step for training, validation, and test steps.
      predict_step(self, batch, batch_idx): Defines the prediction step.
      configure_optimizers(self): Configures the optimizer for training.
  TomographyDataset(torch.utils.data.Dataset):
    A custom dataset class for tomography data.
    Methods:
      __init__(self, data_dir, file_name): Initializes the dataset with given data directory and file name.
      __len__(self): Returns the length of the dataset.
      __getitem__(self, idx): Returns the data and target for the given index.
  TomographyDataModule(L.LightningDataModule):
    A PyTorch Lightning data module for handling data loading and preprocessing.
    Methods:
      __init__(self, data_dir, file_name, batch_size, num_workers=4): Initializes the data module with given parameters.
      prepare_data(self): Prepares the data (e.g., downloads, IO).
      setup(self, stage=None): Sets up the dataset and splits it into training, validation, and test sets.
      train_dataloader(self): Returns the train dataloader.
      val_dataloader(self): Returns the validation dataloader.
      test_dataloader(self): Returns the test dataloader.
Functions:
  objective(trial): Objective function for Optuna hyperparameter optimization.
  tune_tomo_optuna(num_trials=10): Tunes the model using Optuna with the given number of trials.
Usage:
  Run the script to train and tune the tomography model. The best trial configuration will be printed at the end.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
import numpy as np
import os
import sys
import signal
import gc
from tqdm import tqdm
import torch.utils
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.loggers import TensorBoardLogger
import optuna
from optuna.integration import PyTorchLightningPruningCallback

INPUTSIZE = 92
OUTPUTSIZE = 21
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Dataset
DATA_DIR = '/home/orlandi/devel/Tomography/tomo-rfx/Draft_model/data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 1

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'

default_config = {
    "hidden_layer1_size": 256,
    "hidden_layer2_size": 256,
    "hidden_layer3_size": 256,
    "lr": 1e-3,
    "batch_size": 128,
    "activation_function": nn.LeakyReLU(),
}

# Define the model class
class TomoModel(L.LightningModule):
    def __init__(self, inputsize, outputsize, config=default_config):
        super().__init__()
        self.lr = config["lr"]
        self.hidden_layer1_size = config["hidden_layer1_size"]
        self.hidden_layer2_size = config["hidden_layer2_size"]
        self.hidden_layer3_size = config["hidden_layer3_size"]
        self.activation_function = config["activation_function"]
        self.net = nn.Sequential(
            nn.Linear(inputsize, self.hidden_layer1_size),
            self.activation_function,
            nn.Linear(self.hidden_layer1_size, self.hidden_layer2_size),
            self.activation_function,
            nn.Linear(self.hidden_layer2_size, self.hidden_layer3_size),
            self.activation_function,
            nn.Linear(self.hidden_layer3_size, outputsize)
        )
        self.loss_fn = nn.L1Loss()
        self.mse = torchmetrics.MeanSquaredError()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.r2 = torchmetrics.R2Score()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat.view(-1), y.view(-1))
        self.log_dict({'train/loss': loss.item(),
                       'train/mse': mse.item(),
                       'train/mae': mae.item(),
                       'train/r2': r2.item()},
                      on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return {"loss": loss, "preds": y_hat, "target": y}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat.view(-1), y.view(-1))
        self.log_dict({'val/loss': loss.item(),
                       'val/mse': mse.item(),
                       'val/mae': mae.item(),
                       'val/r2': r2.item()},
                      on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        mse = self.mse(y_hat, y)
        mae = self.mae(y_hat, y)
        r2 = self.r2(y_hat.view(-1), y.view(-1))
        self.log_dict({'test/loss': loss.item(),
                       'test/mse': mse.item(),
                       'test/mae': mae.item(),
                       'test/r2': r2.item()},
                      on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def _common_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Define the Dataset and DataModule classes
class TomographyDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, file_name):
        self.data_dir = data_dir
        self.file_name = file_name
        bloated_dataset = np.load(os.path.join(self.data_dir, self.file_name), allow_pickle=True)
        self.data = torch.Tensor(bloated_dataset['data'])
        self.target = torch.Tensor(bloated_dataset['target'])

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

    def prepare_data(self):
        if os.path.exists(os.path.join(self.data_dir, self.file_name)):
            return
        else:
            print("cannot execute the code")
            return

    def setup(self, stage=None):
        entire_dataset = TomographyDataset(self.data_dir, self.file_name)
        mask_pos = entire_dataset.data > 0
        mask_neg = entire_dataset.data < 0
        entire_dataset.data[mask_pos] = (entire_dataset.data[mask_pos] - entire_dataset.data[mask_pos].mean()) / entire_dataset.data[mask_pos].std()
        entire_dataset.data[mask_neg] = -10
        self.mean = entire_dataset.target.mean()
        self.std = entire_dataset.target.std()
        entire_dataset.target = (entire_dataset.target - self.mean) / self.std
        generator = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds, self.test_ds = random_split(entire_dataset, [0.8, 0.1, 0.1], generator=generator)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)

def objective(trial):
    config = {
        "hidden_layer1_size": trial.suggest_categorical("hidden_layer1_size", [32, 64, 128, 256, 512]),
        "hidden_layer2_size": trial.suggest_categorical("hidden_layer2_size", [32, 64, 128, 256, 512]),
        "hidden_layer3_size": trial.suggest_categorical("hidden_layer3_size", [32, 64, 128, 256, 512]),
        "lr": trial.suggest_loguniform("lr", 5e-5, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64, 128, 256]),
        "activation_function": trial.suggest_categorical("activation_function", [nn.ReLU(), nn.LeakyReLU(), nn.Sigmoid(), nn.Tanh()]),
    }

    logger = TensorBoardLogger("TB_logs", name="Optuna_logs")

    dm = TomographyDataModule(
        data_dir=DATA_DIR,
        file_name=FILE_NAME,
        batch_size=config["batch_size"],
        num_workers=NUM_WORKERS,
    )

    model = TomoModel(
        inputsize=INPUTSIZE,
        outputsize=OUTPUTSIZE,
        config=config,
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        max_epochs=NUM_EPOCHS,
        callbacks=[
            PyTorchLightningPruningCallback(trial, monitor="val/loss"),
        ],
        enable_progress_bar=False,
    )

    trainer.fit(model, datamodule=dm)
    return trainer.callback_metrics["val/loss"].item()

def tune_tomo_optuna(num_trials=10):
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=num_trials)
    return study

if __name__ == "__main__":
    study = tune_tomo_optuna(num_trials=200)

    print("Best trial config: {}".format(study.best_trial.params))
    trial = study.best_trial

    print("Best trial: ")
    print("    Number: {}".format(trial.number))
    print("    Value: {}".format(trial.value))
    print("    Params: ")
    for key, value in trial.params.items():
        print("        {}: {}".format(key, value))
