import lightning as L
import torch
import numpy as np
import os
from model import NN
from dataset import TomographyDataModule
import config

def visualize():
  # Define an instance of the model
  # model = NN(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  # Load the best model
  version_num = 4
  model = NN.load_from_checkpoint(
    checkpoint_path=f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt",
    hparams_file=f"TB_logs/my_Tomo_model/version_{version_num}/hparams.yaml",
    inputsize=config.INPUTSIZE,
    learning_rate=config.LEARNING_RATE,
    outputsize=config.OUTPUTSIZE
    )
  # Define the data module
  data_module = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  # Load the data
  data_module.setup()
  # Define the dataloaders
  val_loader = data_module.val_dataloader()
  test_loader = data_module.test_dataloader()
  # Define the trainer
  trainer = L.Trainer(gpus=config.DEVICES, precision=config.PRECISION)
  # Visualize the results
  v = trainer.predict(model, val_loader)
  t = trainer.predict(model, test_loader)
  print(f"Validation results: {v}")
  # print some art to separate the results
  print("*" * 50)
  print(f"Test results: {t}")

if __name__ == "__main__":
  visualize()