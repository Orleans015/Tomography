import lightning as L
import torch
import numpy as np
import os
from model import TomoModel
from dataset import TomographyDataModule
import config
import matplotlib.pyplot as plt

def visualize():
  # Define an instance of the model
  model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  # Load the best model
  version_num = 4
  assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  model.load_state_dict(torch.load(
    f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt",
    )['state_dict'])
  print(model)
  # Define the data module
  data_module = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  # Load the data
  data_module.setup()
  # Define the dataloaders
  val_loader = data_module.val_dataloader()
  test_loader = data_module.test_dataloader()
  # input_data = val_loader.dataset[0][0].unsqueeze(0)
  # reference = val_loader.dataset[0][1]
  input_data = next(iter(val_loader))[0]
  val_reference = next(iter(val_loader))[1]
  # predict on the test dataset 
  test_data = next(iter(test_loader))[0]
  test_reference = next(iter(test_loader))[1]

  v = model(input_data)
  t = model(test_data)
  
  # print the validation predictions
  print(f"Validation predictions: {v}")
  # print the validation reference
  print(f"Validation reference: {val_reference}")
  # compute the error on the validation set
  print(f"Validation error: {v - val_reference}")
  # compute the mean squared error
  print(f"Validation mean squared error: {torch.mean((v - val_reference)**2)}")

  # print some art to separate the results
  print("*" * 50)

  # print the test predictions
  print(f"Test predictions: {t}")
  # print the test reference
  print(f"Test reference: {test_reference}")
  # compute the error on the test set
  print(f"Test error: {t - test_reference}")
  # compute the mean squared error
  print(f"Test mean squared error: {torch.mean((t - test_reference)**2)}")

  # plot the results
  plt.figure()
  plt.plot(val_reference.detach().numpy(), label="Reference", color='orange', marker='o')
  plt.plot(v.detach().numpy(), label="Prediction", color='blue', linestyle='dashed', marker='x')
  # plt.legend()
  plt.savefig("results.png")

  # print some art to separate the results
  print("*" * 50)
  # print(f"Test results: {t}")

if __name__ == "__main__":
  visualize()