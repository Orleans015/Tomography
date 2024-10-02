import lightning as L
import torch
import numpy as np
import os
from model import TomoModel
from dataset import TomographyDataModule
import config
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn_zeros

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
  print(f"Validation dataset: {val_loader.dataset[0]}")
  assert False, "Stop here"
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

def generate_map(coefficients):
  """
  Generate the tomography map starting from the coefficients
  """
  m = 2
  l = 7
  radius = 0.459
  # Define the grid
  x_emiss = .../radius
  y_emiss = .../radius
  x_emiss, y_emiss = np.meshgrid(x_emiss, y_emiss)
  # change the grid to polar coordinates
  radii  = np.sqrt(x_emiss**2 + y_emiss**2)
  angles = np.arctan2(y_emiss, x_emiss)
  # compute the Bessel zeros
  zeros = np.array([jn_zeros(im, l) for im in range(m)])
  zero = np.zeros(1)
  zero = np.concatenate((zero, zeros[1,:-1]))
  zeros[1] = zero # all this to add a zero at the beginning of the array!!!
  # compute the Bessel functions 
  J0_xr = np.array([j0(zeros[0]*r) for r in radii.ravel()])
  J1_xr = np.array([j1(zeros[1]*r) for r in radii.ravel()])
  # reshape the arrays
  J0_xr = J0_xr.reshape(len(radii), len(radii[0]), len(zeros[0]))
  J1_xr = J1_xr.reshape(len(radii), len(radii[0]), len(zeros[1]))
  # initialize the map
  g_r_t = np.zeros((len(radii), len(angles)))
  # compute the map
  a0cl, a1cl, a1sl = np.split(coefficients, 3)
  dot0_c = np.dot(J0_xr, a0cl)
  dot1_c = np.dot(J1_xr, a1cl)
  dot1_s = np.dot(J1_xr, a1sl)
  g_r_t = dot0_c + dot1_c * np.cos(angles) + dot1_s * np.sin(angles)
  g_r_t[np.where(g_r_t < 0)] = 0 # get rid of negative values (unphysical)
  g_r_t[radii > 1.0] = -10 # set the values outside the circle to -10
  return g_r_t/radius # return the normalized mapz

if __name__ == "__main__":
  visualize()