import lightning as L
import torch
import numpy as np
import os
from model import TomoModel
from dataset import TomographyDataModule
import config
import matplotlib.pyplot as plt
from scipy.special import j0, j1, jn_zeros
from utils import compute_bessel_n_mesh
import time
# import ipywidgets as widgets
# from IPython.display import display
import seaborn as sns

# Set the style of the plots
sns.set_theme(context='notebook', 
        style='white', 
        palette='deep', 
        font='sans-serif', 
        font_scale=1, 
        color_codes=True, 
        rc={"axes.grid": False}
        )


def visualize():
  # Define an instance of the model
  model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE, config.OUTPUTSIZE)
  # Load the best model
  version_num = 21
  assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  model.load_state_dict(torch.load(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt",)['state_dict'])
  print(model)
  # Define the data module
  data_module = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  # Load the data
  data_module.setup()
  # Define the dataloaders
  val_loader = data_module.val_dataloader()
  # print(f"Validation dataset: {val_loader.dataset[0]}")
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
  plt.plot(val_reference[0].detach().numpy(), label="Reference", color='orange', marker='o')
  plt.plot(v[0].detach().numpy(), label="Prediction", color='blue', linestyle='dashed', marker='x')
  # plt.legend()
  plt.savefig("results.png")

  # print some art to separate the results
  print("*" * 50)
  # print(f"Test results: {t}")

def generate_maps(dataloader, model):
  """
  Generate the tomography map starting from the coefficients
  """
  # get the machine radius
  radius = dataloader.dataset.dataset.minr
  # Define the grid, i want to access the x_emiss and y_emiss values in the dataset
  x_emiss = (dataloader.dataset.dataset.x_emiss - dataloader.dataset.dataset.majr)/radius
  y_emiss = dataloader.dataset.dataset.y_emiss/radius
  # create the meshgrid
  x_emiss, y_emiss = np.meshgrid(x_emiss, y_emiss)
  # change the grid to polar coordinates
  radii  = np.sqrt(x_emiss**2 + y_emiss**2)
  angles = np.arctan2(y_emiss, x_emiss)

  # get the precomputed coefficients
  pc_coeffs = dataloader.dataset.dataset.target
  # get the coefficients from the model
  coefficients = model(dataloader.dataset.dataset.data).detach().numpy()
  
  # Define the orders of the bessel functions
  m = 2
  l = 7
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
  
  # initialize the maps
  g_r_t_model = np.zeros((len(radii), len(angles)))
  g_r_t_pc = np.zeros((len(radii), len(angles)))

  for index in range(len(pc_coeffs)):
    # compute the map from the model
    a0cl, a1cl, a1sl = np.split(coefficients[index], 3)
    dot0_c = np.dot(J0_xr, a0cl)
    dot1_c = np.dot(J1_xr, a1cl)
    dot1_s = np.dot(J1_xr, a1sl)
    g_r_t_model = dot0_c + dot1_c * np.cos(angles) + dot1_s * np.sin(angles)
    g_r_t_model[np.where(g_r_t_model < 0)] = 0 # get rid of negative values (unphysical)
    g_r_t_model[radii > 1.0] = -10 # set the values outside the circle to -10

    # compute the map from the precomputed coefficients
    a0cl, a1cl, a1sl = np.split(pc_coeffs[index], 3)
    dot0_c = np.dot(J0_xr, a0cl)
    dot1_c = np.dot(J1_xr, a1cl)
    dot1_s = np.dot(J1_xr, a1sl)
    g_r_t_pc = dot0_c + dot1_c * np.cos(angles) + dot1_s * np.sin(angles)
    g_r_t_pc[np.where(g_r_t_pc < 0)] = 0 # get rid of negative values (unphysical)
    g_r_t_pc[radii > 1.0] = -10 # set the values outside the circle to -10
    return g_r_t_model/radius, g_r_t_pc/radius # return the normalized maps

def plot_maps(model, dataloader):
  '''This function uses the calc_em function of the TomoModel class to generate
  the emissivity maps from the coefficients. It then plots the maps side by side
  and saves the figure as a png file.
  '''
  val_loader = dataloader.val_dataloader()
  # get the batch
  batch = next(iter(val_loader))
  y_hat = model(batch[0])
  # if any of the values is -10 in the batch then denormalize the coefficients
  if -10 in batch[0]:
    # denormalize the coefficients
    print("Denormalizing the coefficients")
    print(f"Mean: {dataloader.mean}")
    print(f"Std: {dataloader.std}")
    y_hat = (y_hat * dataloader.std) + dataloader.mean
    batch[1] = (batch[1] * dataloader.std) + dataloader.mean
  
  print(f"shape of the target maps: {batch[1].shape}")
  print(f"shape of the predicted maps: {y_hat.shape}")
  # return the maps with the values outside the circle set to -5
  mask = batch[1] < 0
  print(f"mask shape: {mask[0].shape}")
  batch[1][mask] = -10
  y_hat[mask] = -10
  return batch[1], y_hat

def plot_maps_for_loop(em, em_hat, index, version_num, dataloader):
  '''This function uses the generated emissivity maps and selects just one of 
   them through the index variable. It then plots the maps side by side and 
   their mean squared difference. It eventually saves the figure as a png file.
  '''
  # check if the folder for the saved figures exists
  if not os.path.exists(f"../plots/maps/version_{version_num}"):
    os.makedirs(f"../plots/maps/version_{version_num}")
  # select one of the maps from the batches em and em_hat
  em_map = em[index].detach().numpy().reshape(110, 110)
  em_hat_map = em_hat[index].detach().numpy().reshape(110, 110)
  # compute the absolute difference
  diff_map = np.abs(em_map - em_hat_map)
  # get the coordinates 
  x_emiss = dataloader.dataset.dataset.x_emiss
  y_emiss = dataloader.dataset.dataset.y_emiss
  # plot the maps side by side
  fig, axs = plt.subplots(1, 3, figsize=(15, 5))
  im0 = axs[0].imshow(em_hat_map, cmap='inferno', interpolation='nearest', extent=[x_emiss.min(), x_emiss.max(), y_emiss.min(), y_emiss.max()])
  axs[0].set_title("Model map")
  axs[0].invert_yaxis()
  # plot the colorbar rescaled by 60%
  fig.colorbar(im0, ax=axs[0], shrink=0.6)
  im1 = axs[1].imshow(em_map, cmap='inferno', interpolation='nearest', extent=[x_emiss.min(), x_emiss.max(), y_emiss.min(), y_emiss.max()])
  axs[1].set_title("Precomputed map")
  axs[1].invert_yaxis()
  fig.colorbar(im1, ax=axs[1], shrink=0.6)
  im2 = axs[2].imshow((diff_map/np.max(em_map))*100, cmap='inferno', interpolation='nearest', extent=[x_emiss.min(), x_emiss.max(), y_emiss.min(), y_emiss.max()])
  axs[2].set_title("Difference map (%)")
  axs[2].invert_yaxis()
  fig.colorbar(im2, ax=axs[2], shrink=0.6)
  # save the figure
  fig.savefig(f"../plots/maps/version_{version_num}/maps_{index}.png")
  plt.close()

if __name__ == "__main__":
  model = TomoModel(config.INPUTSIZE, config.LEARNING_RATE)
  version_num = 92 # version number of the model (74 the big one)
  assert os.path.exists(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt"), "The model does not exist"
  model.load_state_dict(torch.load(f"TB_logs/my_Tomo_model/version_{version_num}/best_model.ckpt",)['state_dict'])

  datamodule = TomographyDataModule(config.DATA_DIR, config.FILE_NAME, config.BATCH_SIZE, config.NUM_WORKERS)
  datamodule.setup()

  val_loader = datamodule.val_dataloader()

  # generate the maps
  em, em_hat = plot_maps(model, datamodule)
  # plot the maps
  for i in range(len(em)):
    plot_maps_for_loop(em, em_hat, i, version_num, val_loader)


  # Calculate the mean value of the target emissivity of all the validation dataset (excluding the values < 0)
  all_targets = []
  for batch in val_loader:
      targets = batch[1].detach().numpy()
      targets = targets[targets >= 0]
      all_targets.extend(targets)
  mean_target_emissivity = np.mean(all_targets)
  print(f"Mean value of the target emissivity (excluding values < 0): {mean_target_emissivity}")

