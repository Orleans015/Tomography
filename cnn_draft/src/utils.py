import os
from scipy.special import j0, j1, jn_zeros
import numpy as np
import lightning as L
import torch
import config
from dataset import TomographyDataset

def compute_bessel_n_mesh(minr, majr, x_grid, y_grid):
  # set relative coordinates
  x_grid = (x_grid - majr)/minr
  y_grid = y_grid/minr
  # define the grid
  x_grid, y_grid = np.meshgrid(x_grid, y_grid)
  # change the grid to polar coordinates
  radii  = np.sqrt(x_grid**2 + y_grid**2)
  angles = np.arctan2(y_grid, x_grid)

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
  em = np.zeros((len(radii), len(angles)))
  em_hat = np.zeros((len(radii), len(angles)))
    
  return J0_xr, J1_xr, em, em_hat, radii, angles