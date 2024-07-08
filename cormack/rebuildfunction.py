import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.io import readsav
import matplotlib.pyplot as plt
from time import time as tttt

# Define some important constants
M = 2 # Number of order of Bessel functions
L = 7 # Number of harmonics
RADII = np.linspace(0, 1, 110) # Radial coordinates
ANGLES = np.linspace(0, 2*np.pi, 360) # Angular coordinates

def get_bessel():
  zeros = np.array([jn_zeros(m, L) for m in range(M)])
  J0_xr = np.array([j0(zeros[0]*r) for r in RADII])
  J1_xr = np.array([j1(zeros[1]*r) for r in RADII])
  return J0_xr, J1_xr


def reconstruct_emissivity2(shot):
  # Load the data
  st = readsav(f'../Data/rfx_{shot}_2.sav').st # A standard name for the .sav files should be used

  # Get the data
  coeff = st['emiss'][0]['coeff'][0] # The coefficients are 21 for each time instant
  time = st['emiss'][0]['time'][0] # The time instants
  J0_xr, J1_xr = get_bessel()
  start = tttt()
  g_r_t = np.zeros((len(time), len(RADII), len(ANGLES)))
  for ind in range(len(time)):
    a0Cl, a1cl, a1sl = np.split(coeff[ind], 3)
    dot_0c = np.dot(J0_xr, a0Cl)
    dot_1c = np.dot(J1_xr, a1cl)
    dot_1s = np.dot(J1_xr, a1sl)
    # for i_ang, angle in enumerate(ANGLES):
    #   g_r_t[ind, :, i_ang] = dot_0c + dot_1c*np.cos(angle) + dot_1s*np.sin(angle)
    g_r_t[ind] = dot_0c[:,None] + dot_1c[:,None]*np.cos(ANGLES) + dot_1s[:,None]*np.sin(ANGLES)
  print(f'Time taken for: {tttt() - start} seconds')
  return g_r_t

def reconstruct_emissivity(shot):
  # Load the data
  st = readsav(f'../Data/rfx_{shot}_2.sav').st # A standard name for the .sav files should be used

  # Get the data
  coeff = st['emiss'][0]['coeff'][0] # The coefficients are 21 for each time instant
  time = st['emiss'][0]['time'][0] # The time instants

  J0_xr, J1_xr = get_bessel()
  start = tttt()
  g_r_t = np.zeros((len(time), len(RADII), len(ANGLES)))

  # Compute the dot products using matrix multiplication
  dot_0c = np.dot(J0_xr, coeff[:, 0::3].T).T
  dot_1c = np.dot(J1_xr, coeff[:, 1::3].T).T
  dot_1s = np.dot(J1_xr, coeff[:, 2::3].T).T

  # Compute the emissivity profile using broadcasting
  g_r_t = dot_0c[:, :, None] + dot_1c[:, :, None] * np.cos(ANGLES) + dot_1s[:, :, None] * np.sin(ANGLES)
  print(f'Time taken: {tttt() - start} seconds')
  return g_r_t


def plot_emissivity(emissivity):
  # Convert polar coordinates to Cartesian coordinates
  x = np.outer(RADII, np.cos(ANGLES))
  y = np.outer(RADII, np.sin(ANGLES))

  # Reshape the emissivity profile matrix to match the Cartesian coordinates
  emissivity_profile_cartesian = np.reshape(emissivity, (len(RADII), len(ANGLES)))

  

  # Plot the emissivity profile in Cartesian coordinates
  plt.pcolormesh(x, -y, emissivity_profile_cartesian)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Emissivity Profile (Cartesian Coordinates)')
  plt.show()

if __name__ == '__main__':
  # reconstruct_emissivity(30929)
  emissivities = reconstruct_emissivity(30929)
  emissivities = reconstruct_emissivity2(30929)
  print(emissivities[60, :, 0]) # (110, 120
  plot_emissivity(emissivities[60]) # Plot the emissivity profile for the first time instant
  plt.plot(emissivities[60, :, 0], color='b')
  plt.plot(np.linspace(0, -110, 110), emissivities[60, :, 180], color='b')
  plt.plot(emissivities[60, :, 90], color='g')
  plt.plot(np.linspace(0, -110, 110), emissivities[60, :, 270], color='g')
  plt.show()