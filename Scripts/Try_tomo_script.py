# Imports
import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.io import readsav
import matplotlib.pyplot as plt
from time import time as tttt
from scipy.ndimage import map_coordinates as mp
from tqdm import tqdm
import sys

# Set numpy print_options
np.set_printoptions(precision=3, threshold=sys.maxsize)

# Define some important constants
M = 2 # Number of order of Bessel functions
L = 7 # Number of harmonics
FINESSE = 1300 # Number of points in the radial and angular coordinates
RADII = np.linspace(0, 1, FINESSE) # Radial coordinates
ANGLES = np.linspace(0, 2*np.pi, FINESSE) # Angular coordinates

def read_struct(shot):
    # Read the structure from the file
    filename = f'/home/orleans/projects/Tomography/Data/{shot}_tomo.sav'
    struct = readsav(filename)
    return struct

# Function to calculate the Bessel zeros and functions
def get_bessel():
  zeros = np.array([jn_zeros(m, L) for m in range(M)])
  # Adds a zero to the zeros of the bessel function Jml
  zero = np.zeros(1)
  # Concatenate the zeros[1] minus the last element to the array zero
  zero = np.concatenate((zero, zeros[1, :-1]))
  zeros[1] = zero
  print(zeros)
  J0_xr = np.array([j0(zeros[0]*r) for r in RADII])
  J1_xr = np.array([j1(zeros[1]*r) for r in RADII])
  return J0_xr, J1_xr

# Functions to reconstruct the emissivities, the first one uses a for cicle on 
# the time variable, the second does not

def reconstruct_emissivity(shot):
  # Load the data
  st = read_struct(shot).st # A standard name for the .sav files should be used
  radius = st['bright'][0]['radius'][0] # The machine radius

  # Get the data
  coeff = st['emiss'][0]['coeff'][0] # The coefficients are 21 for each time instant
  time = st['emiss'][0]['time'][0] # The time instants
  J0_xr, J1_xr = get_bessel()
  start = tttt()
  g_r_t = np.zeros((len(time), len(RADII), len(ANGLES)))
  for ind in tqdm(range(len(time))):
    a0Cl, a1cl, a1sl = np.split(coeff[ind], 3)
    dot_0c = np.dot(J0_xr, a0Cl)
    dot_1c = np.dot(J1_xr, a1cl)
    dot_1s = np.dot(J1_xr, a1sl)
    # for i_ang, angle in enumerate(ANGLES):
    #   g_r_t[ind, :, i_ang] = dot_0c + dot_1c*np.cos(angle) + dot_1s*np.sin(angle)
    g_r_t[ind] = dot_0c[:,None] + dot_1c[:,None]*np.cos(ANGLES) + dot_1s[:,None]*np.sin(ANGLES)
  print(f'Time taken for: {tttt() - start} seconds')
  return g_r_t/radius

def reconstruct_emissivity_no_for(shot):
  # Load the data
  st = read_struct(shot).st # A standard name for the .sav files should be used
  radius = st['bright'][0]['radius'][0] # The machine radius

  # Get the data
  coeff = st['emiss'][0]['coeff'][0] # The coefficients are 21 for each time instant
  time = st['emiss'][0]['time'][0] # The time instants

  J0_xr, J1_xr = get_bessel()
  start = tttt()
  g_r_t = np.zeros((len(time), len(RADII), len(ANGLES)))

  # Get the coefficients
  a0cl, a1cl, a1sl = np.split(coeff, 3, axis=1)

  # Compute the dot products using matrix multiplication
  dot_0c = np.dot(J0_xr, a0cl.T).T
  dot_1c = np.dot(J1_xr, a1cl.T).T
  dot_1s = np.dot(J1_xr, a1sl.T).T

  # Compute the emissivity profile using broadcasting
  g_r_t = dot_0c[:, :, None] + dot_1c[:, :, None] * np.cos(ANGLES) + dot_1s[:, :, None] * np.sin(ANGLES)
  print(f'Time taken: {tttt() - start} seconds')
  return g_r_t/radius

def reconstruct_emissivity_from_XYEMISS(shot):
  # Load the data
  data = read_struct(shot)
  st = data.st # A standard name for the .sav files should be used
  st_e = data.st_e
  radius = st['bright'][0]['radius'][0] # The machine radius
  x_emiss = (st_e['X_EMISS'][0] - st_e['MAJR'][0])/radius
  y_emiss = st_e['Y_EMISS'][0]/radius
  # Set to None the values outside the vessel radius
  # x_emiss[(x_emiss < 1) & (x_emiss > 1)] = np.nan
  # y_emiss[(y_emiss < 1) & (y_emiss > 1)] = np.nan
  # x_emiss = x_emiss[~np.isnan(x_emiss)]
  # y_emiss = y_emiss[~np.isnan(y_emiss)]
  # Make a mesh strating from these coordinates
  x_emiss, y_emiss = np.meshgrid(x_emiss, y_emiss)

  # Compute the arrays in polar coordinates
  radii = np.sqrt(x_emiss**2 + y_emiss**2)
  angles = np.arctan2(y_emiss, x_emiss)

  # Now we would like to compute the emissivity over the polar coordinates mesh
  # We will use the same coefficients as before
  coeff = st['emiss'][0]['coeff'][0] # The coefficients are 21 for each time instant
  time = st['emiss'][0]['time'][0] # The time instants
  # we cannot use the get_bessel function because the radial coordinates are not
  # the same as before
  zeros = np.array([jn_zeros(m, L) for m in range(M)])
  # Adds a zero to the zeros of the bessel function Jml
  zero = np.zeros(1)
  # Concatenate the zeros[1] minus the last element to the array zero
  zero = np.concatenate((zero, zeros[1, :-1]))
  zeros[1] = zero
  J0_xr = np.array([j0(zeros[0]*r) for r in radii.ravel()])
  J1_xr = np.array([j1(zeros[1]*r) for r in radii.ravel()])
  J0_xr = J0_xr.reshape(len(radii), len(radii[0]), len(zeros[0]))
  J1_xr = J1_xr.reshape(len(radii), len(radii[0]), len(zeros[1]))
  g_r_t = np.zeros((len(time), len(radii), len(angles)))
  for ind in tqdm(range(len(time))):
    a0Cl, a1cl, a1sl = np.split(coeff[ind], 3)
    dot_0c = np.dot(J0_xr, a0Cl)
    dot_1c = np.dot(J1_xr, a1cl)
    dot_1s = np.dot(J1_xr, a1sl)
    g_r_t[ind] = dot_0c + dot_1c*np.cos(angles) + dot_1s*np.sin(angles)
    g_r_t[ind][np.where(g_r_t[ind] < 0)] = 0
    g_r_t[ind][radii > 1.0] = -5
    g_r_t[ind][radii > 1.1] = -10
  return g_r_t/radius

# This function is used to plot the emissivity profile, it takes the emissivity 
# profile computed in polar coordinates and maps it to cartesian coordinates
# by simply connecting the sides of the square matrix in polar coordinates

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

# This function simply plots the matrix in polar coordinates as is
def plot_emissivity_polar(emissivity):
  plt.pcolormesh(emissivity)
  plt.colorbar()
  plt.xlabel('Angle')
  plt.ylabel('Radius')
  plt.title('Emissivity Profile (Polar Coordinates)')
  plt.show()

# The following function completely remaps the emissivity profile to Cartesian
# coordinates using the map_coordinates function from scipy.ndimage (interpolation)

def plot_emissivity_from_mesh(emissivity, shot):
  # Load the data from the .sav file
  data = read_struct(shot)
  dst_e = data.st_e
  x = dst_e.X_EMISS[0] - dst_e.MAJR[0]
  y = dst_e.Y_EMISS[0]

  # Reshape the emissivity profile matrix to match the Cartesian coordinates
  # emissivity_profile_cartesian = np.reshape(emissivity, (len(RADII), len(ANGLES)))

  # Plot the emissivity profile in Cartesian coordinates
  plt.pcolormesh(x, -y, emissivity)
  plt.colorbar()
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Emissivity Profile (Cartesian Coordinates)')
  plt.show()

# This function simply plots the matrix in polar coordinates as is
def plot_emissivity_polar(emissivity):
  plt.pcolormesh(emissivity)
  plt.colorbar()
  plt.xlabel('Angle')
  plt.ylabel('Radius')
  plt.title('Emissivity Profile (Polar Coordinates)')
  plt.show()


# Auxiliary function to map polar data to a cartesian plane
def polar_to_cart(polar_data, theta_step, range_step, x, y, order=1):
    # "x" and "y" are numpy arrays with the desired cartesian coordinates
    # we make a meshgrid with them
    X, Y = np.meshgrid(x, y)

    # Now that we have the X and Y coordinates of each point in the output plane
    # we can calculate their corresponding theta and range
    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()

    # Negative angles are corrected
    Tc[Tc < 0] = 360 + Tc[Tc < 0]

    # Using the known theta and range steps, the coordinates are mapped to
    # those of the data grid
    Tc = Tc / theta_step
    Rc = Rc / range_step

    # An array of polar coordinates is created stacking the previous arrays
    coords = np.vstack((Tc, Rc))

    # To avoid holes in the 360º - 0º boundary, the last column of the data
    # copied in the begining
    polar_data = np.vstack((polar_data, polar_data[-1,:]))

    # The data is mapped to the new coordinates
    # Values outside range are substituted with nans
    cart_data = mp(polar_data, coords, order=order, mode='constant', cval=-5)

    # The data is reshaped and returned
    return(cart_data.reshape(len(y), len(x)))

def compute_error(emissivity_sav, emissivity_profile, th_step, r_step, x, y):
  error = .0
  for i in tqdm(range(len(emissivity_sav))):
    cart_data = polar_to_cart(emissivity_profile[i].T, th_step, r_step, x, y, order=1)
    error += np.sum(np.abs(emissivity_sav[i][40:60, 40:60] - cart_data[40:60, 40:60]))/len(emissivity_sav[i][40:60, 40:60])
  return round(error/len(emissivity_sav), 6)

def show_error(emissivity_sav, emissivity_profile, shot):
  data = read_struct(shot)
  dst_e = data.st_e
  x = dst_e.X_EMISS[0] - dst_e.MAJR[0]
  y = dst_e.Y_EMISS[0]
  plt.pcolormesh(x, -y, (emissivity_sav - emissivity_profile))
  plt.colorbar(shrink=0.6)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.title('Error')
  plt.show()
  plt.contour(x, -y, (emissivity_sav - emissivity_profile)*100, levels=20)
  plt.colorbar(shrink=0.6)
  plt.xlabel('x')
  plt.ylabel('y')
  plt.xlim(-0.3, 0.3)
  plt.ylim(-0.3, 0.3)
  plt.title('Error (%)')
  plt.show()

def main():
  # Load the data
  shot = 30929
  data = read_struct(shot)
  emissivity_profile = reconstruct_emissivity(shot)

  # Select the time index to plot
  t_ind = 0

  # Select the st_e structure from the data
  dst_e = data.st_e

  # We create the x and y axes of the output cartesian data
  x = dst_e.X_EMISS[0] - dst_e.MAJR[0]
  y = dst_e.Y_EMISS[0]

  # We call the mapping function assuming 3.6 degree of theta step and r/100
  # of range step. 
  th_step = 360/FINESSE
  r_step = data['st']['bright'][0]['radius'][0]/FINESSE
  cart_data = polar_to_cart(emissivity_profile[t_ind].T, th_step, r_step, x, y, order=1)
  # Here the erro is compute and printed
  print("Distance between the two matrices: ",
         compute_error(dst_e.EMISS[0], emissivity_profile, th_step, r_step, x, y))

  # We plot the data in cartesian coordinates
  fig, ax = plt.subplots(1,2, figsize=(10,5))
  im0 = ax[0].imshow(dst_e.EMISS[0][t_ind], origin='lower')
  ax[0].set_title('Original')
  fig.colorbar(im0, ax=ax[0], shrink=0.6)
  im1 = ax[1].imshow(cart_data, origin='lower')
  ax[1].set_title('Remapped')
  fig.colorbar(im1, ax=ax[1], shrink=0.6)

  # We plot the difference between the two profiles at the diameter
  fig1, ax1 = plt.subplots(1,2, figsize=(10,5))
  im3 = ax1[0].imshow(dst_e.EMISS[0][t_ind] - cart_data, origin='lower')
  ax1[0].set_title('Difference of emissivities')
  fig1.colorbar(im3, ax=ax1[0], shrink=0.6)
  ax1[1].plot(dst_e.EMISS[0][t_ind,55,:] - cart_data[55, :])
  ax1[1].plot(dst_e.EMISS[0][t_ind,:,55] - cart_data[:, 55])
  ax1[1].set_xlim(-10, 120)
  ax1[1].set_title('Difference at the diameters')
  plt.show()

def main2():
  shot = 30929
  emissivity_profile = reconstruct_emissivity_from_XYEMISS(shot)
  print(emissivity_profile.shape)
  data = read_struct(shot)
  dst_e = data.st_e
  # extract the emissivity matrix
  emissivity_sav = dst_e.EMISS[0]
  i = 300

  plot_emissivity_from_mesh(emissivity_profile[i], shot)
  plt.imshow(emissivity_sav[i])
  plt.colorbar()
  plt.show()
  show_error(read_struct(shot).st_e.EMISS[0][i], emissivity_profile[i], shot)
  plt.plot(emissivity_profile[i][55])
  plt.plot(emissivity_sav[i][55])
  plt.show()

if __name__ == '__main__':
  main2()

  # Bidirectional GRU
  # Transformer (BERT)
  