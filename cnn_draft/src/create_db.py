from scipy.io import readsav
import numpy as np
import os
import config
from scipy.interpolate import BSpline, splrep 
from tqdm import tqdm
from time import sleep

# The finesse of the Bessel functions is set to 2**6, that is 64 points and it is
# hard coded in the sample_dtype, if this were to change the dtype would have to be
# changed as well!!!
# define a sample of an element in the database
sample_dtype = np.dtype(
    [
        ('label', np.str_, 10), # Label of the sample (ex: 30810_0140)
        ('shot', np.int32), # Shot number (ex: 30810)
        ('time', np.float32), # Time of the shot (ex: 0.0140)
        ('prel', np.float32, (92,)), # Relative coordinates of the detectors
        ('data', np.float32, (92,)), # data in the bright struct the detectors (Line of sight)
        ('data_err', np.float32, (92,)), # error on data in the bright struct (error on line of sight)
        # Coordinates of the lines of sight, to be implemented
        # ('p', np.float32, (106,)), # impact parameters of the detectors l.o.s.
        # ('phi', np.float32, (106,)), # poloidal angles of the detectors l.o.s.
        
        # # THE SIZE OF THESE ARRAYS IS TO BE DETERMINED
        # ('data_vert', np.float32, (106,)), # Brightness of the vertical detectors
        # ('error_vert', np.float32, (106,)), # Error on the brightness of the vertical detectors
        # ('data_hor', np.float32, (106,)), # Brightness of the horizontal detectors
        # ('error_hor', np.float32, (106,)), # Error on the brightness of the horizontal detectors
        
        ('target', np.float32, (21,)), # These are the coefficients of the expansion
        ('emiss', np.float32, (110,110)), # Emissivity in the mesh
        ('rec_emiss', np.float32, (64,64)), # Reconstructed emissivity in the mesh
        ('x_emiss', np.float32, (110,)), # Vector of x coordinates for the mesh
        ('y_emiss', np.float32, (110,)), # Vector of y coordinates for the mesh
        ('majr', np.float32), # Major radius of the machine (for rfx-mod is 2m)
        ('minr', np.float32), # Minor radius of the plasma (for rfx-mod is 0.459m)
        ('theta', np.float32, (64,)), # Vector of angles for the mesh
        ('radii', np.float32, (64,)), # Vector of radii for the mesh
        # Amplitudes and phases of the m = 1, n = -7 to -24 modes, b_tor are the
        # toroidal modes, b_rad are the radial modes, phi_tor are the phases of
        # the toroidal magnetic in the poloidal plane computed at 202.5 degrees,
        # that is the position of the soft X-ray detectors
        ('b_tor', np.float32, (24,)), # amplitude of the toroidal modes 
        ('b_rad', np.float32, (24,)), # amplitude of the radial modes
        ('phi_tor', np.float32, (24,)), # phases of the toroidal mode
    ]    
  )

def maximum_los():
    '''
    This function reads all the .sav files in the directory and returns the 
    maximum number of lines of sight in the database
    '''
    # Maximum number of lines of sight
    dir = "/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/sav_files/"
    n_max = 0
    for file in os.listdir(dir):
        if file.endswith('.sav'):
            datum = readsav(dir + file)
            times = len(datum['st']['bright'][0]['data'][0])
            n_tot = np.max([len(datum['st']['bright'][0]['data'][0][i]) for i in range(times)])
            if n_tot > n_max:
                n_max = n_tot
    print(f'The maximum number of lines of sight in the db is {n_tot}')

def compute_bessel(m = 2, l = 7, finesse = 2**6):
    """
    Computes the Bessel functions J0 and J1 at specific radii and their corresponding zeros.

    Parameters:
        m (int, optional): The number of Bessel functions to compute zeros for. Defaults to 2.
        l (int, optional): The number of zeros to compute for each Bessel function. Defaults to 7.
        finesse (int, optional): The number of points to compute for the radii and theta arrays. Defaults to 2**6.

    Returns:
        tuple: A tuple containing:
            - J0_xr (numpy.ndarray): The values of the Bessel function J0 at the computed zeros times the radii.
            - J1_xr (numpy.ndarray): The values of the Bessel function J1 at the computed zeros times the radii.
            - radii (numpy.ndarray): The array of radii values ranging from 0 to 1.
            - theta (numpy.ndarray): The array of angular values ranging from 0 to 2π.
    """
    from scipy.special import j0, j1, jn_zeros
    ### 1. Compute the zeros of the Bessel functions Jml
	# Stores the zeros
    zeros = np.array([jn_zeros(_m, l) for _m in range(m)])
	# Adds the first trivial zero to the zeros of the bessel function Jml
    zero = np.zeros(1)
	# Concatenate the zeros[1] minus the last element to the array zero
    zero = np.concatenate((zero, zeros[1, :-1]))
    zeros[1] = zero

    ### 2. Compute the values of the Bessel functions J0 and J1 at the zeros times the radius
    radii = np.linspace(0,1,finesse)
    theta = np.linspace(0, 2*np.pi, finesse)
    J0_xr = np.array([j0(zeros[0]*r) for r in radii])
    J1_xr = np.array([j1(zeros[1]*r) for r in radii])
    return J0_xr, J1_xr, radii, theta


def reconstruct_polar_emissivity(J0_xr, J1_xr, theta, coeff):
    """
    Compute the polar emissivity profile from the provided data and coefficients.

    This function reconstructs a 2D emissivity profile in polar coordinates using 
    the provided expansion coefficients and basis functions. The emissivity is 
    calculated as a combination of the zeroth and first-order terms in the expansion.

    Parameters:
    -----------
    J0_xr : numpy.ndarray
        The zeroth-order basis function values.
    J1_xr : numpy.ndarray
        The first-order basis function values.
    theta : numpy.ndarray
        Array of angular coordinates (in radians) for the polar grid.
    coeff : numpy.ndarray
        Expansion coefficients, expected to be a 1D array containing three parts:
        - a0cl: Coefficients for the zeroth-order term.
        - a1cl: Coefficients for the cosine term of the first-order expansion.
        - a1sl: Coefficients for the sine term of the first-order expansion.

    Returns:
    --------
    numpy.ndarray
        A 2D array representing the emissivity profile in polar coordinates, 
        where the rows correspond to the radial positions and the columns 
        correspond to the angular positions.
    """
    # Take the expansion coefficients from the structure
    a0cl, a1cl, a1sl = np.split(coeff, 3)

	# Now compute the 2D emissivity profile in polar coordinates
	# Create an empty list to store the emissivity profile
    emissivity_profile = []
	# Iterate over the angles
    for t in theta:
        emissivity_profile.append((np.dot(J0_xr, a0cl)) + (np.dot(J1_xr, a1cl)*np.cos(t)) + (np.dot(J1_xr, a1sl)*np.sin(t)))
	# Convert the list to a numpy array
    emissivity_profile = np.array(emissivity_profile)
    # transpose the array
    emissivity_profile = emissivity_profile.T
    return emissivity_profile
    
def read_tomography(file): 
    dir = "/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/sav_files/"
    # Load the data from the .sav file
    try:
        datum = readsav(dir + file)
        print(f'file {dir + file} loaded successfully')
    except FileNotFoundError:
        print(f'File {dir + file} not found')
        return None
    # Compute the Bessel functions and their zeros once for all
    J0_xr, J1_xr, radii, theta = compute_bessel()
    st_e = datum['st_e']
    n_tot = len(st_e['t'][0])
    # Create a structured array with the sample dtype
    sample = np.empty(n_tot, dtype=sample_dtype)
    if len(datum['st']['bright'][0]['data'][0][0]) == 92:
        print('The data is already in the correct format')
        k = 0
        for i in range(n_tot):
            if (discriminate_data(np.sort(datum['st_e']['prel_vert'][0]),
                                 datum['st_e']['bright_vert'][0][i][np.argsort(datum['st_e']['prel_vert'][0])],
                                 datum['st_e']['err_vert'][0][i][np.argsort(datum['st_e']['prel_vert'][0])]) &
                discriminate_data(np.sort(datum['st_e']['prel_hor'][0]),
                                 datum['st_e']['bright_hor'][0][i][np.argsort(datum['st_e']['prel_hor'][0])],
                                 datum['st_e']['err_hor'][0][i][np.argsort(datum['st_e']['prel_hor'][0])])):
                tempo = st_e['t'][0][i]
                tt = np.rint(tempo*1e4)
                shot = st_e['shot'][0]
                label = r'%5d_%04d' % (shot, tt)
                sample[i - k]['label'] = label
                sample[i - k]['shot'] = shot
                sample[i - k]['time'] = tempo
                sample[i - k]['prel'] = datum['st']['bright'][0]['prel'][0]
                sample[i - k]['data'] = datum['st']['bright'][0]['data'][0][i]
                sample[i - k]['data_err'] = datum['st']['bright'][0]['err'][0][i]
                sample[i - k]['data_err'][sample[i - k]['data_err'] == 0.] = sample[i - k]['data'][sample[i - k]['data_err'] == 0.]*0.1
                sample[i - k]['target'] = datum['st']['emiss'][0]['coeff'][0][i]
                sample[i - k]['emiss'] = st_e['emiss'][0][i]
                sample[i - k]['rec_emiss'] = reconstruct_polar_emissivity(J0_xr, J1_xr, theta, datum['st']['emiss'][0]['coeff'][0][i])
            else:
                k += 1
        sample['x_emiss'] = st_e['X_EMISS'][0]
        sample['y_emiss'] = st_e['Y_EMISS'][0]
        sample['majr'] = st_e['MAJR'][0]
        sample['minr'] = st_e['radius'][0]
        sample['theta'] = theta
        sample['radii'] = radii
    else:
        print(f'Augmenting data for shot {st_e["shot"][0]}')
        k = 0
        for i in range(n_tot):
            if (discriminate_data(np.sort(datum['st_e']['prel_vert'][0]),
                                 datum['st_e']['bright_vert'][0][i][np.argsort(datum['st_e']['prel_vert'][0])],
                                 datum['st_e']['err_vert'][0][i][np.argsort(datum['st_e']['prel_vert'][0])]) &
                discriminate_data(np.sort(datum['st_e']['prel_hor'][0]),
                                 datum['st_e']['bright_hor'][0][i][np.argsort(datum['st_e']['prel_hor'][0])],
                                 datum['st_e']['err_hor'][0][i][np.argsort(datum['st_e']['prel_hor'][0])])):
                # Augment the data to have the same number of lines of sight and brightness
                # The underscore in the first argument is a placeholder for the lines of sight
                sample[i - k]['prel'], sample[i - k]['data'], sample[i - k]['data_err'] = augment_data(
                    datum['st']['bright'][0]['logical'][0],
                    datum['st']['bright'][0]['prel'][0],
                    datum['st']['bright'][0]['data'][0][i],
                    datum['st']['bright'][0]['err'][0][i]
                    )
                sample[i - k]['data_err'][sample[i - k]['data_err'] == 0.] = sample[i - k]['data'][sample[i - k]['data_err'] == 0.]*0.1
                tempo = st_e['t'][0][i]
                tt = np.rint(tempo*1e4)
                shot = st_e['shot'][0]
                label = r'%5d_%04d' % (shot, tt)
                sample[i - k]['label'] = label
                sample[i - k]['shot'] = shot
                sample[i - k]['time'] = tempo
                sample[i - k]['target'] = datum['st']['emiss'][0]['coeff'][0][i]
                sample[i - k]['emiss'] = st_e['emiss'][0][i]
                sample[i - k]['rec_emiss'] = reconstruct_polar_emissivity(J0_xr, J1_xr, theta, datum['st']['emiss'][0]['coeff'][0][i])
            else:
                k += 1
        sample['x_emiss'] = st_e['X_EMISS'][0]
        sample['y_emiss'] = st_e['Y_EMISS'][0]
        sample['majr'] = st_e['MAJR'][0]
        sample['minr'] = st_e['radius'][0]
        sample['theta'] = theta
        sample['radii'] = radii
    return sample[:i-k+1]

def augment_data(logical_array, prel_array, data_array, error_array):
    # Create new arrays to store the modified data
    new_prel = []
    new_data = []
    new_error = []

    # Coordinates of the most complete prel array
    # Still have to figure out if the most complete prel array is the same for all the shots
    logical = np.array([   3,    4,    5,    6,    7,    8,    9,   10,   12,   13,   14,
         15,   16,   17,   18,   19,   22,   23,   24,   25,   26,   27,
         28,   29,   30,   31,   32,   33,   34,   35,   36,   37,   39,
         40,   41,   42,   43,   44,   45,   46,   47,   48,   49,   50,
         51,   52,   53,   54,   55, 1002, 1003, 1004, 1005, 1006, 1007,
       1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018,
       1019, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030,
       1031, 1032, 1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041,
       1042, 1043, 1045, 1046])

    # Access the data in logical_array and compare them to the logical array
    for elem in logical:
        if elem in logical_array: # if elem is in logical_array then th data is already there and i can append it
            index = np.where(logical_array == elem)[0][0]
            new_prel.append(prel_array[index])
            new_data.append(data_array[index])
            new_error.append(error_array[index])
        else: # if elem is not in logical_array then i have to append a -1 to the data, but in the correct position
            new_prel.append(-1)
            new_data.append(-1)
            new_error.append(-1)

    # Return the modified coordinates
    return np.array(new_prel), np.array(new_data), np.array(new_error)

def get_coefficients(file):
    dir = "/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/"
    filename = os.join(dir, file)
    data = np.load(filename)
    # Now I have to compute the coefficients for each time instant in the .npy file
    pass

def discriminate_data(x, y, yerr):
    '''
    This function gets the x and y data (meaning the coordinates and the 
    brilliances) from the .sav file and returns True if the data points in the y 
    array (of the brilliance) are at distance less than a certain percentage of 
    the reconstructed value via spline (tipically 10%) from the reconstructed 
    value. At least 80% of the data points must be at a distance less than the
    threshold in order for the profile to be kept. 
    '''
    # Compute the spline
    tck = splrep(x, y, w=1/yerr, k=3, s=25)
    # Compute the reconstructed value
    y_reconstructed = BSpline(*tck)
    # Compute the threshold
    threshold = 0.1*np.max(y_reconstructed(x))
    # Compute the distance between the reconstructed value and the data points
    distance = np.abs(y - y_reconstructed(x))
    # Compute the number of points that are at a distance less than the threshold
    n_points = np.sum(distance < threshold)
    # Check if the profile is to be kept
    if n_points >= 0.8*len(y): # Treshold to clean the data
        return True
    else:
        return False

def create_db():
    '''
    This funciton reads the data directory and looks for the data.npy file, if
    the file already exists then it reads all of the .sav files in the sav_files
    directory and saves them in a .npy file named data.npy
    '''
    data_dir = config.DATA_DIR
    dir = '/home/orlandi/devel/Tomography/tomo-rfx/cnn_draft/data/sav_files/'
    file = config.FILE_NAME
    if os.path.exists(data_dir + file):
        return 
    else:
        data = []
        for f in tqdm(os.listdir(dir)):
            if f.endswith('.sav'):
                data.append(read_tomography(f))
            else:
                pass
        if len(data) != 0:
            data = np.concatenate(data)
        # Save data in npy format
        np.save(os.path.join(data_dir, file), data)
        return

if __name__ == "__main__":
    create_db()
    data = np.load(os.path.join(config.DATA_DIR, config.FILE_NAME))
    print(data.shape)
    