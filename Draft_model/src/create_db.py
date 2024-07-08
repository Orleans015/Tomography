from scipy.io import readsav
import numpy as np
import pandas as pd
import os
import h5py

# define a sample of an element in the database
sample_dtype = np.dtype(
    [
        ('label', 'S10'), # Label of the sample b'shot_time'
        ('data', np.float32, (65,)), # Must know how many choords are in total
        ('target', np.int32, (21,)), # These are the coefficients of the expansion
        ('emiss', np.float32, (110,110)), # Emissivity in the mesh
        ('x_emiss', np.float32, (110,)), # Vector of x coordinates for the mesh
        ('y_emiss', np.float32, (110,)), # Vector of y coordinates for the mesh
        ('majr', np.float32), # Major radius of the plasma

    ]    
  )

def read_tomography(file): 
    dir = "/home/orleans/projects/Tomography/Data/"
    # Load the data from the .sav file
    try:
        datum = readsav(dir + file)
    except FileNotFoundError:
        print(f'File {dir + file} not found')
        return None
    
    n_tot = len(datum['st']['emiss'][0]['time'][0])
    # Create a structured array with the sample dtype
    sample = np.empty(n_tot, dtype=sample_dtype)
    for i in range(n_tot):
        tt = np.rint(datum['st']['emiss'][0]['time'][0][i]*1e4)
        shot = datum['st']['bright'][0]['shot'][0]
        label = r'%5d_%04d' % (shot, tt)
        sample[i]['label'] = label
        sample[i]['data'] = datum['st']['bright'][0]['data'][0][i]
        sample[i]['target'] = datum['st']['emiss'][0]['coeff'][0][i]
        sample[i]['emiss'] = datum['st_e']['emiss'][0][i]
    sample['x_emiss'] = datum['st_e']['X_EMISS'][0]
    sample['y_emiss'] = datum['st_e']['Y_EMISS'][0]
    sample['majr'] = datum['st_e']['MAJR'][0]
    return sample

def create_db():
    # Load the data from all the .sav file in the directory
    dir = '/home/orleans/projects/Tomography/Data'
    data = []
    for file in os.listdir(dir):
        if file.endswith('.sav'):
            data.append(read_tomography(file))
        else:
            pass
    data = np.concatenate(data)
    # Save data in npy format
    np.save('/home/orleans/projects/Tomography/Data/data.npy', data)
    # save data in .hdf5 format using h5py
    with h5py.File('/home/orleans/projects/Tomography/Data/data.hdf5', 'w') as f:
        f.create_dataset('data', data=data)

    return data

if __name__ == "__main__":
    df = create_db()
    print(df.dtype)
