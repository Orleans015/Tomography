import lightning as L
import torch
import numpy as np
import os
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import create_db as c_db

# Togliere la dataset class e usare direttamente il datamodule?
class TomographyDataset(torch.utils.data.Dataset):
  def __init__(self, data_dir, file_name):
    self.data_dir = data_dir
    self.file_name = file_name
    # Load the dataset
    bloated_dataset = np.load(
      os.path.join(self.data_dir, self.file_name),
      allow_pickle=True
      )
    # Here i have the entire dataset in a dictionary with keys 'data' and 'target'
    # Data and target are the input and output of the model
    self.data = torch.Tensor(bloated_dataset['data'])
    self.target = torch.Tensor(bloated_dataset['target'])
    # The following data is not used duing training, but it is useful for visualization
    self.labels = bloated_dataset['label']
    self.shots = bloated_dataset['shot']
    self.time = bloated_dataset['time']
    self.dataerr = bloated_dataset['data_err']
    self.emiss = bloated_dataset['emiss']
    self.x_emiss = bloated_dataset['x_emiss']
    self.y_emiss = bloated_dataset['y_emiss']
    self.majr = bloated_dataset['majr']
    self.minr = bloated_dataset['minr']
    self.b_tor = bloated_dataset['b_tor']
    self.b_rad = bloated_dataset['b_rad']
    self.phi_tor = bloated_dataset['phi_tor']

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
    '''
    The prepare_data method is called only once and on a single GPU. Its main
    purpose is to download the dataset and perform any necessary transformations
    to the data (i.e. download, IO, etc). The prepare_data method is called 
    before the setup method.
    '''
    # Generate the dataset if it does not exist yet
    c_db.create_db()

  def setup(self, stage=None):
    '''
    The setup method is called before the dataloaders are created. Its main 
    purpose is to:
    - Load the dataset from the data directory. ()
    - Specifically load only the brilliance and 
    - split the dataset into training, validation, and test sets.
    '''
    # Load the dataset
    entire_dataset = TomographyDataset(self.data_dir, self.file_name)
    # Split the dataset into train, validation, and test sets
    generator = torch.Generator().manual_seed(42) # Seed for reproducibility
    # Here the actual split takes place, 80% for training, 10% for validation, and 10% for testing
    self.train_ds, self.val_ds, self.test_ds = random_split(entire_dataset,
                                              [0.8, 0.1, 0.1],
                                              generator=generator)

  def train_dataloader(self):
    # Return the train dataloader
    return DataLoader(self.train_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=True)

  def val_dataloader(self):
    # Return the validation dataloader
    return DataLoader(self.val_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=False)

  def test_dataloader(self):
    # Return the test dataloader
    return DataLoader(self.test_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=False)
  