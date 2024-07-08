import lightning as L
import torch
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

class MnistDataModule(L.LightningDataModule):
  def __init__(self, data_dir, batch_size, num_workers=4):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

  def prepare_data(self):
    datasets.MNIST(root=self.data_dir, 
                   train=True, 
                   download=True, 
                   )
    datasets.MNIST(root=self.data_dir, 
                   train=False, 
                   download=True,
                   )

  def setup(self, stage=None):
    entire_dataset = datasets.MNIST(
      root=self.data_dir,
      train=True,
      transform=transforms.ToTensor(),
      download=False
      )
    self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
    self.test_ds = datasets.MNIST(
      root=self.data_dir,
      train=False,
      transform=transforms.ToTensor(),
      download=False
      )
    
  def train_dataloader(self):
    return DataLoader(
      self.train_ds,
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      shuffle=True
      )
  
  def val_dataloader(self):
    return DataLoader(self.val_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=False)
  
  def test_dataloader(self):
    return DataLoader(self.test_ds,
                      batch_size=self.batch_size,
                      num_workers=self.num_workers,
                      shuffle=False)


class TomographyDataModule(L.LightningDataModule):
  def __init__(self, data_dir, batch_size, num_workers=4):
    super().__init__()
    self.data_dir = data_dir
    self.batch_size = batch_size
    self.num_workers = num_workers

  def prepare_data(self):
    # Load the dataset from the data_dir
    file = 'data.npy'
    data = np.load(self.data_dir + file)
    return data
    

  def setup(self, stage=None):
    # Split the dataset into train, validation, and test sets
    entire_dataset = self.prepare_data() # Load the dataset
    ds_len = len(entire_dataset) # Get the length of the dataset
    generator = torch.Generator().manual_seed(42) # Seed for reproducibility
    # Here the actual split takes place, 80% for training, 10% for validation, and 10% for testing
    self.train_ds, self.val_ds, self.test_ds = random_split(entire_dataset,
                                              [ds_len*0.8, ds_len*0.1, ds_len*0.1],
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
  
  