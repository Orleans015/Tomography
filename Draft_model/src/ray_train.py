import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
import numpy as np
import os
from tqdm import tqdm
import torch.utils
from torch.utils.data import DataLoader, random_split
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayDDPStrategy, RayLightningEnvironment, RayTrainReportCallback, prepare_trainer

INPUTSIZE = 92
OUTPUTSIZE = 21
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
NUM_EPOCHS = 100

# Dataset
DATA_DIR = '/home/orlandi/devel/Tomography/tomo-rfx/Draft_model/data/'
FILE_NAME = 'data_clean.npy'
NUM_WORKERS = 8

# Compute related
ACCELERATOR = 'gpu'
DEVICES = [0]
PRECISION = '16-mixed'

# Define the model class
class TomoModel(L.LightningModule):
  def __init__(self, inputsize, outputsize, config):
    super().__init__()
    self.lr = config["lr"]  # Define the learning rate
    self.hidden_layer1_size = config["hidden_layer1_size"]  # Define the size of the first hidden layer
    self.hidden_layer2_size = config["hidden_layer2_size"]  # Define the size of the second hidden layer
    self.hidden_layer3_size = config["hidden_layer3_size"]  # Define the size of the third hidden layer
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.
    self.net = nn.Sequential(
        nn.Linear(inputsize, self.hidden_layer1_size),  # Define a linear layer with input size and output size
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(self.hidden_layer1_size, self.hidden_layer2_size),  # Define another linear layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(self.hidden_layer2_size, self.hidden_layer3_size),  # Define another linear layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(self.hidden_layer3_size, outputsize)  # Define Final linear layer with output size
    )
    self.loss_rate = 0.2  # Define the loss rate
    self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
    self.best_val_loss = torch.tensor(float('inf'))  # Initialize the best validation loss
    self.mse = torchmetrics.MeanSquaredError()  # Define Mean Squared Error metric
    self.mae = torchmetrics.MeanAbsoluteError() # Define Root Mean Squared Error metric
    self.r2 = torchmetrics.R2Score()  # Define R2 score metric, using the multioutput parameter the metric will return an array of R2 scores for each output
    self.training_step_outputs = []  # Initialize an empty list to store training step outputs
    
  def forward(self, x):
    x = self.net(x)  # Pass the input through the network
    return x
  
  def training_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train_loss': loss,
                   'train_mse': mse,
                   'train_mae': mae,
                   'train_r2': r2,
                   'sync_dist': True,
                   },
                   on_step=False, on_epoch=True, prog_bar=False
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val_loss': loss,
                   'val_mse': mse,
                   'val_mae': mae,
                   'val_r2': r2,
                   'sync_dist': True,
                   },
                   on_step=False, on_epoch=True, prog_bar=False
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test_loss': loss,
                   'test_mse': mse,
                   'test_mae': mae,
                   'test_r2': r2,
                   'sync_dist': True,
                   },
                   on_step=False, on_epoch=True, prog_bar=False
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    # # Compute the emissivity maps using the coefficients
    #em, em_hat = self.calc_em(batch, y_hat) # Compute the tomography map using the coefficients
    
    # Compute the loss using the y_hat (prediction) and target
    loss = self.loss_fn(y_hat, y)

    # The following lines are used to set the value of the loss_rate variable (0.2)
    # print(f"MSE on the coefficients vector: {self.loss_fn(y_hat, y)}")
    # print(f"MSE on the emissivity maps: {self.loss_fn(em_hat, em)}")

    # # This is the loss function computed as a weighted sum of the loss on the 
    # # coefficients vector and the loss on the emissivity maps
    # loss = ((1 - self.loss_rate) * self.loss_fn(y_hat, y)) + (self.loss_rate * self.loss_fn(em_hat, em))  # Compute the loss using the y_hat (prediction) and target
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate

# Deffine the Dataset and DataModule classes
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
    self.x_emiss = bloated_dataset['x_emiss'][0]
    self.y_emiss = bloated_dataset['y_emiss'][0]
    self.majr = bloated_dataset['majr'][0]
    self.minr = bloated_dataset['minr'][0]
    self.b_tor = bloated_dataset['b_tor']
    self.b_rad = bloated_dataset['b_rad']
    self.phi_tor = bloated_dataset['phi_tor']

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    # Return the data and target for the given index, the other data is used only 
    # if the dataset is used for the computation of the emissivity maps.
    return self.data[idx], self.target[idx] #, self.j0, self.j1, self.em, self.em_hat, self.radii, self.angles
class TomographyDataModule(L.LightningDataModule):
  def __init__(self, data_dir, file_name, batch_size, num_workers=4):
    super().__init__()
    self.data_dir = data_dir
    self.file_name = file_name
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.mean = 0.
    self.std = 0.

  def prepare_data(self):
    '''
    The prepare_data method is called only once and on a single GPU. Its main
    purpose is to download the dataset and perform any necessary transformations
    to the data (i.e. download, IO, etc). The prepare_data method is called 
    before the setup method.
    '''
    # Generate the dataset if it does not exist yet
    data_dir = "/home/orlandi/devel/Tomography/tomo-rfx/Draft_model/data"
    file = "data_clean.npy"
    dir = "../data/sav_files/"
    if os.path.exists("/home/orlandi/devel/Tomography/tomo-rfx/Draft_model/data/data_clean.npy"):
        return 
    else:
        print("cannot execute the code")
        return

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
    # normalize the data but only the ones that are greater than 0
    mask_pos = entire_dataset.data > 0 # Get the data containing real values from the diagnostic
    mask_neg = entire_dataset.data < 0 # Get the data not containing real values from the diagnostic
    entire_dataset.data[mask_pos] = (entire_dataset.data[mask_pos] - entire_dataset.data[mask_pos].mean()) / entire_dataset.data[mask_pos].std() # Normalize the real values
    entire_dataset.data[mask_neg] = -10 # Set the non-real values to -10
    # normalize the targets
    self.mean = entire_dataset.target.mean()
    self.std = entire_dataset.target.std()
    entire_dataset.target = (entire_dataset.target - self.mean) / self.std
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
  
default_config = {
    "hidden_layer1_size": 256,
    "hidden_layer2_size": 256,
    "hidden_layer3_size": 256,
    "lr": 1e-3,
}


def train_function(config):
    dm = TomographyDataModule(
        data_dir=DATA_DIR,
        file_name=FILE_NAME,
        batch_size=config["batch_size"],
        num_workers=NUM_WORKERS,
    )

    model = TomoModel(
        inputsize=INPUTSIZE,
        outputsize=OUTPUTSIZE,
        config=config,
    )

    trainer = L.Trainer(
        accelerator="auto",
        devices="auto",
        strategy=RayDDPStrategy(),
        precision=PRECISION,
        callbacks=[
            RayTrainReportCallback(),
        ],
        plugins=RayLightningEnvironment(),
        enable_progress_bar=False,
    )

    trainer = prepare_trainer(trainer)
    trainer.fit(model, datamodule=dm)

search_space = {
    "hidden_layer1_size": tune.choice([32, 64, 128]),
    "hidden_layer2_size": tune.choice([32, 64, 128]),
    "hidden_layer3_size": tune.choice([32, 64, 128]),
    "lr" : tune.loguniform(5e-5, 1e-2),
    "batch_size": tune.choice([16, 32, 64, 128, 256]),
}

num_epochs = 20
num_samples = 100

# scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

# The following code is to be implemented if training on multiple GPUs
scaling_config = ScalingConfig(
    num_workers=1,
    use_gpu=True,
    resources_per_worker={"CPU": 1, "GPU": 1}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
)

ray_trainer = TorchTrainer(
    train_function,
    scaling_config=scaling_config,
    run_config=run_config,
)

def tune_tomo_asha(num_samples=10):
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        ray_trainer,
        param_space={"train_loop_config": search_space},
        tune_config=tune.TuneConfig(
            metric="val_loss",
            mode="min",
            num_samples=num_samples,
            scheduler=scheduler,
        ),
    )
    return tuner.fit()

if __name__ == "__main__":
    results = tune_tomo_asha(num_samples=num_samples)
    best_trial = results.get_best_result("val_loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))

# Best trial config: {'train_loop_config': {'hidden_layer1_size': 1024, 'hidden_layer2_size': 1024, 'hidden_layer3_size': 128, 'lr': 7.864154774069017e-05, 'batch_size': 32}}