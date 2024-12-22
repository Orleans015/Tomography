import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
from scipy.special import j0, j1, jn_zeros
import utils

class Reshape(nn.Module):
  def __init__(self, shape):
    super(Reshape, self).__init__()
    self.shape = shape

  def forward(self, x):
    return x.view(*self.shape)

class TomoModel(L.LightningModule):
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.
    self.linear = nn.Sequential(
        nn.Linear(inputsize, 256),  # Define a linear layer with input size and output size
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(256, 16*11*11),  # Define another linear layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        Reshape([-1, 16, 11, 11]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
        nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(4, 4, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(4, 4, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(4, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
    )

    self.net = nn.Sequential(
        self.linear,  # Add the linear layer to the network
        self.anti_conv,  # Add the convolutional layer to the network
    )

    self.loss_rate = 0.2  # Define the loss rate
    #self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
    # use the cross entropy loss for the reconstruction of the emissivity maps
    self.loss_fn = nn.MSELoss()  # Define the loss function as CrossEntropyLoss
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
    # mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train_loss': loss,
                  #  'train_mse': mse,
                   'train_mae': mae,
                   'train_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    # mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val_loss': loss,
                  #  'val_mse': mse,
                   'val_mae': mae,
                   'val_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test_loss': loss,
                  #  'test_mse': mse,
                   'test_mae': mae,
                   'test_r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
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
  