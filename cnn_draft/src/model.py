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

class TomoModelv82(L.LightningModule): # This is version 82 of the model, ha una loss che è un ordine di grandezza maggiore rispetto al modello sotto (v81)
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    def compute_block(input_channels, output_channels):
      return nn.Sequential(
        nn.ConvTranspose2d(input_channels, input_channels, kernel_size=2, stride=2, padding=0),
        nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
        nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(),
      )

    self.lr = learning_rate
    self.side = 11
    self.linear = nn.Sequential(
        nn.Linear( inputsize, self.side*self.side),
        nn.LeakyReLU(),
        Reshape([-1, 1, self.side, self.side]),
    )        

    self.tconv = nn.Sequential(
      compute_block(1, 4),
      compute_block(4, 8),
      compute_block(8, 4),
      compute_block(4, 1),
      nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0),
    )

    self.net = nn.Sequential(
        self.linear,
        self.tconv,
    )

    self.loss_fn = nn.L1Loss()
    self.best_val_loss = torch.tensor(float('inf'))
    self.mse = torchmetrics.MeanSquaredError()
    self.r2 = torchmetrics.R2Score()
    self.training_step_outputs = []


  def compute_tconv_output_size(self, input_size, kernel_size, stride, padding):
    return (input_size - 1) * stride - 2 * padding + kernel_size
  
  def compute_conv_output_size(self, input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1

  def forward(self, x):
    x = self.net(x)
    return x
  
  def training_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mse = self.mse(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.training_step_outputs.append(loss.detach().cpu().numpy())
    self.log_dict({'train/loss': loss,
                   'train/mse': mse,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mse = self.mse(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss

  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)
    mse = self.mse(y_hat, y)
    r2 = self.r2(y_hat.view(-1), y.view(-1))
    self.log_dict({'test/loss': loss,
                   'test/mse': mse,
                   'test/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss

  def _common_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch
    x = x.reshape(x.size(0), -1)
    y_hat = self(x)
    preds = torch.argmax(y_hat, dim=1)
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)
  
class TomoModelNew(L.LightningModule):
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    self.side = 11
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.
    self.linear = nn.Sequential(
        nn.Linear(inputsize, self.side**2),  # Define a linear layer with input size and output size
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(self.side**2, self.side**2),  # Define another linear layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        Reshape([-1, 1, self.side, self.side]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
        nn.ConvTranspose2d(1, 5, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(5, 10, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(10, 5, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(5, 5, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 5, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
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
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train/loss': loss,
                   'train_mse': mse,
                   'train/mae': mae,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/mae': mae,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                   'test_mse': mse,
                   'test/mae': mae,
                   'test/r2': r2,
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
  
class TomoModelv77(L.LightningModule): # TomoModelv77
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    self.side = 11
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.
    self.linear = nn.Sequential(
        nn.Linear(inputsize, 2420),  # Define a linear layer with input size and output size
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Linear(2420, 2420),  # Define another linear layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        Reshape([-1, 20, 11, 11]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
        nn.ConvTranspose2d(20, 20, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(20, 20, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(20, 10, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(10, 10, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(10, 5, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.ConvTranspose2d(5, 5, kernel_size=2, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
        nn.Conv2d(5, 5, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 5, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
        nn.LeakyReLU(),  # Apply ReLU activation function
        nn.Conv2d(5, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
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
    self.log_dict({'train/loss': loss,
                  #  'train_mse': mse,
                  'train/mae': mae,
                  'train/r2': r2,
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
    self.log_dict({'val/loss': loss,
                  #  'val/mse': mse,
                  'val/mae': mae,
                  'val/r2': r2,
                  },
                  on_step=False, on_epoch=True, prog_bar=True
                  )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                  #  'test_mse': mse,
                  'test/mae': mae,
                  'test/r2': r2,
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
  
class TomoModelv92(L.LightningModule): # This is the v84 of the model, added a block of transposed convolution and two convolutional layers, moreover added a conv. layer of kernel 5
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    self.side = 10
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.    
    self.linear = nn.Sequential(
        nn.Linear(inputsize, self.side**2),  # Define another linear layer
        Reshape([-1, 1, self.side, self.side]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
      nn.ConvTranspose2d(1, 5, kernel_size=15, stride=4, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(5, 5, kernel_size=11, stride=4, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(5, 10, kernel_size=13, stride=4, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(10, 10, kernel_size=9, stride=3, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(10, 20, kernel_size=11, stride=3, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(20, 20, kernel_size=9, stride=3, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(20, 10, kernel_size=11, stride=3, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(10, 10, kernel_size=7, stride=2, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(10, 5, kernel_size=9, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding      
      nn.Conv2d(5, 5, kernel_size=5, stride=2, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(5, 1, kernel_size=7, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(1, 1, kernel_size=7, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0),  # Define another convolutional layer
    )

    self.net = nn.Sequential(
        self.linear,  # Add the linear layer to the network
        self.anti_conv,  # Add the convolutional layer to the network
    )

    self.loss_rate = 0.2  # Define the loss rate
    # self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
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
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train/loss': loss,
                   'train_mse': mse,
                   'train/mae': mae,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/mae': mae,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                   'test_mse': mse,
                   'test/mae': mae,
                   'test/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    # mask = y > 0
    # loss = self.loss_fn(y_hat[mask], y[mask]) # Non benissimo la loss solo sul cerchio 
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-6)
    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
  
class TomoModelR(L.LightningModule): # This is the v84 of the model, added a block of transposed convolution and two convolutional layers, moreover added a conv. layer of kernel 5
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    self.side = 10
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.    
    self.linear = nn.Sequential(
        nn.Linear(inputsize, self.side**2),  # Define another linear layer
        Reshape([-1, 1, self.side, self.side]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
      nn.ConvTranspose2d(1, 5, kernel_size=15, stride=4, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(5, 5, kernel_size=11, stride=4, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(5, 10, kernel_size=13, stride=4, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(10, 10, kernel_size=9, stride=3, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(10, 20, kernel_size=11, stride=3, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(20, 20, kernel_size=9, stride=3, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(20, 10, kernel_size=11, stride=3, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(10, 10, kernel_size=7, stride=2, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(10, 5, kernel_size=9, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding      
      nn.Conv2d(5, 5, kernel_size=5, stride=2, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(5, 1, kernel_size=7, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      
      nn.ConvTranspose2d(1, 1, kernel_size=7, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(1, 1, kernel_size=2, stride=1, padding=0),  # Define another convolutional layer
    )

    self.net = nn.Sequential(
        self.linear,  # Add the linear layer to the network
        self.anti_conv,  # Add the convolutional layer to the network
    )

    self.loss_rate = 0.2  # Define the loss rate
    # self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
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
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train/loss': loss,
                   'train_mse': mse,
                   'train/mae': mae,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/mae': mae,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                   'test_mse': mse,
                   'test/mae': mae,
                   'test/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    # mask = y > 0
    # loss = self.loss_fn(y_hat[mask], y[mask]) # Non benissimo la loss solo sul cerchio 
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-6)
    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

# I need a layer that reshapes the input tensor by cutting it in the right places 
# and adds padding at the beginning and end of the tensor in order to make each row
# of the matrix of the same size. The input tensor is a vector of shape (batch_size, 92)
# and the output tensor is a matrix of shape (batch_size, 5, 25) with padding between the rows

class vectomat(nn.Module):
  def __init__(self, padding=2):
    super(vectomat, self).__init__()
    self.linear = nn.Linear(92, 125)  # Map the input vector to a 5x20 matrix
    self.row_size = 25
    
  def forward(self, x):
    x0 = x[:, :16]
    x1 = x[:, 16:32]
    x2 = x[:, 32:49]
    x3 = x[:, 49:67]
    x4 = x[:, 67:92]

    # Get the padding size
    padding_list = [self.row_size - x0.size(1), self.row_size - x1.size(1), self.row_size - x2.size(1), self.row_size - x3.size(1), self.row_size - x4.size(1)]
    # generate a list of lines of sight
    los_list = [x0, x1, x2, x3, x4]

    for i, los in enumerate(los_list):
        padding_left = padding_list[i] // 2
        padding_right = padding_list[i] - padding_left
        # Pad the tensor with zeros left and right with half of the padding size
        los_list[i] = F.pad(los, (padding_left, padding_right), mode='constant', value=0)

    # Concatenate the padded tensors along the second dimension
    x = torch.cat(los_list, dim=1)
    # Reshape the tensor to have shape (batch_size, 5, 25)
    x = x.view(x.size(0), 5, self.row_size)
    return x

class TomoModel(L.LightningModule):
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate
    self.side = 10
    # Leaky ReLU activation function takes as argument the negative slope of the
    # rectifier: f(x) = max(0, x) + negative_slope * min(0, x). The default value
    # of the negative slope is 0.01.    
    self.linear = nn.Sequential(
        nn.Linear(inputsize, self.side**2),  # Define another linear layer
        nn.Linear(self.side**2, self.side**3),  # Define another linear layer
        Reshape([-1, self.side, self.side, self.side]),  # Reshape the tensor First dimension is batch size, second dimension is the number of channels, and the last two dimensions are the height and width of the tensor
    )
    self.anti_conv = nn.Sequential(
      nn.ConvTranspose2d(self.side, 2*self.side, kernel_size=3, stride=2, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.Conv2d(2*self.side, 2*self.side, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.Conv2d(2*self.side, 2*self.side, kernel_size=3, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(2*self.side, 4*self.side, kernel_size=3, stride=2, padding=0),
      nn.Conv2d(4*self.side, 4*self.side, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(4*self.side, 4*self.side, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(4*self.side, 2*self.side, kernel_size=3, stride=2, padding=0),
      nn.Conv2d(2*self.side, 2*self.side, kernel_size=5, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(2*self.side, 2*self.side, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(2*self.side, self.side, kernel_size=3, stride=2, padding=0),
      nn.Conv2d(self.side, self.side, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(self.side, self.side, kernel_size=3, stride=1, padding=0),
      nn.LeakyReLU(),
      nn.Conv2d(self.side, 1, kernel_size=2, stride=1, padding=0),
    )

    self.net = nn.Sequential(
        self.linear,  # Add the linear layer to the network
        self.anti_conv,  # Add the convolutional layer to the network
    )

    self.loss_rate = 0.2  # Define the loss rate
    # self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
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
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train/loss': loss,
                   'train_mse': mse,
                   'train/mae': mae,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/mae': mae,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                   'test_mse': mse,
                   'test/mae': mae,
                   'test/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-6)
    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}

class AddDimension(nn.Module):
	'''This class adds a dimension to the input tensor in the specified position.
	Args:
		dim (int): The dimension to add. Default is 1.		
	'''
	def __init__(self, dim=1):
		super(AddDimension, self).__init__()
		self.dim = dim

	def forward(self, x):
		return x.unsqueeze(self.dim)

class cnn(L.LightningModule):
  def __init__(self, inputsize, learning_rate):
    super().__init__()
    self.lr = learning_rate

    self.encoding_net = nn.Sequential(
      vectomat(),  # reshape the input tensor to a 5x25 matrix including padding
      # Now i want to apply a 1D convolutional layer to the input tensor, that iterates over the rows of the matrix
      nn.Conv1d(5, 5, kernel_size=5, stride=1, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.Conv1d(5, 5, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
    )

    # TODO: add linear layers after the conv layers and add them before the transposed conv layers
    # in the decoding net.
    
    # The output of the encoding net is a 5x5 matrix, now i want to apply a transposed convolutional
    # layer to the input tensor to get the 2D emissivity map of the plasma

    self.decoding_generator= nn.Sequential(
      AddDimension(1),  # Add a dimension to the input tensor
      nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define a convolutional layer with input channels, output channels, kernel size, and padding
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
      nn.LeakyReLU(),  # Apply ReLU activation function
      nn.ConvTranspose2d(1, 1, kernel_size=5, stride=1, padding=0),  # Define another convolutional layer
    )

    self.net = nn.Sequential(
        self.encoding_net,  # Add the linear layer to the network
        self.decoding_generator,  # Add the convolutional layer to the network
    )

    self.loss_rate = 0.2  # Define the loss rate
    # self.loss_fn = nn.L1Loss()  # Define the loss function as CrossEntropyLoss
    self.loss_fn = nn.MSELoss()  # Define the loss function as CrossEntropyLoss
    self.best_val_loss = torch.tensor(float('inf'))  # Initialize the best validation loss
    self.mse = torchmetrics.MeanSquaredError()  # Define Mean Squared Error metric
    self.mae = torchmetrics.MeanAbsoluteError() # Define Root Mean Squared Error metric
    self.r2 = torchmetrics.R2Score()  # Define R2 score metric, using the multioutput parameter the metric will return an array of R2 scores for each output
    self.training_step_outputs = []  # Initialize an empty list to store training step outputs

  def forward(self, x):
    x = self.net(x)  # Pass the input through the network
    print(x.shape)
    print(x.shape)
    assert False
    return x
  
  def training_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.training_step_outputs.append(loss.detach().cpu().numpy())  # Append the loss to the training step outputs list
    self.log_dict({'train/loss': loss,
                   'train_mse': mse,
                   'train/mae': mae,
                   'train/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": y_hat, "target": y}
  
  def validation_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    # calculate metrics
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'val/loss': loss,
                   'val/mse': mse,
                   'val/mae': mae,
                   'val/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, y_hat, y = self._common_step(batch, batch_idx)  # Compute loss, y_hat (prediction), and target using a common step function
    mse = self.mse(y_hat, y)  # Compute mse using the y_hat (prediction) and target
    mae = self.mae(y_hat, y)  # Compute mae using the y_hat (prediction) and target
    r2 = self.r2(y_hat.view(-1), y.view(-1))  # Compute r2score using the y_hat (prediction) and target
    self.log_dict({'test/loss': loss,
                   'test_mse': mse,
                   'test/mae': mae,
                   'test/r2': r2,
                   },
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch[0], batch[1]
    y_hat = self(x)
    loss = self.loss_fn(y_hat, y)
    return loss, y_hat, y
  
  def predict_step(self, batch, batch_idx):
    x = batch[0]
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    y_hat = self(x)  # Compute the y_hat (prediction) by passing the input through the network
    preds = torch.argmax(y_hat, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-6)
    return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val/loss"}
