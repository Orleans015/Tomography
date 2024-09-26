import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchmetrics
import lightning as L
import torchvision


class NN(L.LightningModule):
  def __init__(self, inputsize, learning_rate, outputsize):
    super().__init__()
    self.lr = learning_rate
    self.net = nn.Sequential(
        nn.Linear(inputsize, 184),  # Define a linear layer with input size and output size
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(184, 368),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(368, 736),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(736, 736),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(736, 368),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(368, 184),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(184, outputsize)  # Define Final linear layer with output size
    )
    self.loss_fn = nn.MSELoss()  # Define the loss function as CrossEntropyLoss
    self.mae = torchmetrics.MeanAbsoluteError() # Define Root Mean Squared Error metric
    self.md = torchmetrics.MinkowskiDistance(p=3)  # Define F1 score metric
    self.training_step_outputs = []  # Initialize an empty list to store training step outputs

  def forward(self, x):
    x = self.net(x)  # Pass the input through the network
    return x
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    mae = self.mae(scores, y)  # Compute mae using the scores and target
    md = self.md(scores, y)  # Compute F1 score using the scores and target
    self.training_step_outputs.append(loss)  # Append the loss to the training step outputs list
    self.log_dict({'train_loss': loss,
                   'train_mae': mae,
                   'train_md': md},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, mae, and F1 score
    return {"loss": loss, "preds": scores, "target": y}
  
  def on_train_epoch_end(self):
    avg_loss = torch.stack(self.training_step_outputs).mean()  # Compute the average loss over all training steps
    self.log('train_loss_mean', avg_loss)  # Log the average training loss
    self.training_step_outputs.clear()  # Clear the training step outputs list to free up memory

  
  def validation_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    mae = self.mae(scores, y)  # Compute mae using the scores and target
    md = self.md(scores, y)  # Compute F1 score using the scores and target
    self.log_dict({'val_loss': loss,
                   'val_mae': mae,
                   'val_md': md},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, mae, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    mae = self.mae(scores, y)  # Compute mae using the scores and target
    md = self.md(scores, y)  # Compute F1 score using the scores and target
    self.log_dict({'test_loss': loss,
                   'test_mae': mae,
                   'test_md': md},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, mae, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch
    scores = self(x)  # Compute the scores by passing the input through the network
    loss = self.loss_fn(scores, y)  # Compute the loss using the scores and target
    return loss, scores, y
  
  def predict_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
    scores = self(x)  # Compute the scores by passing the input through the network
    preds = torch.argmax(scores, dim=1)  # Compute the predicted labels
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.lr)  # Use Adam optimizer with the specified learning rate
