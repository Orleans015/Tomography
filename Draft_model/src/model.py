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
        nn.Linear(inputsize, 1500),  # Define a linear layer with input size and output size
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(1500, 5000),  # Define another linear layer
        nn.ReLU(),  # Apply ReLU activation function
        nn.Linear(5000, outputsize)  # Define the final linear layer with output size
    )
    self.loss_fn = nn.CrossEntropyLoss()  # Define the loss function as CrossEntropyLoss
    self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=outputsize)  # Define accuracy metric
    self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=outputsize)  # Define F1 score metric
    self.training_step_outputs = []  # Initialize an empty list to store training step outputs

  def forward(self, x):
    x = self.net(x)  # Pass the input through the network
    return x
  
  def training_step(self, batch, batch_idx):
    x, y = batch
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    accuracy = self.accuracy(scores, y)  # Compute accuracy using the scores and target
    f1_score = self.f1_score(scores, y)  # Compute F1 score using the scores and target
    self.training_step_outputs.append(loss)  # Append the loss to the training step outputs list
    self.log_dict({'train_loss': loss,
                   'train_accuracy': accuracy,
                   'train_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the training loss, accuracy, and F1 score
    if batch_idx % 100 == 0:
      x = x[:8]
      grid = torchvision.utils.make_grid(x.view(-1, 1, 28, 28))
      self.logger.experiment.add_image('MNIST_images', grid, self.global_step)  # Add MNIST images to TensorBoard
    
    return {"loss": loss, "preds": scores, "target": y}
  
  def on_train_epoch_end(self):
    avg_loss = torch.stack(self.training_step_outputs).mean()  # Compute the average loss over all training steps
    self.log('train_loss_mean', avg_loss)  # Log the average training loss
    self.training_step_outputs.clear()  # Clear the training step outputs list to free up memory

  
  def validation_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    accuracy = self.accuracy(scores, y)  # Compute accuracy using the scores and target
    f1_score = self.f1_score(scores, y)  # Compute F1 score using the scores and target
    self.log_dict({'val_loss': loss,
                   'val_accuracy': accuracy,
                   'val_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the validation loss, accuracy, and F1 score
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)  # Compute loss, scores, and target using a common step function
    accuracy = self.accuracy(scores, y)  # Compute accuracy using the scores and target
    f1_score = self.f1_score(scores, y)  # Compute F1 score using the scores and target
    self.log_dict({'test_loss': loss,
                   'test_accuracy': accuracy,
                   'test_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )  # Log the test loss, accuracy, and F1 score
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)  # Reshape the input tensor
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
