import lightning as L
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from tqdm import tqdm
import torchmetrics
from torchmetrics import Metric

# Define an homebrewed metric (accuracy)
class MyAccuracy(Metric):
  def __init__(self):
    super().__init__()
    self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
    self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds, target):
    preds = torch.argmax(preds, dim=1)
    assert preds.shape == target.shape
    total = len(target)
    self.correct += torch.sum(preds == target)
    self.total += target.numel()

  def compute(self):
    return self.correct.float() / self.total.float()
  
class NN(L.LightningModule):
  def __init__(self, inputsize, num_classes):
    super().__init__()
    self.l1 = nn.Linear(inputsize, 128)
    self.l2 = nn.Linear(128, num_classes)
    self.loss_fn = nn.CrossEntropyLoss()
    self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    self.f1_score = torchmetrics.F1Score(task='multiclass', num_classes=num_classes)
    self.training_step_outputs = []

  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    return x
  
  def training_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)
    accuracy = self.accuracy(scores, y)
    f1_score = self.f1_score(scores, y)
    self.training_step_outputs.append(loss)
    self.log_dict({'train_loss': loss,
                   'train_accuracy': accuracy,
                   'train_f1_score': f1_score},
                   on_step=False, on_epoch=True, prog_bar=True
                   )
    return loss
  
  def on_train_epoch_end(self):
    avg_loss = torch.stack(self.training_step_outputs).mean()
    self.log('train_loss_mean', avg_loss)
    # free up the memory
    self.training_step_outputs.clear()

  
  def validation_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)
    return loss
  
  def test_step(self, batch, batch_idx):
    loss, scores, y = self._common_step(batch, batch_idx)
    return loss
  
  def _common_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self(x) # this is equal to self.forward(x)
    loss = self.loss_fn(scores, y)
    return loss, scores, y
  
  def predict_step(self, batch, batch_idx):
    x, y = batch
    x = x.reshape(x.size(0), -1)
    scores = self(x)
    preds = torch.argmax(scores, dim=1)
    return preds
  
  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=0.001)

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
  

# Set device cuda for GPU if it's available otherwise use CPU  
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
inputsize = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 10

# Initialize model
model = NN(inputsize=inputsize, num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data
dm = MnistDataModule(data_dir='.', batch_size=batch_size, num_workers=4)

trainer = L.Trainer(accelerator='gpu', devices=[0],
                     min_epochs=1, max_epochs=10,
                     precision=16,
                     callbacks=[L.Callback()],)

# the following function does not exist in version 2.2, the one i am using
# trainer.tune(model, dm) # This will automatically find the best learning rate
# Lightning automatically knows which loader inside the datamodule to use
trainer.fit(model, dm) 
trainer.validate(model, dm)
trainer.test(model, dm)
