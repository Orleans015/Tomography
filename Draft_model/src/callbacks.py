from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping

class PrintingCallback(Callback):
  def __init__(self) -> None:
    super().__init__()
  
  def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
    print("Training is starting!")
    return super().on_train_start(trainer, pl_module)

  def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    print("Training is done!")
    return super().on_train_end(trainer, pl_module)
  