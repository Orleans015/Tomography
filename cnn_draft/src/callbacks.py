from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger


# The following callback is an example of how a callback should be issued in the
# Trainer callbacks list, it prints a message when the training starts and when
# it ends.
class PrintingCallback(Callback):
  def __init__(self) -> None:
    super().__init__()
  
  def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
    print("Training is starting!")
    return super().on_train_start(trainer, pl_module)

  def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    print("Training is done!")
    return super().on_train_end(trainer, pl_module)
  
# Save the best model based on the validation loss, the following callback should
# be used in the Trainer callbacks list, it should check the validation loss value,
# compare it with the already saved validation and save the model if the new value
# is lower than the previous one.

class SaveBest(Callback):
  def __init__(self, monitor: str, logger: TensorBoardLogger) -> None:
    super().__init__()
    self.monitor = monitor
    self.logger = logger
    self.best_val_loss = float('inf')

  def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    val_loss = trainer.callback_metrics[self.monitor]
    if val_loss < self.best_val_loss:
      self.best_val_loss = val_loss
      trainer.save_checkpoint(f"{self.logger.log_dir}/best_model.ckpt")
    return super().on_validation_end(trainer, pl_module)
  
  def on_train_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    print(f"Best validation loss: {self.best_val_loss}")
    return super().on_train_end(trainer, pl_module)