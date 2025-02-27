import lightning as L
import torch
from model import *
from dataset import TomographyDataModule
import config as C
from callbacks import PrintingCallback, SaveBest, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

config = C.config

def main():
    logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

    model = TomoGAN(
        config=config,
    )

    dm = TomographyDataModule(
        config=config,
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator=config["accelerator"],
        devices=config["devices"],
        min_epochs=1,
        max_epochs=config["num_epochs"],
        precision=config["precision"],
        enable_progress_bar=True, # Set to True to enable progress bar
        callbacks=[PrintingCallback(),
                   #SaveBest(monitor="val/loss", logger=logger),
                   # EarlyStopping(monitor="val_loss"),
                   ],
    )

    # the following function does not exist in version 2.2, the one i am using
    # trainer.tune(model, dm) # This will automatically find the best learning rate
    # Lightning automatically knows which loader inside the datamodule to use
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)
    return model, dm

if __name__ == "__main__":
    model, dm = main()
