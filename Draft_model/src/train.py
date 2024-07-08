import lightning as L
import torch
from model import NN
from dataset import MnistDataModule, TomographyDataModule
import config
from callbacks import PrintingCallback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

def main():
    logger = TensorBoardLogger("TB_logs", name="my_MNIST_model")

    model = NN(
        inputsize=config.INPUTSIZE,
        learning_rate=config.LEARNING_RATE,
        outputsize=config.OUTPUTSIZE,
    )

    dm = MnistDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
    )

    trainer = L.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=config.NUM_EPOCHS,
        precision=config.PRECISION,
        callbacks=[PrintingCallback(),
                   EarlyStopping(monitor="train_accuracy"),
                   L.Callback()],
    )

    # the following function does not exist in version 2.2, the one i am using
    # trainer.tune(model, dm) # This will automatically find the best learning rate
    # Lightning automatically knows which loader inside the datamodule to use
    trainer.fit(model, dm)
    trainer.validate(model, dm)
    trainer.test(model, dm)


if __name__ == "__main__":
    main()
