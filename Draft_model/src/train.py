import lightning as L
import torch
from model import TomoModel
from dataset import TomographyDataModule
import config
from callbacks import PrintingCallback, SaveBest, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("medium")

def main():
    logger = TensorBoardLogger("TB_logs", name="my_Tomo_model")

    model = TomoModel(
        inputsize=config.INPUTSIZE,
        learning_rate=config.LEARNING_RATE,
        outputsize=config.OUTPUTSIZE,
    )

    dm = TomographyDataModule(
        data_dir=config.DATA_DIR,
        file_name=config.FILE_NAME,
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
        enable_progress_bar=True, # Set to True to enable progress bar
        callbacks=[PrintingCallback(),
                   SaveBest(monitor="val_loss", logger=logger),
                   EarlyStopping(monitor="val_loss"),
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
    # # Test the model on a batch from the test set
    # test_loader = dm.test_dataloader()
    # test_batch = next(iter(test_loader))
    # x, y = test_batch
    # model.eval()
    # with torch.no_grad():
    #     pred = model(x)
    # print("Predicted: ", pred)
    # print("Actual: ", y)
    # print("Difference: ", pred - y)