from argparse import ArgumentParser

from pytorch_lightning.callbacks import ModelCheckpoint,RichProgressBar

from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from stock_predictor import StockPredictor
from stock_predictor_data_module import StockPredictorDataModule
import torch
import torch.multiprocessing as mp
from utilities import populate_scaler
import os


def run(base_directory, accelerator, devices):
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Populating scaler parameters...")
    populate_scaler(base_directory)

    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(base_directory, "checkpoints"),
                                          filename="stock-predictor-checkpoint",
                                          save_top_k=1,
                                          verbose=True,
                                          monitor="val_loss",
                                          mode="min")
    logger = TensorBoardLogger('lightning_logs',
                               name="stock_predictor")

    trainer = pl.Trainer(logger=logger, accelerator=accelerator,
                         callbacks=[checkpoint_callback, RichProgressBar()], max_epochs=1)
    if devices != None:
        trainer = pl.Trainer(logger=logger, accelerator=accelerator, devices=devices, callbacks=[checkpoint_callback, RichProgressBar()], max_epochs=1)

    model = StockPredictor(device).to(device)
    model = torch.compile(model)
    data_module = StockPredictorDataModule(base_directory=base_directory,
                                           device=device,
                                           train_workers=15,
                                           val_workers=15,
                                           test_workers=1,
                                           train_batch_size=128,
                                           val_batch_size=128,
                                           test_batch_size=16)
    model.train()
    trainer.fit(model, data_module)
    trainer.save_checkpoint(os.path.join(os.path.join(base_directory, "checkpoints"), "stock-predictor-final.ckpt"))


if __name__ ==  '__main__':
    mp.set_start_method('spawn', force=True)
#    base_directory = "/data/datasets/stockPredictor"
    base_directory = "/data/datasets/stockPredictor"
    parser = ArgumentParser()
    parser.add_argument("--accelerator", default='gpu')
    parser.add_argument("--devices", default=None)
    args = parser.parse_args()
    run(base_directory, args.accelerator, args.devices)
