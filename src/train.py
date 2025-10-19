from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from stock_predictor import StockPredictor
from stock_predictor_data_module import StockPredictorDataModule
import torch
import torch.multiprocessing as mp
from utilities import populate_scaler
import os


def run(base_directory):
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print("Populating scaler parameters...")
    populate_scaler(os.path.join(base_directory, 'outputs'), os.path.join(base_directory, 'cache'))

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

    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], max_epochs=1)

    model = StockPredictor(device).to(device)
    data_module = StockPredictorDataModule(base_directory=base_directory,
                                           device=device,
                                           train_workers=5,
                                           val_workers=5,
                                           test_workers=1,
                                           train_batch_size=128,
                                           val_batch_size=128,
                                           test_batch_size=16)
    model.train()
    trainer.fit(model, data_module)
    trainer.save_checkpoint(os.path.join(os.path.join(base_directory, "checkpoints"), "stock-predictor-final.ckpt"))


if __name__ ==  '__main__':
    mp.set_start_method('spawn', force=True)
    base_directory = "/data/datasets/stockPredictor"
    run(base_directory)