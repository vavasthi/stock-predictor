from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl

from stock_predictor import StockPredictor
from stock_predictor_data_module import StockPredictorDataModule
import torch
import torch.multiprocessing as mp

def run():
    has_mps = torch.backends.mps.is_built()
    device = "mps" if has_mps else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", filename="best-checkpoint", save_top_k=1, verbose=True,
                                          monitor="val_loss", mode="min")
    logger = TensorBoardLogger('lightning_logs', name="stock_predictor")

    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint_callback], max_epochs=500)

    model = StockPredictor(device).to(device)
    data_module = StockPredictorDataModule(base_directory="/data/datasets/stockPredictor", device=device,
                                           train_batch_size=1024, val_batch_size=64, test_batch_size=32)
    data_module.setup()
    model.train()
    trainer.fit(model, data_module)


if __name__ ==  '__main__':
    mp.set_start_method('spawn', force=True)
    run()