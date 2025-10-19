from torch.optim.lr_scheduler import ReduceLROnPlateau

from transformer_model import TransformerModel
import pytorch_lightning as pl
import torch
import torch.nn as nn

class StockPredictor(pl.LightningModule):
    def __init__(self, device, learning_rate: float = 1e-4):
        super().__init__()
        self.model = TransformerModel().to(device)
        self.learning_rate = learning_rate
        self.criterion = nn.MSELoss()

    def forward(self, x, outputs=None):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        print("Training Step", batch_idx)
        outputs, loss = self.common_step(batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        print("Val Step", batch_idx)
        outputs, loss = self.common_step(batch, batch_idx)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        print("Test Step", batch_idx)
        outputs, loss = self.common_step(batch, batch_idx)
        self.log('test_loss', loss)
        return loss

    def common_step(self, batch, batch_idx):
        inputs, targets = batch
        print('Input Shape = ', inputs.shape, ' target shape = ' , targets.shape)
        outputs = self.forward(inputs, targets)
        loss = 0
        if targets is not None:
            loss = self.criterion(outputs, targets)
        predictions = torch.argmax(outputs, dim=1)
        return outputs, loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, patience=3, factor=0.5, min_lr=1e-6)
        return {'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler' : scheduler,
                    'interval': 'epoch',
                    'frequency': 1,
                    'monitor': 'val_loss'
                }
                }