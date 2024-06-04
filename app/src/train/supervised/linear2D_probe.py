import pytorch_lightning as pl
import torch
from train_utils.trainer import MyModel
from datamodule import Worldview3LabelledDataModule

root= '../eodata/AZURE/cleaned_gta_labelled_256m/'
batch_size = 4
num_workers = 0

dm = Worldview3LabelledDataModule(
            root=root,batch_size=batch_size,num_workers=num_workers
        )

dm.setup("fit")

net = MyModel()

task = MyModel(
    model="unet",
    backbone="resnet50",
    weights=None,
    in_channels=11,
    num_classes=8,
    loss="ce",
    ignore_index=None,
    lr=0.001,
    patience=10,
)
trainer = pl.Trainer(max_epochs=2,accelerator='gpu')

trainer.fit(model=task, train_dataloaders = dm.train_dataloader(), val_dataloaders = dm.val_dataloader())
