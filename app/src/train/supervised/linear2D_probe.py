import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from pathlib import Path
from train_utils.trainer import MyModel
import argparse
from data_utils.wv3_labelled_datamodule import Worldview3LabelledDataModule

parser = argparse.ArgumentParser(description='Worldview 3')
parser.add_argument('--lr', default=0.001, help='Learning Rate')
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--max_epochs', type=int, default=4,  metavar='N', help='number of data loader workers')
parser.add_argument('--batch_size', default=32, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--model_name', default="unet", type=str, help='model name')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of data loader workers')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path, metavar='DIR', help='path to checkpoint directory')

            
def main():
            print("Starting...")

            args = parser.parse_args()
            dict_args = vars(args)
            root = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
            batch_size = dict_args['batch_size']
            num_workers = dict_args['num_workers']
            lr = float(dict_args['lr'])
            MODEL_NAME = dict_args['model_name']
            
            dm = Worldview3LabelledDataModule(
                        root=root,batch_size=batch_size,num_workers=num_workers
                    )
            
            dm.setup("fit")
            
            net = MyModel()

            checkpoint_callback = ModelCheckpoint(
                monitor="val_loss",
                dirpath=str(dict_args['checkpoint_dir']) + MODEL_NAME + "_logs/",
                save_top_k=1,
                save_last=True,
            )

            early_stopping_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=0.00,
                patience=10,
            )
            csv_logger = CSVLogger(
                        save_dir=dict_args['checkpoint_dir'],
                        name=MODEL_NAME + "_logs"
            )
            
            task = MyModel(
                model=MODEL_NAME,
                backbone="resnet50",
                weights=None,
                in_channels=11,
                num_classes=8,
                loss="ce",
                ignore_index=None,
                lr=lr,
                patience=10,
            )
            trainer = pl.Trainer(max_epochs=dict_args['max_epochs'], 
                                 accelerator="gpu",
                                 callbacks=[early_stopping_callback], #checkpoint_callback,
                                 logger=[csv_logger],
                                 devices=[0], 
                                 num_nodes=1, 
                                 default_root_dir = str(dict_args['checkpoint_dir']) + MODEL_NAME + "_logs/")
            
            trainer.fit(model=task, train_dataloaders = dm.train_dataloader(), val_dataloaders = dm.val_dataloader())
            
            dm.setup("test")
            trainer.test(model=task, dataloaders = dm.test_dataloader())
            

if __name__=='__main__':
   main()
