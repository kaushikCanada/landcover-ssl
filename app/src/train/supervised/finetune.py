import os
from pathlib import Path
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import pytorch_lightning as pl
from collections.abc import Sequence
from typing import Any
import argparse
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
import torch
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.models._api import WeightsEnum
from torchgeo.datasets import stack_samples, unbind_samples
from torchgeo.trainers import utils
from torchgeo.models import FCN, get_weight
import timm
import segmentation_models_pytorch as smp
from data_utils.wv3_labelled_datamodule import Worldview3LabelledDataModule

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')

def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device='cuda'):
	model = model.to(device)
	for epoch in tqdm(range(num_epochs)):
		model.train()
		running_loss = 0.0
		i=1
		for batch in dataloader:
			images = batch['image']
			masks = batch['mask']
			print(images.shape)
			print(masks.shape)
			asd
		
        #     images = images.to(device)
        #     masks = masks.to(device) - 1  # Shift labels from 1-8 to 0-7 for CrossEntropyLoss
        #     # print('data read')
            
        #     optimizer.zero_grad()
        #     outputs = model(images)
        #     loss = criterion(outputs, masks)
        #     loss.backward()
        #     optimizer.step()

        #     running_loss += loss.item() * images.size(0)
        #     # print('loop')
        #     i=i+1
        #     if i>5:
        #         break
        # epoch_loss = running_loss / len(dataloader.dataset)
        # print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

	return model
    
def main():
	args = parser.parse_args()
	args.ngpus_per_node = torch.cuda.device_count()
	dict_args = vars(args)
	if 'SLURM_JOB_ID' in os.environ:
		pass
	model = smp.Unet(
	encoder_name="resnet50",        
	encoder_weights=None,     
	in_channels=11,                 # Input channels
	classes=8                       # Number of output classes
	)
	
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	cleaned_gta_labelled_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
	batch_size = dict_args['batch_size']
	num_workers = dict_args['workers']
	
	dm = Worldview3LabelledDataModule(
		root=cleaned_gta_labelled_256m_path,batch_size=batch_size
	    )
	
	dm.setup("fit")
	
	train_loader = dm.train_dataloader()

	print('making model')
	trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=dict_args['epochs'])

if __name__ == '__main__':
    main()
