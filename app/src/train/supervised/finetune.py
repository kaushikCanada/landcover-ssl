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

# Metrics
train_accuracy = MulticlassAccuracy(num_classes=8).to('cuda:0')
val_accuracy = MulticlassAccuracy(num_classes=8).to('cuda:0')
test_accuracy = MulticlassAccuracy(num_classes=8).to('cuda:0')

# Helper function for converting 1-8 labels to 0-7
def mask_labels_1_to_8_to_0_to_7(mask):
    return mask - 1

# Loss and Optimizer
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, targets):
        targets = mask_labels_1_to_8_to_0_to_7(targets)
        return self.criterion(inputs, targets)

# Training Function
def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
	model.train()
	epoch_loss = 0
	epoch_accuracy = 0
	for batch in tqdm(dataloader, desc="Training", leave=False):
		images, masks = batch['image'].to(device), batch['mask'].to(device)
		optimizer.zero_grad()
		outputs = model(images)
		loss = loss_fn(outputs, masks)
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
		preds = torch.argmax(outputs, dim=1)
		epoch_accuracy += train_accuracy(preds, mask_labels_1_to_8_to_0_to_7(masks)).item()
	return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

# Validation Function
def validate_one_epoch(model, dataloader, loss_fn, device):
	model.eval()
	epoch_loss = 0
	epoch_accuracy = 0
	with torch.no_grad():
		for batch in tqdm(dataloader, desc="Validation", leave=False):
			images, masks = batch['image'].to(device), batch['mask'].to(device)
			outputs = model(images)
			loss = loss_fn(outputs, masks)
			epoch_loss += loss.item()
			preds = torch.argmax(outputs, dim=1)
			epoch_accuracy += val_accuracy(preds, mask_labels_1_to_8_to_0_to_7(masks)).item()
	return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

# Test Function
def test_one_epoch(model, dataloader, loss_fn, device):
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Testing", leave=False):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            epoch_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            epoch_accuracy += test_accuracy(preds, mask_labels_1_to_8_to_0_to_7(masks)).item()

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

# Save and Load Checkpoint
def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint '{filename}' (epoch {epoch})")
        return epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return 0

# Main Training Loop
def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=20, resume = False, checkpoint_path="checkpoint.pth.tar"):
	if resume:
    		start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
	else:
		start_epoch = 0

	best_loss = float('inf')

	for epoch in range(start_epoch, num_epochs):
		print(f"Epoch {epoch + 1}/{num_epochs}")
		train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
		val_loss, val_acc = validate_one_epoch(model, val_loader, loss_fn, device)
		print(f"Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
		
		if val_loss < best_loss:
			best_loss = val_loss
			if resume:
				save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint_path)
				print("Model saved!")
	# Testing after training
	test_loss, test_acc = test_one_epoch(model, test_loader, loss_fn, device)
	print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
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
	loss_fn = WeightedCrossEntropyLoss()
	cleaned_gta_labelled_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
	batch_size = dict_args['batch_size']
	num_workers = dict_args['workers']
	
	dm = Worldview3LabelledDataModule(
		root=cleaned_gta_labelled_256m_path,batch_size=batch_size
	    )
	
	dm.setup("fit")
	
	train_loader = dm.train_dataloader()
	val_loader = dm.val_dataloader()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	print('making model')
	train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs=dict_args['epochs'], resume=False, checkpoint_path=dict_args['checkpoint_dir']/"checkpoint.pth.tar")
	# trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=dict_args['epochs'])

if __name__ == '__main__':
    main()
