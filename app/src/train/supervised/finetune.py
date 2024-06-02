import os
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import pytorch_lightning as pl
from collections.abc import Sequence
from typing import Any
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.models._api import WeightsEnum
from torchgeo.datasets import stack_samples, unbind_samples
from torchgeo.trainers import utils
from torchgeo.models import FCN, get_weight
import timm
import segmentation_models_pytorch as smp
from datamodule import Worldview3LabelledDataModule

model = smp.Unet(
    encoder_name="resnet50",        # Choose encoder, e.g., resnet50, efficientnet-b7, etc.
    encoder_weights=None,     # Use pre-trained weights for encoder initialization
    in_channels=11,                 # Input channels
    classes=8                       # Number of output classes
)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, dataloader, criterion, optimizer, num_epochs=25, device='cuda'):
    model = model.to(device)
    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0
        i=1
        for batch in dataloader:
            images = batch['image']
            masks = batch['mask']
            images = images.to(device)
            masks = masks.to(device) - 1  # Shift labels from 1-8 to 0-7 for CrossEntropyLoss
            # print('data read')
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            # print('loop')
            i=i+1
            if i>50:
                break
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch}/{num_epochs - 1}, Loss: {epoch_loss:.4f}')

    return model

root= '../eodata/AZURE/cleaned_gta_labelled_256m/'
batch_size = 12
num_workers = 0

dm = Worldview3LabelledDataModule(
            root=root,batch_size=batch_size
        )

dm.setup("fit")

train_loader = dm.train_dataloader()

trained_model = train_model(model, train_loader, criterion, optimizer, num_epochs=2)
