import os
import time
import datetime
import math
from torchgeo.trainers import BaseTask
import argparse
import tempfile
import numpy as np
from collections.abc import Sequence
from typing import Callable, Optional,Any, Union
from torch.utils.data import DataLoader
import torch
import rasterio
import glob
import timm
from torch import nn
from tqdm import tqdm
from rasterio.enums import Resampling
from matplotlib import colors
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from PIL import Image
from torch import Tensor
from torchgeo.samplers.utils import _to_tuple
from torchgeo.datasets.geo import NonGeoDataset
from torchgeo.datamodules.geo import NonGeoDataModule
from torchgeo.datasets import NAIP, ChesapeakeDE, stack_samples, unbind_samples
from torchgeo.datasets.utils import download_url,draw_semantic_segmentation_masks,extract_archive,rgb_to_mask,percentile_normalization
from torchgeo.trainers import SemanticSegmentationTask
import torchvision.transforms as T
import torchvision
from einops import rearrange
from torch.utils.data import DataLoader
from torch.masked import masked_tensor, as_masked_tensor
from torchgeo.datamodules.utils import dataset_split
from torchgeo.transforms.transforms import _RandomNCrop
from torchgeo.transforms import AugmentationSequential, indices
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor
import kornia.augmentation as K
from torchmetrics import MetricCollection
from torchvision.models._api import WeightsEnum
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, Accuracy,FBetaScore,JaccardIndex,Precision,Recall
import lightning as L
from lightly.loss import BarlowTwinsLoss
from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules.heads import SimSiamPredictionHead, SimSiamProjectionHead, BarlowTwinsProjectionHead
from lightly.transforms.multi_view_transform import MultiViewTransform
from data_utils.statistics import WORLDVIEW3_NORMALIZE
from data_utils.wv3_unlabelled_dataset import Worldview3UnlabelledDataset
from train_utils.custom_multi_view_transform import CustomMultiViewTransform
from train_utils.barlowtwins import BarlowTwins
from train_utils.barlowtwinstrainer import BarlowTwinsTask
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.utils.data.distributed
import argparse

parser = argparse.ArgumentParser(description='cifar10 classification models, distributed data parallel test')
parser.add_argument('--lr', default=0.1, help='')
parser.add_argument('--batch_size', type=int, default=768, help='')
parser.add_argument('--log_dir', help='logdir for models and losses. default = .', default='./', type=str)
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--start_epoch', help='# of epochs. default = 0', default=0, type=int)
parser.add_argument('--max_epochs', help='# of epochs. default = 2', default=2, type=int)
parser.add_argument('--num_workers', type=int, default=1, help='')
parser.add_argument('--dist-backend', default='nccl', type=str, help='')
parser.add_argument('--world_size', default=1, type=int, help='')
parser.add_argument('--init_method', default='tcp://127.0.0.1:3456', type=str, help='')
parser.add_argument('--distributed', action='store_true', help='')
parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained checkpoint including hyperparams,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)

def main():
	print("Starting...")
	
	args = parser.parse_args()
	ngpus_per_node = torch.cuda.device_count()
	dict_args = vars(args)

	cleaned_all_gta_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_gta_256m/"
	cleaned_all_montreal_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_montreal_256m/"
	cleaned_gta_labelled_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
	output_path = dict_args['log_dir']
	
	""" This next line is the key to getting DistributedDataParallel working on SLURM:
		SLURM_NODEID is 0 or 1 in this example, SLURM_LOCALID is the id of the 
		current process inside a node and is also 0 or 1 in this example."""
	
	local_rank = int(os.environ.get("SLURM_LOCALID")) 
	rank = int(os.environ.get("SLURM_NODEID"))*ngpus_per_node + local_rank
	
	current_device = local_rank
	
	torch.cuda.set_device(current_device)
	
	""" this block initializes a process group and initiate communications
		between all processes running on all nodes """
	
	print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
	#init the process group
	dist.init_process_group(backend=args.dist_backend, init_method=args.init_method, world_size=args.world_size, rank=rank)
	print("process group ready!")
	
	print('From Rank: {}, ==> Making model..'.format(rank))
	
	class Net(nn.Module):
	
	def __init__(self):
	  super(Net, self).__init__()
	
	  self.conv1 = nn.Conv2d(3, 6, 5)
	  self.pool = nn.MaxPool2d(2, 2)
	  self.conv2 = nn.Conv2d(6, 16, 5)
	  self.fc1 = nn.Linear(16 * 5 * 5, 120)
	  self.fc2 = nn.Linear(120, 84)
	  self.fc3 = nn.Linear(84, 10)
	
	def forward(self, x):
	  x = self.pool(F.relu(self.conv1(x)))
	  x = self.pool(F.relu(self.conv2(x)))
	  x = x.view(-1, 16 * 5 * 5)
	  x = F.relu(self.fc1(x))
	  x = F.relu(self.fc2(x))
	  x = self.fc3(x)
	  return x
	
	net = Net()
	
	net.cuda()
	net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[current_device])
	
	print('From Rank: {}, ==> Preparing data..'.format(rank))

	toronto_unlabelled_dataset = Worldview3UnlabelledDataset(root = cleaned_all_gta_256m_path
                                                             ,transforms=CustomMultiViewTransform(input_size = 256,normalize=WORLDVIEW3_NORMALIZE)
                                                             )
	print(len(toronto_unlabelled_dataset))
	# print(toronto_unlabelled_dataset[0])
	
	montreal_unlabelled_dataset = Worldview3UnlabelledDataset(root = cleaned_all_montreal_256m_path
							      ,transforms=CustomMultiViewTransform(input_size = 256,normalize=WORLDVIEW3_NORMALIZE)
							      )
	print(len(montreal_unlabelled_dataset))
	# print(montreal_unlabelled_dataset[0])
 	dataset = torch.utils.data.ConcatDataset([toronto_unlabelled_dataset, montreal_unlabelled_dataset])
    	print(len(dataset))
	mysampler = torch.utils.data.distributed.DistributedSampler(dataset)
	mydataloader = DataLoader(dataset, batch_size=dict_args['batch_size'], shuffle=(mysampler is None), num_workers=dict_args['num_workers'], sampler=mysampler)

	print(len(mydataloader))
    	model = BarlowTwinsTask(model='resnet18',in_channels=11, batch_size = dict['batch_size'])
	print(model)

	transform_train = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	
	dataset_train = CIFAR10(root='~/scratch/landcover-ssl/data', train=True, download=False, transform=transform_train)
	
	train_sampler = torch.utils.data.distributed.DistributedSampler(dataset_train)
	train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler)
	
	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
	
	for epoch in range(args.max_epochs):
	
	train_sampler.set_epoch(epoch)
	
	# train(epoch, net, criterion, optimizer, train_loader, rank)

def train(epoch, net, criterion, optimizer, train_loader, train_rank):

    train_loss = 0
    correct = 0
    total = 0

    epoch_start = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):

       start = time.time()

       inputs = inputs.cuda()
       targets = targets.cuda()
       outputs = net(inputs)
       loss = criterion(outputs, targets)

       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

       train_loss += loss.item()
       _, predicted = outputs.max(1)
       total += targets.size(0)
       correct += predicted.eq(targets).sum().item()
       acc = 100 * correct / total

       batch_time = time.time() - start

       elapse_time = time.time() - epoch_start
       elapse_time = datetime.timedelta(seconds=elapse_time)
       print("From Rank: {}, Training time {}".format(train_rank, elapse_time))

if __name__=='__main__':
   main()
