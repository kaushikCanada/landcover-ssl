# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import datetime
import timm
from timm.scheduler.cosine_lr import CosineLRScheduler
from data_utils.statistics import WORLDVIEW3_NORMALIZE
from data_utils.wv3_unlabelled_dataset import Worldview3UnlabelledDataset
from train_utils.custom_multi_view_transform import CustomMultiViewTransform

parser = argparse.ArgumentParser(description='Barlow Twins Training')
parser.add_argument("--data_dir", type=str, help="path to data")
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch_size', default=2048, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')
parser.add_argument('--weight_decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                    metavar='DIR', help='path to checkpoint directory')


def main():
	args = parser.parse_args()
	args.ngpus_per_node = torch.cuda.device_count()
	dict_args = vars(args)
	if 'SLURM_JOB_ID' in os.environ:
		# single-node and multi-node distributed training on SLURM cluster
		# requeue job on SLURM preemption
		signal.signal(signal.SIGUSR1, handle_sigusr1)
		signal.signal(signal.SIGTERM, handle_sigterm)
		# find a common host name on all nodes
		# assume scontrol returns hosts in the same order on all nodes
		cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
		stdout = subprocess.check_output(cmd.split())
		host_name = stdout.decode().splitlines()[0]
		# args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
		args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
		args.dist_url = f'tcp://{host_name}:39778'

	local_rank = int(os.environ.get("SLURM_LOCALID")) 
	current_device = local_rank
	torch.cuda.set_device(current_device)
	torch.backends.cudnn.benchmark = True
	
	rank = int(os.environ.get("SLURM_NODEID"))*args.ngpus_per_node + local_rank
	args.rank = rank
	print('From Rank: {}, ==> Initializing Process Group...'.format(rank))
	torch.distributed.init_process_group(
		backend='nccl', init_method=args.dist_url,
		world_size=args.world_size, rank=rank)
	print("process group ready!")

	print('From Rank: {}, ==> Loading checkpoint..'.format(rank))
	if args.rank == 0:
		args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
		print(' '.join(sys.argv))
		# print(' '.join(sys.argv), file=stats_file)

	
	print('From Rank: {}, ==> Loading model..'.format(rank))
	model = BarlowTwins(args).cuda()
	model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
	param_weights = []
	param_biases = []
	for param in model.parameters():
		if param.ndim == 1:
		    param_biases.append(param)
		else:
		    param_weights.append(param)
	parameters = [{'params': param_weights}, {'params': param_biases}]
	model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[current_device])

	optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=args.weight_decay)
	# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
	# scheduler = CosineLRScheduler(optimizer, t_initial=10, lr_min=2e-8,
 #                  cycle_mul=2.0, cycle_decay=.5, cycle_limit=5,
 #                  warmup_t=10, warmup_lr_init=1e-6, warmup_prefix=False, t_in_epochs=True,
 #                  noise_range_t=None, noise_pct=0.67, noise_std=1.0,
 #                  noise_seed=42, k_decay=1.0, initialize=True)

	# automatically resume from checkpoint if it exists
	if (args.checkpoint_dir / 'checkpoint.pth').is_file():
		ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
				  map_location='cpu')
		start_epoch = ckpt['epoch']
		model.load_state_dict(ckpt['model'])
		optimizer.load_state_dict(ckpt['optimizer'])
	else:
		start_epoch = 0

	dict_args = vars(args)

	print('From Rank: {}, ==> Preparing data..'.format(rank))
	cleaned_all_gta_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_gta_256m/"
	cleaned_all_montreal_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_montreal_256m/"
	cleaned_gta_labelled_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
	
	toronto_unlabelled_dataset = Worldview3UnlabelledDataset(root = cleaned_all_gta_256m_path,transforms=CustomMultiViewTransform(input_size = 256,normalize=WORLDVIEW3_NORMALIZE))
	print(len(toronto_unlabelled_dataset))
	# print(toronto_unlabelled_dataset[0])
	
	montreal_unlabelled_dataset = Worldview3UnlabelledDataset(root = cleaned_all_montreal_256m_path,transforms=CustomMultiViewTransform(input_size = 256,normalize=WORLDVIEW3_NORMALIZE))
	print(len(montreal_unlabelled_dataset))
	# print(montreal_unlabelled_dataset[0])
	
	dataset = torch.utils.data.ConcatDataset([toronto_unlabelled_dataset, montreal_unlabelled_dataset])
	# print(len(dataset))

	assert args.batch_size % args.world_size == 0
	per_device_batch_size = args.batch_size // args.world_size
		
	sampler = torch.utils.data.distributed.DistributedSampler(dataset)
	loader = torch.utils.data.DataLoader(dataset, batch_size=per_device_batch_size, shuffle=(sampler is None),pin_memory=True, num_workers=dict_args['workers'], sampler=sampler)
	
	print(len(loader))
	avg_output_std = 0.0
	losses = []
	start_time = time.time()
	scaler = torch.cuda.amp.GradScaler()
	for epoch in range(start_epoch, args.epochs):
		
		np.random.seed(epoch)
		random.seed(epoch)
		sampler.set_epoch(epoch)
		epoch_start = time.time()
		print("From Rank: {}, EPOCH STARTED DATA LOADING---------------- {}".format(rank, datetime.timedelta(seconds=(epoch_start-start_time))))
		for step, batch in enumerate(loader, start=epoch * len(loader)):
			start = time.time()
			# print("From Rank: {}, BATCH {} STARTED ---------------- {}".format(rank, step, datetime.timedelta(seconds=(start-epoch_start))))
			new_batch = []
			for image in batch['image']:
				new_batch+=[image.squeeze(1)]
			y1,y2 = new_batch
			y1 = y1.cuda(non_blocking=True)
			y2 = y2.cuda(non_blocking=True)
			optimizer.zero_grad()
			with torch.cuda.amp.autocast():
				# print("From Rank: {}, BATCH {} LOSS FORWARD STARTED ---------------- {}".format(rank, step, datetime.timedelta(seconds=(time.time()-start))))
				loss,feature_vector = model.forward(y1, y2)
			# print("From Rank: {}, BATCH {} LOSS BACKWARD STARTED ---------------- {}".format(rank, step, datetime.timedelta(seconds=(time.time()-start))))
			scaler.scale(loss).backward()
			# print("From Rank: {}, BATCH {} STEP OPTIMIZER STARTED ---------------- {}".format(rank, step, datetime.timedelta(seconds=(time.time()-start))))
			scaler.step(optimizer)
			scaler.update()
			# scheduler.step(epoch)
			
			batch_time = time.time() - start
			elapse_time = time.time() - epoch_start
			
			if args.rank == 0:
				# Calculate the mean normalized standard deviation over features dimensions.
			        # If this is << 1 / sqrt(feature_vector.shape[1]), then the model is not learning anything.
			        output = feature_vector.detach()
			        output = F.normalize(output, dim=1)
			        output_std = torch.std(output, dim=0)
			        output_std = torch.mean(output_std, dim=0)
			        avg_output_std = 0.9 * self.avg_output_std + (1 - 0.9) * output_std.item()
				
			if step % args.print_freq == 0:
				if args.rank == 0:
					stats = dict(epoch=epoch, step=step,
						 loss=loss.item(),
						 time=int(time.time() - start_time))
					print(json.dumps(stats))
					print(json.dumps(stats), file=stats_file)
			# elapse_time = datetime.timedelta(seconds=elapse_time)
			# print("From Rank: {}, Training time {}".format(rank, elapse_time))
		print("From Rank: {}, EPOCH FINISHED DATA MIGHT BE LOADING ---------------- {}".format(rank,  datetime.timedelta(seconds=(time.time()-epoch_start))))
		if args.rank == 0:
			# the level of collapse is large if the standard deviation of the l2
			# normalized output is much smaller than 1 / sqrt(dim)
			collapse_level = max(0.0, 1 - math.sqrt(feature_vector.shape[1]) * avg_output_std)
			print("From Rank: {}, COLLAPSE LEVEL TILL NOW = {} ---------------- {}".format(rank, collapse_level, datetime.timedelta(seconds=(time.time()-epoch_start))))
			# save checkpoint
			state = dict(epoch=epoch + 1, model=model.state_dict(), collapse_level=collapse_level, optimizer=optimizer.state_dict())
			torch.save(state, args.checkpoint_dir / 'checkpoint.pth')
	if args.rank == 0:
		# save final model
		torch.save(model.module.backbone.state_dict(),
		   args.checkpoint_dir / 'resnet50.pth')
	print("From Rank: {}, TRAINING FINISHED ---------------- {}".format(rank,  datetime.timedelta(seconds=(time.time()-start_time))))
	torch.distributed.destroy_process_group()

def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.backbone = torchvision.models.resnet50(zero_init_residual=True)
        backbone = timm.create_model(
            'resnet50', in_chans=11, pretrained=False, num_classes=0
        )

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [2048] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
	h1: Tensor = self.backbone(y1)
	h2: Tensor = self.backbone(y2)
        z1 = self.projector(h1)
        z2 = self.projector(h2)

        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag
        return loss, h1


if __name__ == '__main__':
    main()
