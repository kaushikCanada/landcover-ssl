import math
from torchgeo.trainers import BaseTask
import os
import argparse
import datetime
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

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, help="path to data")
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='number of threads')
    parser.add_argument('--logdir', help='logdir for models and losses. default = .', default='./', type=str)
    parser.add_argument('--lr', help='learning_rate for pose. default = 0.001', default=0.001, type=float)
    parser.add_argument('--display_freq', help='Frequency to display result image on Tensorboard, in batch units', default=64, type=int)
    parser.add_argument('--epoch', help='# of epochs. default = 2', default=2, type=int)
    parser.add_argument('--gpus_per_node', help='# of gpus per node. default = 1', default=1, type=int)
    parser.add_argument('--number_of_nodes', help='# of nodes. default = 1', default=1, type=int)
    parser.add_argument('--clip_grad_norm', help='Clipping gradient norm, 0 means no clipping', type=float, default=0.)
    parser.add_argument('--pin_memory', help='Whether to utilize pin_memory in dataloader', type=bool, default=True)
    parser.add_argument("--limit", type=int, default=5, help="no. of records to process")
    parser.add_argument("--resume_from_checkpoint",
                        help="Directory of pre-trained checkpoint including hyperparams,  \n"
                             "None --> Do not use pre-trained model. Training will start from random initialized model", default=None)
    args = parser.parse_args()
    return args

def main(args):
    """Main function of the script."""

    dict_args = vars(args)

    # Initialize logging paths
    now = datetime.datetime.now().strftime('%m%d%H%M%S')
    weight_save_dir = os.path.join(dict_args["logdir"], os.path.join('models', 'state_dict', now))

    cleaned_all_gta_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_gta_256m/"
    cleaned_all_montreal_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_all_montreal_256m/"
    cleaned_gta_labelled_256m_path = dict_args['data_dir'] + "/AZURE/cleaned_gta_labelled_256m/"
    output_path = dict_args['data_dir'] + "/output/"

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
    # dataset = LightlyDataset.from_torch_dataset(dataset)
    # print(len(dataset))
    dataloader = torch.utils.data.DataLoader(
                                            dataset,
                                            batch_size=dict_args['batch_size'],
                                            shuffle=True,
                                            drop_last=True,
                                            num_workers=dict_args['num_workers'],
                                            pin_memory=dict_args['pin_memory']
                                            )
    print(len(dataloader))
    model = BarlowTwinsTask(model='resnet18',in_channels=11, batch_size = dict['batch_size'])
    print(model)
    trainer = L.Trainer(max_epochs=dict_args['epoch'],
                        gradient_clip_val=dict_args['clip_grad_norm'],
                        accelerator="gpu", 
                        devices=dict_args['gpus_per_node'], 
                        num_nodes=dict_args['number_of_nodes'], 
                        strategy='ddp',
                        enable_progress_bar=True
                        )

    # for batch in tqdm(dataloader):
    #     pass
    trainer.fit(model=model, train_dataloaders=dataloader)


if __name__ == "__main__":

    # mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        # f"path to toronto data: {args.toronto_data}",
        # f"path to montreal data: {args.montreal_data}",
        # f"path to output data: {args.output_data}",
        # f"num workers: {args.num_workers}",
        # f"no. of records to process: {args.limit}",

    ]

    for line in lines:
        print(line)
    
    main(args)

    # mlflow.end_run()
