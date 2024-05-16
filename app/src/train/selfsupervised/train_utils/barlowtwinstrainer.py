from torchgeo.trainers import BaseTask, utils
import os
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation as K
from torch import Tensor
import timm
from torchgeo.models import get_weight
from torchvision.models._api import WeightsEnum
from typing import Any
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.loss import BarlowTwinsLoss


class BarlowTwinsTask(BaseTask):
    """Implementation of BarlowTwins[0] network.

    Recommended loss: :py:class:`lightly.loss.barlow_twins_loss.BarlowTwinsLoss`

    Default params are the ones explained in the original paper [0].
    [0] Zbontar,J. et.al. 2021. Barlow Twins... https://arxiv.org/abs/2103.03230

    Attributes:
        backbone:
            Backbone model to extract features from images.
            ResNet-50 in original paper [0].
        num_ftrs:
            Dimension of the embedding (before the projection head).
        proj_hidden_dim:
            Dimension of the hidden layer of the projection head. This should
            be the same size as `num_ftrs`.
        out_dim:
            Dimension of the output (after the projection head).

    """

    monitor = 'train_loss'

    def __init__(
        self,
        model: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 11,
        lr: float = 1e-3,
        patience: int = 10,
        batch_size : int = 64,
    ) -> None:
        """Initialize a new BarlowTwinsTask instance.

        Args:
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.

        .. versionchanged:: 0.4
           *backbone_name* was renamed to *backbone*. Changed backbone support from
           torchvision.models to timm.

        .. versionchanged:: 0.5
           *backbone*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *model*, *lr*, and *patience*.
        """
        self.weights = weights
        self.batch_size = batch_size
        super().__init__(ignore='weights')

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights
        in_channels: int = self.hparams['in_channels']

        # Create backbone
        network = timm.create_model(
            self.hparams['model'], in_chans=in_channels, pretrained=False, num_classes=0
        )

        self.backbone = nn.Sequential(*list(network.children())[:-1])

        self.projection_head = BarlowTwinsProjectionHead(512, 2048, 2048)

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """

        self.criterion = BarlowTwinsLoss()

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
        
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through BarlowTwins.

        Extracts features with the backbone and applies the projection
        head to the output space. 
        """
        x = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(x)
        return z
        
    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.

        Raises:
            AssertionError: If channel dimensions are incorrect.
        """
        x = batch['image']
        new_batch = []
        # loss=0
        for image in batch['image']:
            new_batch+=[image.squeeze(1)]
        x0,x1 = new_batch
        z0 = self.forward(x0)
        z1 = self.forward(x1)
        loss = self.criterion(z0, z1)
        self.log('train_loss', loss, batch_size=self.batch_size)
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""