import os
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
from torch.optim import AdamW


class MyModel(pl.LightningModule):
    #: Model to train.
    model: Any

    #: Performance metric to monitor in learning rate scheduler and callbacks.
    monitor = 'val_loss'

    #: Whether the goal is to minimize or maximize the performance metric to monitor.
    mode = 'min'
    
            
    def __init__(self,
                model: str = 'unet',
                backbone: str = 'resnet50',
                weights: WeightsEnum | str | bool | None = None,
                in_channels: int = 11,
                num_classes: int = 8,
                num_filters: int = 3,
                loss: str = 'ce',
                class_weights: Tensor | None = None,
                ignore_index: int | None = None,
                lr: float = 1e-3,
                patience: int = 10,
                freeze_backbone: bool = False,
                freeze_decoder: bool = False,
                ignore: Sequence[str] | str | None = None,
         ) -> None:
        """Initialize a new model instance.

        Args:
            ignore: Arguments to skip when saving hyperparameters.
        """
        super().__init__()
        self.weights = weights
        ignore = 'weights'
        self.save_hyperparameters(ignore=ignore)
        self.configure_models()
        self.configure_losses()
        self.configure_metrics()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']
        num_filters: int = self.hparams['num_filters']

        if model == 'unet':
                self.model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
        elif model == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights='imagenet' if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == 'fcn':
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )
            
        if model != 'fcn':
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams['freeze_backbone'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False


    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        ignore_index = self.hparams['ignore_index']
        if loss == 'ce':
            ignore_value = 0 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                # ignore_index=ignore_value, weight=self.hparams['class_weights']
            )
        elif loss == 'jaccard':
            # JaccardLoss requires a list of classes to use instead of a class
            # index to ignore.
            classes = [
                i for i in range(self.hparams['num_classes']) if i != ignore_index
            ]

            self.criterion = smp.losses.JaccardLoss(mode='multiclass', classes=classes)
        elif loss == 'focal':
            self.criterion = smp.losses.FocalLoss(
                'multiclass', ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams['num_classes']
        ignore_index: int | None = self.hparams['ignore_index']
        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average='global',
                    average='micro',
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average='micro'
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')
        
    def configure_optimizers(
            self,
        ) -> 'lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig':
            """Initialize the optimizer and learning rate scheduler.
    
            Returns:
                Optimizer and learning rate scheduler.
            """
            optimizer = AdamW(self.parameters(), lr=self.hparams['lr'])
            scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams['patience'])

            return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": self.monitor,
            "frequency": 1,
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
        },
    }

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
        """
        x = batch['image']
        y = batch['mask'] - 1
        batch_size = x.shape[0]
        y_hat = self(x)
        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size, prog_bar=True)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size, prog_bar=True)
        return loss
        
    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask'] - 1
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size, prog_bar=True)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size, prog_bar=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['mask'] - 1
        batch_size = x.shape[0]
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, batch_size=batch_size, prog_bar=True)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size, prog_bar=True)
        
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            args: Arguments to pass to model.
            kwargs: Keyword arguments to pass to model.

        Returns:
            Output of the model.
        """
        return self.model(*args, **kwargs)
