import torchvision.transforms as T
import torchvision
from torch.utils.data import random_split
from torch import Tensor
import torch
import kornia.augmentation as K
from data_utils.statistics import WORLDVIEW3_NORMALIZE
from data_utils.wv3_labelled_dataset import Worldview3LabelledDataset

class Worldview3LabelledDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Worldview 3 dataset.

    Uses the train/test splits from the dataset.


    """

    mean=Tensor(WORLDVIEW3_NORMALIZE["mean"])
    std=Tensor(WORLDVIEW3_NORMALIZE["std"])
    
    def __init__(
        self,
        batch_size: int = 8,
        num_workers: int = 0,
        val_split_pct: float = 0.1,
        test_split_pct: float = 0.2,
        patch_size: Union[tuple[int, int], int] = 256,
        **kwargs,
    ) -> None:
        """Initialize a new Worldview3UnlabelledDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~Worldview3UnlabelledDataset`.
        """
        super().__init__(Worldview3LabelledDataset, batch_size, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            # _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image"],
        )
        # self.aug=None

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        # print(batch_size)
        self.dataset = Worldview3LabelledDataset(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        if stage in ['fit', 'validate','test']:
            self.train_dataset, self.val_dataset, self.test_dataset = random_split(
                self.dataset, [1 - (self.val_split_pct+self.test_split_pct),self.val_split_pct, self.test_split_pct], generator
            )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        """Apply batch augmentations to the batch after it is transferred to the device.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch of data.
        """
        pass
