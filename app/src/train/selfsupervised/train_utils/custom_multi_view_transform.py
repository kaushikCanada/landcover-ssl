from typing import Dict, List, Optional, Tuple, Union

import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
import kornia.augmentation as K
from lightly.transforms.gaussian_blur import GaussianBlur
from lightly.transforms.multi_view_transform import MultiViewTransform
from lightly.transforms.rotation import random_rotation_transform
from data_utils.statistics import WORLDVIEW3_NORMALIZE


class CustomMultiViewTransform(MultiViewTransform):

    def __init__(
        self,
        input_size: int = 224,
        normalize: Union[None, Dict[str, List[float]]] = None
    ):
        view_transform = CustomViewTransform(
            input_size=input_size,
            normalize=normalize,
        )
        super().__init__(transforms=[view_transform, view_transform])

class CustomViewTransform:
    def __init__(
        self,
        input_size: int = 224,
        normalize: Union[None, Dict[str, List[float]]] = None
    ):
        transform = K.AugmentationSequential(
                    K.RandomResizedCrop(size=(input_size, input_size),scale=(0.2, 1.0)),
                    # K.RandomRotation(degrees=45.0, p=0.5),
                    K.RandomHorizontalFlip(p=0.5),
                    # K.RandomVerticalFlip(p=0.5),
                    K.RandomBoxBlur(p=0.3),
                    K.RandomPerspective(p=0.3),
                    # K.RandomJigsaw(p=0.3),
                    # K.RandomChannelShuffle(p=0.3),
                    # K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5),
            
         )
        
        if normalize:
            transform = K.AugmentationSequential(
                        transform,
                        K.Normalize(mean=Tensor(normalize["mean"]), std=Tensor(normalize["std"])),
                        # data_keys = ['image'],
                        )
            
        self.transform = transform

    def __call__(self, image: Union[Tensor, Image]) -> Tensor:
        """
        Applies the transforms to the input image.

        Args:
            image:
                The input image to apply the transforms to.

        Returns:
            The transformed image.

        """
        transformed: Tensor = self.transform(image)
        return transformed
