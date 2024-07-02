import math
from torchgeo.trainers import BaseTask
import os
import tempfile
import numpy as np
from collections.abc import Sequence
from typing import Callable, Optional,Any, Union
import torch
import rasterio
import glob
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
import torchvision.transforms as T
import torchvision
from torch import Tensor
import kornia.augmentation as K
from data_utils.statistics import WORLDVIEW3_NORMALIZE

class Worldview3UnlabelledDataset(NonGeoDataset):

    classes = []
    bands = ['coastal','blue','green','yellow','red','rededge','nir1','nir2']
    rgb_bands = ['red','green','blue']
    # WORLDVIEW3_NORMALIZE = {
    #                     "mean": [89.618358, 124.71933, 166.69266, 172.63822, 176.134245, 338.817245, 493.08148, 246.70549, 0.33602989, -0.3283134, -180.537485], 
    #                     "std": [93.307245, 112.43568, 120.601442, 139.562745, 154.26631, 187.33944, 306.17737, 325.04117, 0.439463715, 0.456241605, 193.490225]
    #                 }
                    
    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
        normalize: bool = True,
    ) -> None:
        self.root = root
        self.split = split
        self.transforms = transforms
        self.normalization = normalize
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.files = self._load_files()

        print(self.files[0])
        print('----')
        pass

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.files)

    def _load_files(self) -> list[dict[str, str]]:
        """Return the paths of the files in the dataset.

        Returns:
            list of dicts containing paths for each pair of image/dem/mask
        """
        directory = os.path.join(self.root)
        msi_paths = glob.glob(os.path.join(directory, 'msi', "*.tif"))

        files = []
        for msi_path in sorted(msi_paths):
            filename = msi_path.split("/")[-1]
            ndvi_path = os.path.join(directory, 'ndvi', filename)
            ndwi_path = os.path.join(directory, 'ndwi', filename)
            pisi_path = os.path.join(directory, 'pisi', filename)
            files.append(dict(msi=msi_path, ndvi=ndvi_path, ndwi=ndwi_path, pisi=pisi_path))

        return files

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        files = self.files[index]
        msi = self._load_image(files["msi"])
        ndvi = self._load_image(files["ndvi"], shape=msi.shape[1:])
        ndwi = self._load_image(files["ndwi"], shape=msi.shape[1:])
        # pisi = self._load_image(files["pisi"], shape=msi.shape[1:])

        image = torch.cat(tensors=[msi, ndvi, ndwi
                                   # , pisi
                                  ], dim=0).nan_to_num()

        sample = {"image": image}
        
        if self.transforms is not None:
            sample = self.transforms(sample['image'])
            # print(len(sample))
            sample = {"image": sample}

        return sample

    def _load_image(self, path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
        """Load a single image.

        Args:
            path: path to the image
            shape: the (h, w) to resample the image to

        Returns:
            the image
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.float_]" = f.read(
                out_shape=shape, out_dtype="float32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            return tensor

    def plot(
        self,
        sample: dict[str, Tensor],
        show_titles: bool = True,
        suptitle: Optional[str] = None,
        alpha: float = 0.5,
    ) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            show_titles: flag indicating whether to show titles above each panel
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        rgb_indices = []
        for band in self.rgb_bands:
            if band in self.bands:
                rgb_indices.append(self.bands.index(band))
            else:
                raise ValueError("Dataset doesn't contain some of the RGB bands")

        if(sample['image'].__class__ == list):
            views = sample['image']
            # print(len(views))
            ncols=len(views)
            fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
            for i in range(ncols):
                # print(i)
                image = views[i]
                if self.normalization:
                    image = K.Denormalize(mean=Tensor(WORLDVIEW3_NORMALIZE["mean"]), std=Tensor(WORLDVIEW3_NORMALIZE["std"]))(image)
                # print(image.shape)
                image = image.squeeze()
                image = image[rgb_indices]
                # print(image.shape)
                image = image.permute(1, 2, 0).numpy().astype(np.uint16)

                axs[i].imshow(image)
                axs[i].axis("off")
        else:
            ncols = 1
            
            image = sample["image"][rgb_indices]
            # image = image.to(torch.uint16)
            image = image.permute(1, 2, 0).numpy().astype(np.uint16)
    
            fig, axs = plt.subplots(ncols=1, figsize=(ncols * 10, 10))
            axs.imshow(image)
            axs.axis("off")

        if suptitle is not None:
            plt.suptitle(suptitle)

        # return fig
        pass
    pass
