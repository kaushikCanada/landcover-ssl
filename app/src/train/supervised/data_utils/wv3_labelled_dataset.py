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

class Worldview3LabelledDataset(NonGeoDataset):
    classes = [
        "other",
        "road",
        "bare",
        "tree",
        "grass",
        "shrub",
        "building",
        "water",
    ]
    
    C = np.array([
    [.91,.59,.48],   # other impervious
    [.50,.50,.50],     # road
    [.82,.41,.12],    #  bare
    [.2,.6,.2],       #  trees
    [0,1,0],          #  grass
    [.3,.7,0],        #  shrub
    [1,0,0],          #  building
    [0,0,.88],        #  water
    ])
    
    
    msi_root = "msi"
    ndvi_root = "ndvi"
    ndwi_root = "ndwi"
    pisi_root = "pisi"
    lbl_root = "lbl"

    bands = ['coastal','blue','green','yellow','red','rededge','nir1','nir2']
    rgb_bands = ['red','green','blue']

    def __init__(
        self,
        root: str = "data",
        split: str = "train",
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        download: bool = False,
        checksum: bool = False,
    ) -> None:
        self.root = root
        self.split = split
        self.transforms = transforms

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
        images = glob.glob(
            os.path.join(directory, self.msi_root, "*.tif"), recursive=True
        )

        files = []
        for msi in sorted(images):
            ndvi = msi.replace(self.msi_root, self.ndvi_root)
            ndwi = msi.replace(self.msi_root, self.ndwi_root)
            pisi = msi.replace(self.msi_root, self.pisi_root)

            if self.split == "train":
                target = msi.replace(self.msi_root, self.lbl_root)
                files.append(dict(msi=msi, ndvi=ndvi, ndwi=ndwi, pisi=pisi, target=target))
            else:
                files.append(dict(msi=msi, ndvi=ndvi, ndwi=ndwi, pisi=pisi))

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
        pisi = self._load_image(files["pisi"], shape=msi.shape[1:])
        image = torch.cat(tensors=[msi, ndvi, ndwi, pisi], dim=0)

        sample = {"image": image}
        
        if self.split == "train":
            mask = self._load_target(files["target"])
            sample["mask"] = mask

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample
        pass

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

    def _load_target(self, path: str) -> Tensor:
        """Load the target mask for a single image.

        Args:
            path: path to the image

        Returns:
            the target mask
        """
        with rasterio.open(path) as f:
            array: "np.typing.NDArray[np.int_]" = f.read(
                indexes=1, out_dtype="int32", resampling=Resampling.bilinear
            )
            tensor = torch.from_numpy(array)
            tensor = tensor.to(torch.long)
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

        # image = sample["image"][rgb_indices].permute(1, 2, 0).float()
        ncols = 4
        # image1 = draw_semantic_segmentation_masks(
        #     sample["image"][rgb_indices], sample["mask"], alpha=alpha, colors=self.colormap
        # )
        msi = sample["image"][rgb_indices]
        # image = image.to(torch.uint16)
        msi = msi.permute(1, 2, 0).numpy().astype(np.uint16)

        ndvi = sample["image"][-3].numpy()
        ndvi = percentile_normalization(ndvi, lower=0, upper=100, axis=(0, 1))
        ndwi = sample["image"][-2].numpy()
        ndwi = percentile_normalization(ndwi, lower=0, upper=100, axis=(0, 1))
        pisi = sample["image"][-1].numpy()
        pisi = percentile_normalization(pisi, lower=0, upper=100, axis=(0, 1))

        showing_mask = "mask" in sample
        showing_prediction = "prediction" in sample
        cmap = colors.ListedColormap(self.C)

        if showing_mask:
            mask = sample["mask"].numpy()
            ncols += 1
        
        if showing_prediction:
            pred = sample["prediction"].numpy()
            ncols += 1
            
        fig, axs = plt.subplots(ncols=ncols, figsize=(ncols * 10, 10))
        axs[0].imshow(msi)
        axs[0].axis("off")
        axs[1].imshow(ndvi, cmap = plt.cm.viridis)
        axs[1].axis("off")
        axs[2].imshow(ndwi, cmap = plt.cm.plasma)
        axs[2].axis("off")
        axs[3].imshow(pisi, cmap = plt.cm.Greys)
        axs[3].axis("off")
        if showing_mask:
            axs[4].imshow(mask, vmin=1, vmax=8, cmap=cmap, interpolation="none")
            axs[4].axis("off")
            if showing_prediction:
                axs[5].imshow(pred,vmin=1, vmax=8, cmap=cmap, interpolation="none")
                axs[5].axis("off")
        elif showing_prediction:
            axs[4].imshow(pred, vmin=1, vmax=8, cmap=cmap, interpolation="none")
            axs[4].axis("off")

        if show_titles:
            axs[0].set_title("MSI")
            axs[1].set_title("NDVI")
            axs[2].set_title("NDWI")
            axs[3].set_title("PISI")

            if showing_mask:
                axs[4].set_title("GT")
                if showing_prediction:
                    axs[5].set_title("PRED")
            elif showing_prediction:
                axs[4].set_title("PRED")

        if suptitle is not None:
            plt.suptitle(suptitle)

        return None
