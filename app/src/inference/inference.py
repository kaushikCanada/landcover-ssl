from typing import Any, Optional
import matplotlib.pyplot as plt
from matplotlib import colors
import segmentation_models_pytorch as smp
import torch.nn as nn
import geopandas as gpd
from collections.abc import Sequence
from torch import Tensor
import earthpy.plot as ep
import torch
import rasterio
from tqdm import tqdm
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import kornia.augmentation as K
import numpy as np

WORLDVIEW3_NORMALIZE = {
                        "mean": [89.618358, 124.71933, 166.69266, 172.63822, 176.134245, 338.817245, 493.08148, 246.70549, 0.33602989, -0.3283134], 
                        "std": [93.307245, 112.43568, 120.601442, 139.562745, 154.26631, 187.33944, 306.17737, 325.04117, 0.439463715, 0.456241605]
                    }

def load_image(path: str, shape: Optional[Sequence[int]] = None) -> Tensor:
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
            
# Function to load the trained model
def load_model(checkpoint_path, device):
    task = MyModel(
    model="unet",
    backbone="resnet50",
    weights=None,
    in_channels=11,
    num_classes=8,
    loss="ce",
    ignore_index=None,
    lr=0.001,
    patience=10,
    )
    model = task.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    return model


# Function to postprocess the output mask to GeoTIFF format
def postprocess_mask(mask, reference_path, output_path):
    with rasterio.open(reference_path) as src:
        profile = src.profile
        transform = src.transform
        crs = src.crs

    profile.update(dtype=rasterio.uint8, count=1)

    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)
        dst.transform = transform
        dst.crs = crs

# Function to make predictions
def predict(image_path, model, device):
    image = load_image(image_path)
    ndvi_path = image_path.replace('msi','ndvi')
    ndwi_path = image_path.replace('msi','ndwi')
    pisi_path = image_path.replace('msi','pisi')
    # print(image_path,ndvi_path)
    ndvi = load_image(ndvi_path, shape=image.shape[1:])
    ndwi = load_image(ndwi_path, shape=image.shape[1:])
    pisi = load_image(pisi_path, shape=image.shape[1:])

    image = torch.cat(tensors=[image, ndvi, ndwi, pisi], dim=0).nan_to_num()

    image = K.Normalize(mean=WORLDVIEW3_NORMALIZE['mean'], std=WORLDVIEW3_NORMALIZE['std'])(image).to(device)
    # image = preprocess_image(image_path)
    # image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).to(device)
    # image = image.unsqueeze(0).to(device)
    # print(image.shape)

    with torch.no_grad():
        output = model(image)+1
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return output,image

# Main function to load model, predict, and save the output
def predict_geotiff(image_path, checkpoint_path, output_path, device, labels = True, labels_path = None):
        model = load_model(checkpoint_path, device)
        prediction,image = predict(image_path, model, device)
        # print(prediction.shape)
        arr = rasterio.open(image_path).read()
    
    
        # C = np.array([
        # [.91,.59,.48],   # other impervious
        # [.50,.50,.50],     # road
        # [.82,.41,.12],    #  bare
        # [.2,.6,.2],       #  trees
        # [0,1,0],          #  grass
        # [.3,.7,0],        #  shrub
        # [1,0,0],          #  building
        # [0,0,.88],        #  water
        # ])
        # cmap = colors.ListedColormap(C)
        # ep.plot_rgb(arr,rgb=(4, 2, 1),stretch=True)
        # ep.plot_bands(prediction,vmin=1,vmax=8, cmap=cmap)
        if labels:
                lbl_path = image_path.replace("msi","lbl")
                lbl = rasterio.open(lbl_path).read()
                ep.plot_bands(lbl,vmin=1,vmax=8, cmap=cmap)
                if labels_path is not None:
                        postprocess_mask(lbl, lbl_path_path, lbl_output_path)
    
        postprocess_mask(prediction, image_path, output_path)

# Usage 000000001600
input_path_base = "../eodata/AZURE/"
image_path_labelled = input_path_base + "cleaned_gta_labelled_256m/msi/"
image_path_toronto = input_path_base + "cleaned_all_gta_256m/msi/"
image_path_montreal = input_path_base + "cleaned_all_montreal_256m/msi/"

output_path_base = "../eodata/UNET_OUTPUT/"
checkpoint_path = output_path_base + "checkpoint/" + 'checkpoint.pth.tar'
output_path_labelled = output_path_base + "predicted/labelled/"
output_path_toronto = output_path_base + "predicted/toronto/"
output_path_montreal = output_path_base + "predicted/montreal/"

# predict_geotiff(image_path_labelled+"000000001600.tif", checkpoint_path, output_path_labelled + "000000001600.tif", device='cuda')

from glob import glob
inputs = image_path_labelled # MUST MATCH
outputs = output_path_labelled # MUST MATCH

images=glob(inputs+'*.tif')
for image_path in tqdm(images):

    filename = image_path.split("/")[-1]
    output_path = outputs + filename
    # print(image_path,output_path)
    predict_geotiff(image_path, checkpoint_path, output_path, device='cuda')
