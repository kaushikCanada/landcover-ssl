import timm
import segmentation_models_pytorch as smp
from datamodule import Worldview3LabelledDataModule
from rasterio.transform import from_origin
from rasterio.enums import Resampling
import earthpy.plot as ep
from utils import WORLDVIEW3_NORMALIZE
import kornia.augmentation as K


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
    model = smp.Unet(encoder_name="resnet50", encoder_weights=None, in_channels=11, classes=8)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
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
    print(image.shape)

    with torch.no_grad():
        output = model(image)+1
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    return output,image

# Main function to load model, predict, and save the output
def predict_geotiff(image_path, checkpoint_path, output_path, device):
    model = load_model(checkpoint_path, device)
    prediction,image = predict(image_path, model, device)
    # print(prediction.shape)
    arr = rasterio.open(image_path).read()
    ep.plot_rgb(arr,rgb=(4, 2, 1),stretch=True)
    ep.plot_bands(prediction,vmin=1,vmax=8,cmap="Paired")
    # postprocess_mask(prediction, image_path, output_path)

# Usage
image_path = '../eodata/AZURE/cleaned_gta_labelled_256m/msi/000000001600.tif'
checkpoint_path = 'checkpoint.pth.tar'
output_path = 'output.tif'

predict_geotiff(image_path, checkpoint_path, output_path, device='cuda')
