import umap
import matplotlib.pyplot as plt
import numpy as np

embeddings = []
filenames = []

# disable gradients for faster calculations
model.eval()
with torch.no_grad():
    for i, (x, _, fnames) in enumerate(dataloader_test):
        # move the images to the gpu
        x = x.to(device)
        # embed the images with the pre-trained backbone
        y = model.backbone(x).flatten(start_dim=1)
        # store the embeddings and filenames in lists
        embeddings.append(y)
        filenames = filenames + list(fnames)

# concatenate the embeddings and convert to numpy
embeddings = torch.cat(embeddings, dim=0)
embeddings = embeddings.cpu().numpy()

# Perform UMAP
reducer = umap.UMAP()
umap_embeddings = reducer.fit_transform(embeddings)

# Plot the UMAP embeddings
plt.figure(figsize=(10, 8))
plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], s=5, cmap='Spectral')
plt.title('UMAP projection of the embeddings')
plt.show()
