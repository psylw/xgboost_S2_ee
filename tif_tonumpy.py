# %%
import os
import rioxarray as rxr
import numpy as np

root_dir = os.getcwd()
image_folder = os.path.join(root_dir, 'bands')

mask_folder = os.path.join(root_dir, 'labels')


images = os.listdir(image_folder)
masks = os.listdir(mask_folder)


for i in range(len(images)):
    im = images[i]
    t = rxr.open_rasterio(os.path.join(image_folder,im)).sel(band=1).values
    np.save(im.split('.tif')[0]+'.npy', t)


for i in range(len(masks)):
    im = masks[i]
    t = rxr.open_rasterio(os.path.join(mask_folder,im)).sel(band=1).values
    np.save(im.split('.tif')[0]+'.npy', t)

