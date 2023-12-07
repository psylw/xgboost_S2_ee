# %%
import rioxarray as rxr
from os import listdir
import os
import matplotlib.pyplot as plt
from rasterio.warp import reproject, Resampling,calculate_default_transform
import rasterio
import numpy as np

filenames = listdir(os.getcwd()+'\\bands')
filenames2 = listdir(os.getcwd()+'\\labels')


# %%
for i in range(len(filenames)):
    sar = 'bands\\'+filenames[i]
    label = [s for s in filenames2 if filenames[i][0:5] in s][0]

    label_dst = rasterio.open('labels\\'+label)
    dst_crs = label_dst.crs

    with rasterio.open(sar) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })

        # save reprojected and regridded tifs to temp folder
        with rasterio.open('temp\\test.tif', 'w',**kwargs) as dst:
            for j in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, j),
                    destination=rasterio.band(dst, j),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear)

            kwargs = dst.meta.copy()
            kwargs.update({
                'transform': label_dst.transform,
                'width': label_dst.width,
                'height': label_dst.height
            })

            with rasterio.open('temp\\'+filenames[i], 'w', **kwargs) as dst2:
                for k in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(dst, k),
                        destination=rasterio.band(dst2, k),
                        src_transform=dst.transform,
                        src_crs=dst.crs,
                        dst_transform=dst2.transform,
                        dst_crs=dst2.crs,
                        resampling=Resampling.bilinear)    




    # %%
