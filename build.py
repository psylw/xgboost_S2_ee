# %%
import numpy as np
from os import listdir
import os
import matplotlib.pyplot as plt
import pandas as pd

root_dir = os.getcwd()

image_folder = os.path.join(root_dir, 'band')
mask_folder = os.path.join(root_dir, 'label')
images = listdir(image_folder)

mlist = os.listdir(mask_folder)

masks = [] # get label for image (2 images, or i could concat images)
for i in range(len(images)):
    for j in mlist:
        if images[i][0:5] in j:
            masks.append(j)

# %%

# look
'''for image_idx in range(len(images)):
    print(image_idx)
    image_name = images[image_idx]
    mask_name = masks[image_idx]
    image = np.load(os.path.join(image_folder,image_name)).astype(np.float32)
    mask = np.load(os.path.join(mask_folder,mask_name))

    plt.imshow(image)
    plt.title(image_name)
    plt.show()


    plt.imshow(mask)
    plt.title(mask_name)
    plt.show()'''


# %%
site_id = [images[i][0:5] for i in range(len(images))]
site_id = np.unique(site_id)

all = []
for site_id in site_id:

    images_id = [] 
    for i in range(len(images)):
        if images[i][0:5] in site_id:
            images_id.append(images[i])

    flat_data = []
    c=[]
    for images_id in images_id:
        image_name = images_id

        image = np.load(os.path.join(image_folder,image_name)).astype(np.float32)
        
        band_id = image_name[-7:-4]

        # flatten both
        image_flat = image[::8,::8].flatten()
        
        #image = image[~np.isnan(image)]
        #masked_array = masked_array[~np.isnan(masked_array)]

        flat_data.append(pd.DataFrame({band_id:image_flat}))

    for j in mlist:
        if site_id in j:
            mask_name = j

    mask = np.load(os.path.join(mask_folder,mask_name))
        # mask labels where image is nan
    m = np.isnan(image)
    # Mask array1 with values from array2 where array2 is not NaN
    masked_array = np.copy(mask.astype(float))
    masked_array[m] = np.nan
    test = masked_array[::8,::8]
    masked_array = masked_array[::8,::8].flatten()

    flat_data.append(pd.DataFrame({'label':masked_array}))

    flat_data = pd.concat(flat_data,axis=1)
    flat_data['site_id'] = site_id

    all.append(flat_data.dropna())
    break
    

# %%
pd.concat(all).reset_index(drop=True).to_feather('alldata')