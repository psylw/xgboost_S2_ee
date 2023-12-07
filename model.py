# %%

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

df = pd.read_feather('alldata')
df.loc[df.label ==2 , 'label'] = 1
df.loc[df.label ==1 , 'label'] = 0
df.loc[df.label ==3 , 'label'] = 1
# %%
# split data

site_id = df.site_id.unique()
train = site_id[0:55]
#train = site_id[0]
test = site_id[56:]
df_train = df.loc[df.site_id.isin(train)]
df_test = df.loc[df.site_id.isin(test)]

# %%
X_train,X_test,y_train,y_test = df_train.drop(columns='label'),df_test.drop(columns='label'),df_train.label,df_test.label

# %%
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(5)

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.svm import SVC

names = [
    "Random Forest",
    "Logistic Regression",
    "xgboost",
    "SVC"
]

classifiers = [
    RandomForestClassifier(random_state=0,n_jobs=-1),
    LogisticRegression(random_state=0,max_iter=600),
    xgb.XGBClassifier(random_state=0,n_jobs=-1),
    SVC(random_state=0,probability=True)
]

# %%
clf = RandomForestClassifier(random_state=0,n_jobs=-1)
param = [
    
    # RF
    {"n_estimators": [64, 100, 200,400], 
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
                                 "max_depth":[5,10,20,30],
                                          "min_samples_split":range(2,9),
                                         "min_samples_leaf":range(1,8)}]

# %%
# initial hyperparameter tuning
param = [
    
    # RF
    {"n_estimators": [64, 100, 200,400], 
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
                                 "max_depth":[5,10,20,30],
                                          "min_samples_split":range(2,9),
                                         "min_samples_leaf":range(1,8)},
    
    # logistic
    {
     'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}]},
    #xgb
    {"learning_rate": [0.5, 0.25, 0.1, 0.05, 0.01,.001], 
                                "n_estimators": [64, 100, 200,400,600],
                                "max_depth":[5,10,20,30],
                                 'min_child_weight': [1, 5, 10],
                                 'gamma': [1, 1.5, 2, 5,6],
                                 'subsample': [0.6, 0.8, 1.0],
                                 'colsample_bytree': [0.6, 0.8, 1.0],
                                'scale_pos_weight':[10,20,30,50]},

    # SVC
    {'C': [0.1, 1, 10, 100],'class_weight':['balanced',{0: 1.0, 1: 10.0},{0: 1.0, 1: 20.0},{0: 1.0, 1: 50.0}],
     'kernel':['linear', 'poly', 'rbf', 'sigmoid']} 
]
# %%
X_test = X_test.drop(columns='site_id')
X_train = X_train.drop(columns='site_id')

# %%
from sklearn.model_selection import RandomizedSearchCV


mod = RandomizedSearchCV(estimator=clf,
                    param_distributions = param[0],
                    n_iter=5, 
                    scoring=["f1"],
                    refit="f1",
                    cv=cv,
                    n_jobs=-1)

_ = mod.fit(X_train,y_train)  

mod.cv_results_
# %%
# look at test results with parameters, choose best model, save fitted model with parameters

params = {'n_estimators': 64,
 'min_samples_split': 5,
 'min_samples_leaf': 1,
 'max_depth': 10,
 'class_weight': 'balanced'}


clf = RandomForestClassifier(random_state=0,n_jobs=-1,**params)
clf.fit(X_train,y_train)

# %%
df['predict'] = clf.predict(df.drop(columns=['site_id','label']))
results = df.groupby('site_id').agg(list).predict
truth = df.groupby('site_id').agg(list).label
# %%
from sklearn.metrics import f1_score
f1 = [f1_score(truth.iloc[i],results.iloc[i]) for i in range(len(truth))]
# %%
green = df.groupby('site_id').agg(list)['_B3']
nir = df.groupby('site_id').agg(list)['_B8']
# %%

index = []
for i in range(len(green)):
    g = np.array(green.iloc[i])
    n = np.array(nir.iloc[i])

    i = (g-n)/(g+n)
    condition = i < .3

    # Create a new binary array based on the condition
    binary_array = np.where(condition, 0,1)
    index.append(binary_array)
# %%
f1_t = [f1_score(truth.iloc[i],index[i]) for i in range(len(truth))]

# %%
from os import listdir
import os
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

root_dir = os.getcwd()

image_folder = os.path.join(root_dir, 'band')
mask_folder = os.path.join(root_dir, 'label')
images = listdir(image_folder)

mlist = os.listdir(mask_folder)

sar_folder = os.path.join(root_dir, 'images')
sar = listdir(sar_folder)


# %%
site_id = [images[i][0:5] for i in range(len(images))]
site_id = np.unique(site_id)

for k,site_id in enumerate(site_id):
    sar_id = [] 
    for i in range(len(sar)):
        if sar[i][0:5] in site_id:
            sar[i]
            image = np.load(os.path.join(sar_folder,sar[i])).astype(np.float32)
            fig = plt.subplots(figsize=(3, 3))
            plt.imshow(image,cmap='gray')
            plt.title(sar[i])
            plt.axis('off') 
            plt.show()


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

        flat_data.append(pd.DataFrame({band_id:image_flat}))

    for j in mlist:
        if site_id in j:
            mask_name = j

    mask = np.load(os.path.join(mask_folder,mask_name))
        # mask labels where image is nan

    masked_array = mask[::8,::8].flatten()

    flat_data.append(pd.DataFrame({'label':masked_array}))

    flat_data = pd.concat(flat_data,axis=1)

    red_channel = flat_data._B4.values.reshape(128,128)
    green_channel = flat_data._B3.values.reshape(128,128)
    blue_channel = flat_data._B2.values.reshape(128,128)
    red_channel = (red_channel * 255).astype(np.uint8)
    green_channel = (green_channel * 255).astype(np.uint8)
    blue_channel = (blue_channel * 255).astype(np.uint8)
    # Stack the channels to form the RGB image
    rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)

    # Create a PIL Image from the RGB NumPy array
    image = Image.fromarray(rgb_image, 'RGB')

    # Convert the image to a NumPy array
    image_array = np.array(image)

    # Increase the brightness by scaling pixel values
    brightness_factor = 6  # You can adjust this value based on your preference
    brighter_image_array = np.clip(image_array * brightness_factor, 0, 255).astype(np.uint8)
    fig, axs = plt.subplots(1, 4, figsize=(12, 3))
    fig.suptitle(site_id)
    # Create a new Pillow Image from the modified array
    brighter_image = Image.fromarray(brighter_image_array)
    axs[0].imshow(brighter_image)
    axs[0].set_title('true color')
    axs[0].axis('off') 

    idx = flat_data.dropna().reset_index()['index'].values
    a = np.empty(128**2)
    a[:] = np.nan

    a[[idx]]=index[k]
    image_flat = a.reshape((128, 128))
    axs[1].imshow(image_flat,cmap='gray')
    axs[1].set_title('threshold, f1-score = '+str(f1_t[k])[0:4])
    axs[1].axis('off') 

    a = np.empty(128**2)
    a[:] = np.nan

    a[[idx]]=results[site_id]
    image_flat = a.reshape((128, 128))

    axs[2].imshow(image_flat,cmap='gray')
    axs[2].set_title('RF, f1-score = '+str(f1[k])[0:4])
    axs[2].axis('off') 

    a = np.empty(128**2)
    a[:] = np.nan

    a[[idx]]=truth[site_id]
    image_flat = a.reshape((128, 128))

    axs[3].imshow(image_flat,cmap='gray')
    axs[3].set_title('labels')
    axs[3].axis('off')
    plt.show()
    
    # %%
    from sklearn.inspection import permutation_importance

result = permutation_importance(
    clf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2,scoring='f1'
)

sorted_importances_idx = result.importances_mean.argsort()
importances = pd.DataFrame(
    result.importances[sorted_importances_idx].T,
    columns=X_test.columns[sorted_importances_idx],
)
ax = importances.plot.box(vert=False, whis=10)
ax.set_title("Permutation Importances (test set)")
ax.axvline(x=0, color="k", linestyle="--")
ax.set_xlabel("Decrease in f1 score")
ax.figure.tight_layout()


mdi_importances = pd.Series(
    clf.feature_importances_, index=X_test.columns
).sort_values(ascending=True)


ax = mdi_importances.plot.barh()
ax.set_title("Random Forest Feature Importances (MDI)")
ax.figure.tight_layout()

correlation_matrix = df.drop(columns=['site_id','label','predict','B12','_B2','_B4']).corr()

import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap of the correlation matrix
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")

# Show the plot
plt.show()
# %%

water = df.loc[df.label==1]
notwater = df.loc[df.label==0]
# %%
d = []
name = []
for i in ['_B2', '_B3', '_B4', '_B8','B11', 'B12']:
    d.append(water[i])
    d.append(notwater[i])
    name.append('water, ' + i)
    name.append('not water, ' + i)
# %%
plt.boxplot(d)
plt.xticks(list(range(0,12)),labels=name,rotation=45)

# %%
fig, axs = plt.subplots(2,3, figsize=(12,8), facecolor='w', edgecolor='k')
fig.subplots_adjust(hspace = .7, wspace=.3)

axs = axs.ravel()

title = ['B2 (blue)', 'B3 (green)', 'B4 (red)', 'B8 (NIR)','B11 (SWIR1)', 'B12 (SWIR1)']
j=0
c = ['_B2', '_B3', '_B4', '_B8','B11', 'B12']

lim = [[-0.02,.25],[-0.02,.3],[-0.02,.5],[-0.02,.5],[-0.02,.6],[-0.02,.5]]

for i in range(len(['_B2', '_B3', '_B4', '_B8','B11', 'B12'])):
    #column = feature_importance_new['Gradient Boosted'][0:3].index[i]
    
    ######COMENT IF MODEL CHANGES, UNCOMMENT IF NOT RUNNING FEATURE IMPORTANCE

    d = [water[c[i]],notwater[c[i]]]
    axs[i].boxplot(d)
    axs[i].set_xticks([1,2],labels=['water','not water'],rotation=45)
    axs[i].set_title(title[j])

    axs[i].set_ylim(lim[i])

    axs[i].set_ylabel('reflectance')
    j+=1
    

# %%
