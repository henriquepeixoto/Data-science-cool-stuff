#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import os
import shutil
from collections import Counter
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, UpSampling2D,    Dense, Layer, Reshape, InputLayer, Flatten, Input, MaxPooling2D
from alibi_detect.od import OutlierAE
from alibi_detect.utils.visualize import plot_instance_score, plot_feature_outlier_image
import io
import random
import numpy as np
import PIL
import tensorflow as tf

from matplotlib import pyplot as plt
from base64 import b64encode
from google.cloud import storage


# In[2]:


path_train = 'ProdutoConforme/'
path_test = 'Sujidade na solda/'


# In[3]:


def img_to_np(foldername):
    client = storage.Client()

    bucket = client.get_bucket('imagens-para-treinamento-do-modelo')
    blobs = bucket.list_blobs(prefix=foldername)
    by = []
    img_array = []
    label = []
    for idx, bl in enumerate(blobs):
        if idx == 0:
            continue
        data = bl.download_as_string()
        by.append(data)
    for i in by:
        b = io.BytesIO(i)
        img = Image.open(b).convert("RGB")
        img = img.resize((64,64))
        img_array.append(np.asarray(img))
    images = np.array(img_array)
    return images


# In[4]:


train = img_to_np(path_train)
test = img_to_np(path_test)


# In[5]:


train = train.astype('float32') / 255.
test = test.astype('float32') / 255.


# In[13]:


train[0].shape


# In[22]:


encoding_dim = 1024
dense_dim = [32, 32, 128]


# In[23]:


encoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=train[0].shape),
      Conv2D(128, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(1024, 4, strides=2, padding='same', activation=tf.nn.relu),

      Flatten(),
      Dense(encoding_dim,)
  ])

decoder_net = tf.keras.Sequential(
  [
      InputLayer(input_shape=(encoding_dim,)),
      Dense(np.prod(dense_dim)),
      Reshape(target_shape=dense_dim),
      Conv2D(1024, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2D(512, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(256, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(64, 4, strides=2, padding='same', activation=tf.nn.relu),
      Conv2DTranspose(3, 4, strides=2, padding='same', activation='sigmoid')
  ])


# In[32]:


od = OutlierAE( threshold = 0.01,
                encoder_net=encoder_net,
                decoder_net=decoder_net)


# In[33]:



adam = tf.keras.optimizers.Adam(lr=1e-4)

model = od.fit(train, epochs=100, verbose=True, optimizer = adam)

od.infer_threshold(test, threshold_perc=95)

preds = od.predict(test, outlier_type='instance',
            return_instance_score=True,
            return_feature_score=True)


# In[34]:


predi = preds['data']['is_outlier']
name = []

client = storage.Client()
bucket = client.get_bucket('imagens-para-treinamento-do-modelo')

for blob in bucket.list_blobs(prefix=path_test):
    name.append(str(blob.name))
name.remove(name[0]) 
df = pd.DataFrame(name)
df['Pred'] = predi

recon = od.ae(test).numpy()

plot_feature_outlier_image(preds, test, 
                           X_recon=recon,  
                           max_instances=5,
                           outliers_only=True,
                           figsize=(64,64))


# In[ ]:





# In[35]:


predi


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




