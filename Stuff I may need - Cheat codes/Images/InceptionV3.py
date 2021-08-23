#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from google.cloud import storage
from matplotlib import pyplot as plt
from base64 import b64encode
from PIL import Image
import numpy as np
import io
from sklearn import preprocessing
from shutil import copy2


# In[2]:


# Download the inception v3 weights
get_ipython().system('wget --no-check-certificate     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Create an instance of the inception model from the local pre-trained weights
local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (512, 512, 3), 
                                include_top = False, 
                                weights = None)

pre_trained_model.load_weights(local_weights_file)

# Make all the layers in the pre-trained model non-trainable
for layer in pre_trained_model.layers:
    layer.trainable = False


# In[3]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# In[4]:


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.999):
      print("\nReached 99.9% accuracy so cancelling training!")
      self.model.stop_training = True


# In[5]:


from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)              
# Add a final sigmoid layer for classification
x = layers.Dense  (1, activation='sigmoid')(x)             

model = Model( pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])

model.summary()


# In[6]:


def split_dataset(BASE_PATH, DATASET_PATH = 'dataset', train_images = 200, val_images = 7):
    # Specify path to the downloaded folder
    classes = os.listdir(BASE_PATH)

    # Specify path for copying the dataset into train and val sets
    os.makedirs(DATASET_PATH, exist_ok=True)

    # Creating train directory
    train_dir = os.path.join(DATASET_PATH, 'train')
    os.makedirs(train_dir, exist_ok=True)

    # Creating val directory
    val_dir = os.path.join(DATASET_PATH, 'val')
    os.makedirs(val_dir, exist_ok=True)    

    # Copying images from original folder to dataset folder
    for class_name in classes:
        if len(class_name.split('.')) >= 2:
            continue
        print(f"Copying images for {class_name}...")
        
        # Creating destination folder (train and val)
        class_train_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        
        class_val_dir = os.path.join(val_dir, class_name)
        os.makedirs(class_val_dir, exist_ok=True)

        # Shuffling the image list
        class_path = os.path.join(BASE_PATH, class_name)
        class_images = os.listdir(class_path)
        np.random.shuffle(class_images)

        for image in class_images[:train_images]:
            copy2(os.path.join(class_path, image), class_train_dir)
        for image in class_images[train_images:train_images+val_images]:
            copy2(os.path.join(class_path, image), class_val_dir)


# In[7]:


split_dataset('Dataset')


# In[8]:


train_dir = './dataset/train'
validation_dir = './dataset/val'

train_ProdutoConforme_dir = os.path.join(train_dir, 'ProdutoConforme') 
train_ProdutoConforme_fnames = os.listdir(train_ProdutoConforme_dir)

train_Prega_dir = os.path.join(train_dir, 'Prega') 
train_Prega_fnames = os.listdir(train_Prega_dir)


train_Sujidade_dir = os.path.join(train_dir, 'Sujidade') 
train_Sujidade_fnames = os.listdir(train_Sujidade_dir)



print(len(train_ProdutoConforme_fnames))
print(len(train_Prega_fnames))
print(len(train_Sujidade_fnames))


# In[9]:


validation_ProdutoConforme_dir = os.path.join(validation_dir, 'ProdutoConforme') 
validation_ProdutoConforme_fnames =  os.listdir(validation_ProdutoConforme_dir)

validation_Prega_dir = os.path.join(validation_dir, 'Prega') 
validation_Prega_fnames =  os.listdir(validation_Prega_dir)

validation_Sujidade_dir = os.path.join(validation_dir, 'Sujidade') 
validation_Sujidade_fnames =  os.listdir(validation_Sujidade_dir)

print(len(validation_ProdutoConforme_fnames))
print(len(validation_Prega_fnames))
print(len(validation_Sujidade_fnames))


# In[10]:


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale = 1.0/255.)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 4,
                                                    class_mode = 'binary', 
                                                    target_size = (512, 512))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                          batch_size  = 2 ,
                                                          class_mode  = 'binary', 
                                                          target_size = (512, 512))


# In[ ]:


callbacks = myCallback()
history = model.fit(train_generator,
            validation_data = validation_generator,
            steps_per_epoch = int(868/4),
            epochs = 30, callbacks=[callbacks],
            validation_steps = int(42/2),
            verbose = 2)


# In[ ]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()


# In[ ]:




