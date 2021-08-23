#!/usr/bin/env python
# coding: utf-8

# # MNIST Image Classification with TensorFlow on Cloud AI Platform
# 
# This notebook demonstrates how to implement different image models on MNIST using the [tf.keras API](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras).
# 
# ## Learning Objectives
# 1. Understand how to build a Dense Neural Network (DNN) for image classification
# 2. Understand how to use dropout (DNN) for image classification
# 3. Understand how to use Convolutional Neural Networks (CNN)
# 4. Know how to deploy and use an image classifcation model using Google Cloud's [AI Platform](https://cloud.google.com/ai-platform/)
# 
# First things first. Configure the parameters below to match your own Google Cloud project details.

# In[ ]:


get_ipython().system('sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst')


# In[1]:


from datetime import datetime
import os

PROJECT = "your-project-id-here"  # REPLACE WITH YOUR PROJECT ID
BUCKET = "your-bucket-id-here"  # REPLACE WITH YOUR BUCKET NAME
REGION = "us-central1"  # REPLACE WITH YOUR BUCKET REGION e.g. us-central1
MODEL_TYPE = "cnn"  # "linear", "dnn", "dnn_dropout", or "dnn"

# Do not change these
os.environ["PROJECT"] = PROJECT
os.environ["BUCKET"] = BUCKET
os.environ["REGION"] = REGION
os.environ["MODEL_TYPE"] = MODEL_TYPE
os.environ["TFVERSION"] = "2.1"  # Tensorflow  version
os.environ["IMAGE_URI"] = os.path.join("gcr.io", PROJECT, "mnist_models")


# ## Building a dynamic model
# 
# In the previous notebook, <a href="mnist_linear.ipynb">mnist_linear.ipynb</a>, we ran our code directly from the notebook. In order to run it on the AI Platform, it needs to be packaged as a python module.
# 
# The boilerplate structure for this module has already been set up in the folder `mnist_models`. The module lives in the sub-folder, `trainer`, and is designated as a python package with the empty `__init__.py` (`mnist_models/trainer/__init__.py`) file. It still needs the model and a trainer to run it, so let's make them.
# 
# Let's start with the trainer file first. This file parses command line arguments to feed into the model.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mnist_models/trainer/task.py', 'import argparse\nimport json\nimport os\nimport sys\n\nfrom . import model\n\n\ndef _parse_arguments(argv):\n    """Parses command-line arguments."""\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\n        \'--model_type\',\n        help=\'Which model type to use\',\n        type=str, default=\'linear\')\n    parser.add_argument(\n        \'--epochs\',\n        help=\'The number of epochs to train\',\n        type=int, default=10)\n    parser.add_argument(\n        \'--steps_per_epoch\',\n        help=\'The number of steps per epoch to train\',\n        type=int, default=100)\n    parser.add_argument(\n        \'--job-dir\',\n        help=\'Directory where to save the given model\',\n        type=str, default=\'mnist_models/\')\n    return parser.parse_known_args(argv)\n\n\ndef main():\n    """Parses command line arguments and kicks off model training."""\n    args = _parse_arguments(sys.argv[1:])[0]\n\n    # Configure path for hyperparameter tuning.\n    trial_id = json.loads(\n        os.environ.get(\'TF_CONFIG\', \'{}\')).get(\'task\', {}).get(\'trial\', \'\')\n    output_path = args.job_dir if not trial_id else args.job_dir + \'/\'\n\n    model_layers = model.get_layers(args.model_type)\n    image_model = model.build_model(model_layers, args.job_dir)\n    model_history = model.train_and_evaluate(\n        image_model, args.epochs, args.steps_per_epoch, args.job_dir)\n\n\nif __name__ == \'__main__\':\n    main()')


# Next, let's group non-model functions into a util file to keep the model file simple. We'll copy over the `scale` and `load_dataset` functions from the previous lab.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mnist_models/trainer/util.py', 'import tensorflow as tf\n\n\ndef scale(image, label):\n    """Scales images from a 0-255 int range to a 0-1 float range"""\n    image = tf.cast(image, tf.float32)\n    image /= 255\n    image = tf.expand_dims(image, -1)\n    return image, label\n\n\ndef load_dataset(\n        data, training=True, buffer_size=5000, batch_size=100, nclasses=10):\n    """Loads MNIST dataset into a tf.data.Dataset"""\n    (x_train, y_train), (x_test, y_test) = data\n    x = x_train if training else x_test\n    y = y_train if training else y_test\n    # One-hot encode the classes\n    y = tf.keras.utils.to_categorical(y, nclasses)\n    dataset = tf.data.Dataset.from_tensor_slices((x, y))\n    dataset = dataset.map(scale).batch(batch_size)\n    if training:\n        dataset = dataset.shuffle(buffer_size).repeat()\n    return dataset')


# Finally, let's code the models! The [tf.keras API](https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras) accepts an array of [layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) into a [model object](https://www.tensorflow.org/api_docs/python/tf/keras/Model), so we can create a dictionary of layers based on the different model types we want to use. The below file has two functions: `get_layers` and `create_and_train_model`. We will build the structure of our model in `get_layers`. Last but not least, we'll copy over the training code from the previous lab into `train_and_evaluate`.
# 
# **TODO 1**: Define the Keras layers for a DNN model   
# **TODO 2**: Define the Keras layers for a dropout model  
# **TODO 3**: Define the Keras layers for a CNN model  
# 
# Hint: These models progressively build on each other. Look at the imported `tensorflow.keras.layers` modules and the default values for the variables defined in `get_layers` for guidance.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mnist_models/trainer/model.py', 'import os\nimport shutil\n\nimport matplotlib.pyplot as plt\nimport numpy as np\nimport tensorflow as tf\nfrom tensorflow.keras import Sequential\nfrom tensorflow.keras.callbacks import TensorBoard\nfrom tensorflow.keras.layers import (\n    Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Softmax)\n\nfrom . import util\n\n\n# Image Variables\nWIDTH = 28\nHEIGHT = 28\n\n\ndef get_layers(\n        model_type,\n        nclasses=10,\n        hidden_layer_1_neurons=400,\n        hidden_layer_2_neurons=100,\n        dropout_rate=0.25,\n        num_filters_1=64,\n        kernel_size_1=3,\n        pooling_size_1=2,\n        num_filters_2=32,\n        kernel_size_2=3,\n        pooling_size_2=2):\n    """Constructs layers for a keras model based on a dict of model types."""\n    model_layers = {\n        \'linear\': [\n            Flatten(),\n            Dense(nclasses),\n            Softmax()\n        ],\n        \'dnn\': [\n            # TODO\n        ],\n        \'dnn_dropout\': [\n            # TODO\n        ],\n        \'cnn\': [\n            # TODO\n        ]\n    }\n    return model_layers[model_type]\n\n\ndef build_model(layers, output_dir):\n    """Compiles keras model for image classification."""\n    model = Sequential(layers)\n    model.compile(optimizer=\'adam\',\n                  loss=\'categorical_crossentropy\',\n                  metrics=[\'accuracy\'])\n    return model\n\n\ndef train_and_evaluate(model, num_epochs, steps_per_epoch, output_dir):\n    """Compiles keras model and loads data into it for training."""\n    mnist = tf.keras.datasets.mnist.load_data()\n    train_data = util.load_dataset(mnist)\n    validation_data = util.load_dataset(mnist, training=False)\n\n    callbacks = []\n    if output_dir:\n        tensorboard_callback = TensorBoard(log_dir=output_dir)\n        callbacks = [tensorboard_callback]\n\n    history = model.fit(\n        train_data,\n        validation_data=validation_data,\n        epochs=num_epochs,\n        steps_per_epoch=steps_per_epoch,\n        verbose=2,\n        callbacks=callbacks)\n\n    if output_dir:\n        export_path = os.path.join(output_dir, \'keras_export\')\n        model.save(export_path, save_format=\'tf\')\n\n    return history')


# ## Local Training
# 
# With everything set up, let's run locally to test the code. Some of the previous tests have been copied over into a testing script `mnist_models/trainer/test.py` to make sure the model still passes our previous checks. On `line 13`, you can specify which model types you would like to check. `line 14` and `line 15` has the number of epochs and steps per epoch respectively.
# 
# Moment of truth! Run the code below to check your models against the unit tests. If you see "OK" at the end when it's finished running, congrats! You've passed the tests!

# In[ ]:


get_ipython().system('python3 -m mnist_models.trainer.test')


# Now that we know that our models are working as expected, let's run it on the [Google Cloud AI Platform](https://cloud.google.com/ml-engine/docs/). We can run it as a python module locally first using the command line.
# 
# The below cell transfers some of our variables to the command line as well as create a job directory including a timestamp.

# In[ ]:


current_time = datetime.now().strftime("%y%m%d_%H%M%S")
model_type = 'cnn'

os.environ["MODEL_TYPE"] = model_type
os.environ["JOB_DIR"] = "mnist_models/models/{}_{}/".format(
    model_type, current_time)


# The cell below runs the local version of the code. The epochs and steps_per_epoch flag can be changed to run for longer or shorther, as defined in our `mnist_models/trainer/task.py` file.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'python3 -m mnist_models.trainer.task \\\n    --job-dir=$JOB_DIR \\\n    --epochs=5 \\\n    --steps_per_epoch=50 \\\n    --model_type=$MODEL_TYPE')


# ## Training on the cloud
# 
# Since we're using an unreleased version of TensorFlow on AI Platform, we can instead use a [Deep Learning Container](https://cloud.google.com/ai-platform/deep-learning-containers/docs/overview) in order to take advantage of libraries and applications not normally packaged with AI Platform. Below is a simple [Dockerlife](https://docs.docker.com/engine/reference/builder/) which copies our code to be used in a TF2 environment.

# In[ ]:


get_ipython().run_cell_magic('writefile', 'mnist_models/Dockerfile', 'FROM gcr.io/deeplearning-platform-release/tf2-cpu\nCOPY mnist_models/trainer /mnist_models/trainer\nENTRYPOINT ["python3", "-m", "mnist_models.trainer.task"]')


# The below command builds the image and ships it off to Google Cloud so it can be used for AI Platform. When built, it will show up [here](http://console.cloud.google.com/gcr) with the name `mnist_models`. ([Click here](https://console.cloud.google.com/cloud-build) to enable Cloud Build)

# In[ ]:


get_ipython().system('docker build -f mnist_models/Dockerfile -t $IMAGE_URI ./')


# In[ ]:


get_ipython().system('docker push $IMAGE_URI')


# Finally, we can kickoff the [AI Platform training job](https://cloud.google.com/sdk/gcloud/reference/ai-platform/jobs/submit/training). We can pass in our docker image using the `master-image-uri` flag.

# In[ ]:


current_time = datetime.now().strftime("%y%m%d_%H%M%S")
model_type = 'cnn'

os.environ["MODEL_TYPE"] = model_type
os.environ["JOB_DIR"] = "gs://{}/mnist_{}_{}/".format(
    BUCKET, model_type, current_time)
os.environ["JOB_NAME"] = "mnist_{}_{}".format(
    model_type, current_time)


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'echo $JOB_DIR $REGION $JOB_NAME\ngcloud ai-platform jobs submit training $JOB_NAME \\\n    --staging-bucket=gs://$BUCKET \\\n    --region=$REGION \\\n    --master-image-uri=$IMAGE_URI \\\n    --scale-tier=BASIC_GPU \\\n    --job-dir=$JOB_DIR \\\n    -- \\\n    --model_type=$MODEL_TYPE')


# ## Deploying and predicting with model
# 
# Once you have a model you're proud of, let's deploy it! All we need to do is give AI Platform the location of the model. Below uses the keras export path of the previous job, but `${JOB_DIR}keras_export/` can always be changed to a different path.
# 
# Uncomment the delete commands below if you are getting an "already exists error" and want to deploy a new model.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'MODEL_NAME="mnist"\nMODEL_VERSION=${MODEL_TYPE}\nMODEL_LOCATION=${JOB_DIR}keras_export/\necho "Deleting and deploying $MODEL_NAME $MODEL_VERSION from $MODEL_LOCATION ... this will take a few minutes"\n#yes | gcloud ai-platform versions delete ${MODEL_VERSION} --model ${MODEL_NAME}\n#yes | gcloud ai-platform models delete ${MODEL_NAME}\ngcloud ai-platform models create ${MODEL_NAME} --regions $REGION\ngcloud ai-platform versions create ${MODEL_VERSION} \\\n    --model ${MODEL_NAME} \\\n    --origin ${MODEL_LOCATION} \\\n    --framework tensorflow \\\n    --runtime-version=2.1')


# To predict with the model, let's take one of the example images.
# 
# **TODO 4**: Write a `.json` file with image data to send to an AI Platform deployed model

# In[ ]:


import json, codecs
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_models.trainer import util

HEIGHT = 28
WIDTH = 28
IMGNO = 12

mnist = tf.keras.datasets.mnist.load_data()
(x_train, y_train), (x_test, y_test) = mnist
test_image = x_test[IMGNO]

jsondata = test_image.reshape(HEIGHT, WIDTH, 1).tolist()
json.dump(jsondata, codecs.open("test.json", "w", encoding = "utf-8"))
plt.imshow(test_image.reshape(HEIGHT, WIDTH));


# Finally, we can send it to the prediction service. The output will have a 1 in the index of the corresponding digit it is predicting. Congrats! You've completed the lab!

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud ai-platform predict \\\n    --model=mnist \\\n    --version=${MODEL_TYPE} \\\n    --json-instances=./test.json')


# Copyright 2020 Google Inc.
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
