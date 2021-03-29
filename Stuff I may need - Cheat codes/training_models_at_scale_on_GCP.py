#!/usr/bin/env python
# coding: utf-8

# # Training Models at Scale with AI Platform
# **Learning Objectives:**
#   1. Learn how to organize your training code into a Python package
#   1. Train your model using cloud infrastructure via Google Cloud AI Platform Training Service
#   1. (optional) Learn how to run your training package using Docker containers and push training Docker images on a Docker registry
# 
# ## Introduction
# 
# In this notebook we'll make the jump from training locally, to do training in the cloud. We'll take advantage of Google Cloud's [AI Platform Training Service](https://cloud.google.com/ai-platform/). 
# 
# AI Platform Training Service is a managed service that allows the training and deployment of ML models without having to provision or maintain servers. The infrastructure is handled seamlessly by the managed service for us.
# 
# Each learning objective will correspond to a __#TODO__ in the [student lab notebook](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive2/art_and_science_of_ml/labs/training_models_at_scale.ipynb) -- try to complete that notebook first before reviewing this solution notebook.

# In[ ]:


# Use the chown command to change the ownership of repository to user
get_ipython().system('sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst')


# In[ ]:


# Install the Google Cloud BigQuery
get_ipython().system('pip install --user google-cloud-bigquery==1.25.0')


# **Note**: Restart your kernel to use updated packages.

# Kindly ignore the deprecation warnings and incompatibility errors related to google-cloud-storage.

# Specify your project name, bucket name and region in the cell below.

# In[ ]:


# The OS module in Python provides functions for interacting with the operating system
import os

from google.cloud import bigquery


# In[ ]:


# Change with your own bucket and project below:
BUCKET =  "<BUCKET>"
PROJECT = "<PROJECT>"
REGION = "<YOUR REGION>"

OUTDIR = "gs://{bucket}/taxifare/data".format(bucket=BUCKET)

# Store the value of `BUCKET`, `OUTDIR`, `PROJECT`, `REGION` and `TFVERSION` in environment variables.
os.environ['BUCKET'] = BUCKET
os.environ['OUTDIR'] = OUTDIR
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = "2.1"


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Create BigQuery tables

# If you have not already created a BigQuery dataset for our data, run the following cell:

# In[ ]:


# Created a BigQuery dataset for our data
bq = bigquery.Client(project = PROJECT)
dataset = bigquery.Dataset(bq.dataset("taxifare"))

try:
    bq.create_dataset(dataset)
    print("Dataset created")
except:
    print("Dataset already exists")


# Let's create a table with 1 million examples.
# 
# Note that the order of columns is exactly what was in our CSV files.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\n# Creating the table in our dataset.\nCREATE OR REPLACE TABLE taxifare.feateng_training_data AS\n\nSELECT\n    (tolls_amount + fare_amount) AS fare_amount,\n    pickup_datetime,\n    pickup_longitude AS pickuplon,\n    pickup_latitude AS pickuplat,\n    dropoff_longitude AS dropofflon,\n    dropoff_latitude AS dropofflat,\n    passenger_count*1.0 AS passengers,\n    'unused' AS key\nFROM `nyc-tlc.yellow.trips`\nWHERE ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 1000)) = 1\nAND\n    trip_distance > 0\n    AND fare_amount >= 2.5\n    AND pickup_longitude > -78\n    AND pickup_longitude < -70\n    AND dropoff_longitude > -78\n    AND dropoff_longitude < -70\n    AND pickup_latitude > 37\n    AND pickup_latitude < 45\n    AND dropoff_latitude > 37\n    AND dropoff_latitude < 45\n    AND passenger_count > 0")


# Make the validation dataset be 1/10 the size of the training dataset.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\n# Creating the table in our dataset.\nCREATE OR REPLACE TABLE taxifare.feateng_valid_data AS\n\nSELECT\n    (tolls_amount + fare_amount) AS fare_amount,\n    pickup_datetime,\n    pickup_longitude AS pickuplon,\n    pickup_latitude AS pickuplat,\n    dropoff_longitude AS dropofflon,\n    dropoff_latitude AS dropofflat,\n    passenger_count*1.0 AS passengers,\n    'unused' AS key\nFROM `nyc-tlc.yellow.trips`\nWHERE ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 10000)) = 2\nAND\n    trip_distance > 0\n    AND fare_amount >= 2.5\n    AND pickup_longitude > -78\n    AND pickup_longitude < -70\n    AND dropoff_longitude > -78\n    AND dropoff_longitude < -70\n    AND pickup_latitude > 37\n    AND pickup_latitude < 45\n    AND dropoff_latitude > 37\n    AND dropoff_latitude < 45\n    AND passenger_count > 0")


# ## Export the tables as CSV files

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\n# Deleting the current contents of output directory.\necho "Deleting current contents of $OUTDIR"\ngsutil -m -q rm -rf $OUTDIR\n\n# Fetching the training data to output directory.\necho "Extracting training data to $OUTDIR"\nbq --location=US extract \\\n   --destination_format CSV  \\\n   --field_delimiter "," --noprint_header \\\n   taxifare.feateng_training_data \\\n   $OUTDIR/taxi-train-*.csv\n\necho "Extracting validation data to $OUTDIR"\nbq --location=US extract \\\n   --destination_format CSV  \\\n   --field_delimiter "," --noprint_header \\\n   taxifare.feateng_valid_data \\\n   $OUTDIR/taxi-valid-*.csv\n\n# The `ls` command will show the content of working directory\ngsutil ls -l $OUTDIR')


# In[ ]:


# The `cat` command will outputs the contents of one or more URLs
# Using `head -2` we are showing only top two output files
get_ipython().system('gsutil cat gs://$BUCKET/taxifare/data/taxi-train-000000000000.csv | head -2')


# ## Make code compatible with AI Platform Training Service
# In order to make our code compatible with AI Platform Training Service we need to make the following changes:
# 
# 1. Upload data to Google Cloud Storage 
# 2. Move code into a trainer Python package
# 4. Submit training job with `gcloud` to train on AI Platform

# ### Upload data to Google Cloud Storage (GCS)
# 
# Cloud services don't have access to our local files, so we need to upload them to a location the Cloud servers can read from. In this case we'll use GCS.

# In[ ]:


# The `ls` command will show the content of working directory
get_ipython().system('gsutil ls gs://$BUCKET/taxifare/data')


# ### Move code into a Python package
# 
# 
# The first thing to do is to convert your training code snippets into a regular Python package that we will then `pip install` into the Docker container. 
# 
# A Python package is simply a collection of one or more `.py` files along with an `__init__.py` file to identify the containing directory as a package. The `__init__.py` sometimes contains initialization code but for our purposes an empty file suffices.

# #### Create the package directory

# Our package directory contains 3 files:

# In[ ]:


ls ./taxifare/trainer/


# #### Paste existing code into model.py
# 
# A Python package requires our code to be in a .py file, as opposed to notebook cells. So, we simply copy and paste our existing code for the previous notebook into a single file.

# In the cell below, we write the contents of the cell into `model.py` packaging the model we 
# developed in the previous labs so that we can deploy it to AI Platform Training Service.  

# In[1]:


get_ipython().run_cell_magic('writefile', './taxifare/trainer/model.py', '# The datetime module used to work with dates as date objects.\nimport datetime\n# The logging module in Python allows writing status messages to a file or any other output streams. \nimport logging\n# The OS module in Python provides functions for interacting with the operating system\nimport os\n# The shutil module in Python provides many functions of high-level operations on files and collections of files.\n# This module helps in automating process of copying and removal of files and directories.\nimport shutil\n\n# Here we\'ll import data processing libraries like Numpy and Tensorflow\nimport numpy as np\nimport tensorflow as tf\n\nfrom tensorflow.keras import activations\nfrom tensorflow.keras import callbacks\nfrom tensorflow.keras import layers\nfrom tensorflow.keras import models\n\nfrom tensorflow import feature_column as fc\n\nlogging.info(tf.version.VERSION)\n\n\n# Defining the feature names into a list `CSV_COLUMNS`\nCSV_COLUMNS = [\n        \'fare_amount\',\n        \'pickup_datetime\',\n        \'pickup_longitude\',\n        \'pickup_latitude\',\n        \'dropoff_longitude\',\n        \'dropoff_latitude\',\n        \'passenger_count\',\n        \'key\',\n]\nLABEL_COLUMN = \'fare_amount\'\n# Defining the default values into a list `DEFAULTS`\nDEFAULTS = [[0.0], [\'na\'], [0.0], [0.0], [0.0], [0.0], [0.0], [\'na\']]\nDAYS = [\'Sun\', \'Mon\', \'Tue\', \'Wed\', \'Thu\', \'Fri\', \'Sat\']\n\n\ndef features_and_labels(row_data):\n    for unwanted_col in [\'key\']:\n        row_data.pop(unwanted_col)\n# The .pop() method will return item and drop from frame.\n    label = row_data.pop(LABEL_COLUMN)\n    return row_data, label\n\n\ndef load_dataset(pattern, batch_size, num_repeat):\n# The tf.data.experimental.make_csv_dataset() method reads CSV files into a dataset\n    dataset = tf.data.experimental.make_csv_dataset(\n        file_pattern=pattern,\n        batch_size=batch_size,\n        column_names=CSV_COLUMNS,\n        column_defaults=DEFAULTS,\n        num_epochs=num_repeat,\n    )\n# The `map()` function executes a specified function for each item in an iterable.\n# The item is sent to the function as a parameter.\n    return dataset.map(features_and_labels)\n\n\ndef create_train_dataset(pattern, batch_size):\n    dataset = load_dataset(pattern, batch_size, num_repeat=None)\n# The `prefetch()` method will start a background thread to populate a ordered buffer that acts like a queue, so that downstream pipeline stages need not block.\n    return dataset.prefetch(1)\n\n\ndef create_eval_dataset(pattern, batch_size):\n    dataset = load_dataset(pattern, batch_size, num_repeat=1)\n# The `prefetch()` method will start a background thread to populate a ordered buffer that acts like a queue, so that downstream pipeline stages need not block.\n    return dataset.prefetch(1)\n\n\ndef parse_datetime(s):\n    if type(s) is not str:\n        s = s.numpy().decode(\'utf-8\')\n    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %Z")\n\n\ndef euclidean(params):\n    lon1, lat1, lon2, lat2 = params\n    londiff = lon2 - lon1\n    latdiff = lat2 - lat1\n    return tf.sqrt(londiff*londiff + latdiff*latdiff)\n\n\ndef get_dayofweek(s):\n    ts = parse_datetime(s)\n    return DAYS[ts.weekday()]\n\n\n@tf.function\ndef dayofweek(ts_in):\n    return tf.map_fn(\n        lambda s: tf.py_function(get_dayofweek, inp=[s], Tout=tf.string),\n        ts_in\n    )\n\n\n@tf.function\ndef fare_thresh(x):\n    return 60 * activations.relu(x)\n\n\ndef transform(inputs, NUMERIC_COLS, STRING_COLS, nbuckets):\n    # Pass-through columns\n    transformed = inputs.copy()\n    del transformed[\'pickup_datetime\']\n\n    feature_columns = {\n        colname: fc.numeric_column(colname)\n        for colname in NUMERIC_COLS\n    }\n\n    # Scaling longitude from range [-70, -78] to [0, 1]\n    for lon_col in [\'pickup_longitude\', \'dropoff_longitude\']:\n        transformed[lon_col] = layers.Lambda(\n            lambda x: (x + 78)/8.0,\n            name=\'scale_{}\'.format(lon_col)\n        )(inputs[lon_col])\n\n    # Scaling latitude from range [37, 45] to [0, 1]\n    for lat_col in [\'pickup_latitude\', \'dropoff_latitude\']:\n        transformed[lat_col] = layers.Lambda(\n            lambda x: (x - 37)/8.0,\n            name=\'scale_{}\'.format(lat_col)\n        )(inputs[lat_col])\n\n    # Adding Euclidean dist (no need to be accurate: NN will calibrate it)\n    transformed[\'euclidean\'] = layers.Lambda(euclidean, name=\'euclidean\')([\n        inputs[\'pickup_longitude\'],\n        inputs[\'pickup_latitude\'],\n        inputs[\'dropoff_longitude\'],\n        inputs[\'dropoff_latitude\']\n    ])\n    feature_columns[\'euclidean\'] = fc.numeric_column(\'euclidean\')\n\n    # hour of day from timestamp of form \'2010-02-08 09:17:00+00:00\'\n    transformed[\'hourofday\'] = layers.Lambda(\n        lambda x: tf.strings.to_number(\n            tf.strings.substr(x, 11, 2), out_type=tf.dtypes.int32),\n        name=\'hourofday\'\n    )(inputs[\'pickup_datetime\'])\n    feature_columns[\'hourofday\'] = fc.indicator_column(\n        fc.categorical_column_with_identity(\n            \'hourofday\', num_buckets=24))\n\n    latbuckets = np.linspace(0, 1, nbuckets).tolist()\n    lonbuckets = np.linspace(0, 1, nbuckets).tolist()\n    b_plat = fc.bucketized_column(\n        feature_columns[\'pickup_latitude\'], latbuckets)\n    b_dlat = fc.bucketized_column(\n            feature_columns[\'dropoff_latitude\'], latbuckets)\n    b_plon = fc.bucketized_column(\n            feature_columns[\'pickup_longitude\'], lonbuckets)\n    b_dlon = fc.bucketized_column(\n            feature_columns[\'dropoff_longitude\'], lonbuckets)\n    ploc = fc.crossed_column(\n            [b_plat, b_plon], nbuckets * nbuckets)\n    dloc = fc.crossed_column(\n            [b_dlat, b_dlon], nbuckets * nbuckets)\n    pd_pair = fc.crossed_column([ploc, dloc], nbuckets ** 4)\n    feature_columns[\'pickup_and_dropoff\'] = fc.embedding_column(\n            pd_pair, 100)\n\n    return transformed, feature_columns\n\n\ndef rmse(y_true, y_pred):\n    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))\n\n\ndef build_dnn_model(nbuckets, nnsize, lr):\n    # input layer is all float except for pickup_datetime which is a string\n    STRING_COLS = [\'pickup_datetime\']\n    NUMERIC_COLS = (\n            set(CSV_COLUMNS) - set([LABEL_COLUMN, \'key\']) - set(STRING_COLS)\n    )\n    inputs = {\n        colname: layers.Input(name=colname, shape=(), dtype=\'float32\')\n        for colname in NUMERIC_COLS\n    }\n    inputs.update({\n        colname: layers.Input(name=colname, shape=(), dtype=\'string\')\n        for colname in STRING_COLS\n    })\n\n    # transforms\n    transformed, feature_columns = transform(\n        inputs, NUMERIC_COLS, STRING_COLS, nbuckets=nbuckets)\n    dnn_inputs = layers.DenseFeatures(feature_columns.values())(transformed)\n\n    x = dnn_inputs\n    for layer, nodes in enumerate(nnsize):\n        x = layers.Dense(nodes, activation=\'relu\', name=\'h{}\'.format(layer))(x)\n    output = layers.Dense(1, name=\'fare\')(x)\n\n    model = models.Model(inputs, output)\n    #TODO 1a\n    lr_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)\n    model.compile(optimizer=lr_optimizer, loss=\'mse\', metrics=[rmse, \'mse\'])\n    \n    return model\n\n\ndef train_and_evaluate(hparams):\n    #TODO 1b\n    batch_size = hparams[\'batch_size\'] \n    nbuckets = hparams[\'nbuckets\'] \n    lr = hparams[\'lr\'] \n    nnsize = hparams[\'nnsize\']\n    eval_data_path = hparams[\'eval_data_path\']\n    num_evals = hparams[\'num_evals\']\n    num_examples_to_train_on = hparams[\'num_examples_to_train_on\']\n    output_dir = hparams[\'output_dir\']\n    train_data_path = hparams[\'train_data_path\']\n\n    timestamp = datetime.datetime.now().strftime(\'%Y%m%d%H%M%S\')\n    savedmodel_dir = os.path.join(output_dir, \'export/savedmodel\')\n    model_export_path = os.path.join(savedmodel_dir, timestamp)\n    checkpoint_path = os.path.join(output_dir, \'checkpoints\')\n    tensorboard_path = os.path.join(output_dir, \'tensorboard\')\n\n    if tf.io.gfile.exists(output_dir):\n        tf.io.gfile.rmtree(output_dir)\n\n    model = build_dnn_model(nbuckets, nnsize, lr)\n    logging.info(model.summary())\n\n    trainds = create_train_dataset(train_data_path, batch_size)\n    evalds = create_eval_dataset(eval_data_path, batch_size)\n\n    steps_per_epoch = num_examples_to_train_on // (batch_size * num_evals)\n\n    checkpoint_cb = callbacks.ModelCheckpoint(\n        checkpoint_path,\n        save_weights_only=True,\n        verbose=1\n    )\n    tensorboard_cb = callbacks.TensorBoard(tensorboard_path)\n\n    history = model.fit(\n        trainds,\n        validation_data=evalds,\n        epochs=num_evals,\n        steps_per_epoch=max(1, steps_per_epoch),\n        verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch\n        callbacks=[checkpoint_cb, tensorboard_cb]\n    )\n\n    # Exporting the model with default serving function.\n    tf.saved_model.save(model, model_export_path)\n    return history')


# ### Modify code to read data from and write checkpoint files to GCS 
# 
# If you look closely above, you'll notice a new function, `train_and_evaluate` that wraps the code that actually trains the model. This allows us to parametrize the training by passing a dictionary of parameters to this function (e.g, `batch_size`, `num_examples_to_train_on`, `train_data_path` etc.)
# 
# This is useful because the output directory, data paths and number of train steps will be different depending on whether we're training locally or in the cloud. Parametrizing allows us to use the same code for both.
# 
# We specify these parameters at run time via the command line. Which means we need to add code to parse command line parameters and invoke `train_and_evaluate()` with those params. This is the job of the `task.py` file. 

# In[2]:


get_ipython().run_cell_magic('writefile', 'taxifare/trainer/task.py', '# The argparse module makes it easy to write user-friendly command-line interfaces. It parses the defined arguments from the `sys.argv`.\n# The argparse module also automatically generates help & usage messages and issues errors when users give the program invalid arguments.\nimport argparse\n\nfrom trainer import model\n\n\n# Write an `task.py` file for adding code to parse command line parameters and invoke `train_and_evaluate()` with those parameters. \nif __name__ == \'__main__\':\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\n        "--batch_size",\n        help="Batch size for training steps",\n        type=int,\n        default=32\n    )\n    parser.add_argument(\n        "--eval_data_path",\n        help="GCS location pattern of eval files",\n        required=True\n    )\n    parser.add_argument(\n        "--nnsize",\n        help="Hidden layer sizes (provide space-separated sizes)",\n        nargs="+",\n        type=int,\n        default=[32, 8]\n    )\n    parser.add_argument(\n        "--nbuckets",\n        help="Number of buckets to divide lat and lon with",\n        type=int,\n        default=10\n    )\n    parser.add_argument(\n        "--lr",\n        help = "learning rate for optimizer",\n        type = float,\n        default = 0.001\n    )\n    parser.add_argument(\n        "--num_evals",\n        help="Number of times to evaluate model on eval data training.",\n        type=int,\n        default=5\n    )\n    parser.add_argument(\n        "--num_examples_to_train_on",\n        help="Number of examples to train on.",\n        type=int,\n        default=100\n    )\n    parser.add_argument(\n        "--output_dir",\n        help="GCS location to write checkpoints and export models",\n        required=True\n    )\n    parser.add_argument(\n        "--train_data_path",\n        help="GCS location pattern of train files containing eval URLs",\n        required=True\n    )\n    parser.add_argument(\n        "--job-dir",\n        help="this model ignores this field, but it is required by gcloud",\n        default="junk"\n    )\n    args = parser.parse_args()\n    hparams = args.__dict__\n    hparams.pop("job-dir", None)\n\n    model.train_and_evaluate(hparams)')


# ### Run trainer module package locally
# 
# Now we can test our training code locally as follows using the local test data. We'll run a very small training job over a single file with a small batch size and one eval step.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\n# Testing our training code locally\nEVAL_DATA_PATH=./taxifare/tests/data/taxi-valid*\nTRAIN_DATA_PATH=./taxifare/tests/data/taxi-train*\nOUTPUT_DIR=./taxifare-model\n\ntest ${OUTPUT_DIR} && rm -rf ${OUTPUT_DIR}\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare\n    \npython3 -m trainer.task \\\n--eval_data_path $EVAL_DATA_PATH \\\n--output_dir $OUTPUT_DIR \\\n--train_data_path $TRAIN_DATA_PATH \\\n--batch_size 5 \\\n--num_examples_to_train_on 100 \\\n--num_evals 1 \\\n--nbuckets 10 \\\n--lr 0.001 \\\n--nnsize 32 8')


# ### Run your training package on Cloud AI Platform
# 
# Once the code works in standalone mode locally, you can run it on Cloud AI Platform. To submit to the Cloud we use [`gcloud ai-platform jobs submit training [jobname]`](https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training) and simply specify some additional parameters for AI Platform Training Service:
# - jobid: A unique identifier for the Cloud job. We usually append system time to ensure uniqueness
# - region: Cloud region to train in. See [here](https://cloud.google.com/ml-engine/docs/tensorflow/regions) for supported AI Platform Training Service regions
# 
# The arguments before `-- \` are for AI Platform Training Service.
# The arguments after `-- \` are sent to our `task.py`.
# 
# Because this is on the entire dataset, it will take a while. You can monitor the job from the GCP console in the Cloud AI Platform section.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\n# Output directory and jobID\nOUTDIR=gs://${BUCKET}/taxifare/trained_model_$(date -u +%y%m%d_%H%M%S)\nJOBID=taxifare_$(date -u +%y%m%d_%H%M%S)\necho ${OUTDIR} ${REGION} ${JOBID}\ngsutil -m rm -rf ${OUTDIR}\n\n# Model and training hyperparameters\nBATCH_SIZE=50\nNUM_EXAMPLES_TO_TRAIN_ON=100\nNUM_EVALS=100\nNBUCKETS=10\nLR=0.001\nNNSIZE="32 8"\n\n# GCS paths\nGCS_PROJECT_PATH=gs://$BUCKET/taxifare\nDATA_PATH=$GCS_PROJECT_PATH/data\nTRAIN_DATA_PATH=$DATA_PATH/taxi-train*\nEVAL_DATA_PATH=$DATA_PATH/taxi-valid*\n\n#TODO 2\ngcloud ai-platform jobs submit training $JOBID \\\n    --module-name=trainer.task \\\n    --package-path=taxifare/trainer \\\n    --staging-bucket=gs://${BUCKET} \\\n    --python-version=3.7 \\\n    --runtime-version=${TFVERSION} \\\n    --region=${REGION} \\\n    -- \\\n    --eval_data_path $EVAL_DATA_PATH \\\n    --output_dir $OUTDIR \\\n    --train_data_path $TRAIN_DATA_PATH \\\n    --batch_size $BATCH_SIZE \\\n    --num_examples_to_train_on $NUM_EXAMPLES_TO_TRAIN_ON \\\n    --num_evals $NUM_EVALS \\\n    --nbuckets $NBUCKETS \\\n    --lr $LR \\\n    --nnsize $NNSIZE ')


# ### (Optional) Run your training package using Docker container
# 
# AI Platform Training also supports training in custom containers, allowing users to bring their own Docker containers with any pre-installed ML framework or algorithm to run on AI Platform Training. 
# 
# In this last section, we'll see how to submit a Cloud training job using a customized Docker image. 

# Containerizing our `./taxifare/trainer` package involves 3 steps:
# 
# * Writing a Dockerfile in `./taxifare`
# * Building the Docker image
# * Pushing it to the Google Cloud container registry in our GCP project

# The `Dockerfile` specifies
# 1. How the container needs to be provisioned so that all the dependencies in our code are satisfied
# 2. Where to copy our trainer Package in the container and how to install it (`pip install /trainer`)
# 3. What command to run when the container is ran (the `ENTRYPOINT` line)

# In[ ]:


get_ipython().run_cell_magic('writefile', './taxifare/Dockerfile', '# Writing the Dockerfile\nFROM gcr.io/deeplearning-platform-release/tf2-cpu\n# TODO 3\n\nCOPY . /code\n\nWORKDIR /code\n\nENTRYPOINT ["python3", "-m", "trainer.task"]')


# In[ ]:


get_ipython().system('gcloud auth configure-docker')


# In[ ]:


get_ipython().run_cell_magic('bash', '', '\n# Pushing the docker image to Google Cloud container registry in our GCP project\nPROJECT_DIR=$(cd ./taxifare && pwd)\nPROJECT_ID=$(gcloud config list project --format "value(core.project)")\nIMAGE_NAME=taxifare_training_container\nDOCKERFILE=$PROJECT_DIR/Dockerfile\nIMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME\n\ndocker build $PROJECT_DIR -f $DOCKERFILE -t $IMAGE_URI\n\ndocker push $IMAGE_URI')


# **Remark:** If you prefer to build the container image from the command line, we have written a script for that `./taxifare/scripts/build.sh`. This script reads its configuration from the file `./taxifare/scripts/env.sh`. You can configure these arguments the way you want in that file. You can also simply type `make build` from within `./taxifare` to build the image (which will invoke the build script). Similarly, we wrote the script `./taxifare/scripts/push.sh` to push the Docker image, which you can also trigger by typing `make push` from within `./taxifare`.

# ### Train using a custom container on AI Platform
# 
# To submit to the Cloud we use [`gcloud ai-platform jobs submit training [jobname]`](https://cloud.google.com/sdk/gcloud/reference/ml-engine/jobs/submit/training) and simply specify some additional parameters for AI Platform Training Service:
# - jobname: A unique identifier for the Cloud job. We usually append system time to ensure uniqueness
# - master-image-uri: The uri of the Docker image we pushed in the Google Cloud registry
# - region: Cloud region to train in. See [here](https://cloud.google.com/ml-engine/docs/tensorflow/regions) for supported AI Platform Training Service regions
# 
# 
# The arguments before `-- \` are for AI Platform Training Service.
# The arguments after `-- \` are sent to our `task.py`.

# You can track your job and view logs using [cloud console](https://console.cloud.google.com/mlengine/jobs).

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nPROJECT_ID=$(gcloud config list project --format "value(core.project)")\nBUCKET=$PROJECT_ID\nREGION="us-central1"\n\n# Output directory and jobID\nOUTDIR=gs://${BUCKET}/taxifare/trained_model\nJOBID=taxifare_container_$(date -u +%y%m%d_%H%M%S)\necho ${OUTDIR} ${REGION} ${JOBID}\ngsutil -m rm -rf ${OUTDIR}\n\n# Model and training hyperparameters\nBATCH_SIZE=50\nNUM_EXAMPLES_TO_TRAIN_ON=100\nNUM_EVALS=100\nNBUCKETS=10\nNNSIZE="32 8"\n\n# AI-Platform machines to use for training\nMACHINE_TYPE=n1-standard-4\nSCALE_TIER=CUSTOM\n\n# GCS paths.\nGCS_PROJECT_PATH=gs://$BUCKET/taxifare\nDATA_PATH=$GCS_PROJECT_PATH/data\nTRAIN_DATA_PATH=$DATA_PATH/taxi-train*\nEVAL_DATA_PATH=$DATA_PATH/taxi-valid*\n\nIMAGE_NAME=taxifare_training_container\nIMAGE_URI=gcr.io/$PROJECT_ID/$IMAGE_NAME\n\ngcloud beta ai-platform jobs submit training $JOBID \\\n   --staging-bucket=gs://$BUCKET \\\n   --region=$REGION \\\n   --master-image-uri=$IMAGE_URI \\\n   --master-machine-type=$MACHINE_TYPE \\\n   --scale-tier=$SCALE_TIER \\\n  -- \\\n  --eval_data_path $EVAL_DATA_PATH \\\n  --output_dir $OUTDIR \\\n  --train_data_path $TRAIN_DATA_PATH \\\n  --batch_size $BATCH_SIZE \\\n  --num_examples_to_train_on $NUM_EXAMPLES_TO_TRAIN_ON \\\n  --num_evals $NUM_EVALS \\\n  --nbuckets $NBUCKETS \\\n  --nnsize $NNSIZE ')


# **Remark:** If you prefer submitting your jobs for training on the AI-platform using the command line, we have written the `./taxifare/scripts/submit.sh` for you (that you can also invoke using `make submit` from within `./taxifare`). As the other scripts, it reads it configuration variables from `./taxifare/scripts/env.sh`.

# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
