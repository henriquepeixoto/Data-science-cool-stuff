#!/usr/bin/env python
# coding: utf-8

# # Performing the Hyperparameter tuning
# 
# **Learning Objectives**
# 1. Learn how to use `cloudml-hypertune` to report the results for Cloud hyperparameter tuning trial runs
# 2. Learn how to configure the `.yaml` file for submitting a Cloud hyperparameter tuning job
# 3. Submit a hyperparameter tuning job to Cloud AI Platform
# 
# ## Introduction
# 
# Let's see if we can improve upon that by tuning our hyperparameters.
# 
# Hyperparameters are parameters that are set *prior* to training a model, as opposed to parameters which are learned *during* training. 
# 
# These include learning rate and batch size, but also model design parameters such as type of activation function and number of hidden units.
# 
# Here are the four most common ways to finding the ideal hyperparameters:
# 1. Manual
# 2. Grid Search
# 3. Random Search
# 4. Bayesian Optimzation
# 
# **1. Manual**
# 
# Traditionally, hyperparameter tuning is a manual trial and error process. A data scientist has some intution about suitable hyperparameters which they use as a starting point, then they observe the result and use that information to try a new set of hyperparameters to try to beat the existing performance. 
# 
# Pros
# - Educational, builds up your intuition as a data scientist
# - Inexpensive because only one trial is conducted at a time
# 
# Cons
# - Requires alot of time and patience
# 
# **2. Grid Search**
# 
# On the other extreme we can use grid search. Define a discrete set of values to try for each hyperparameter then try every possible combination. 
# 
# Pros
# - Can run hundreds of trials in parallel using the cloud
# - Gauranteed to find the best solution within the search space
# 
# Cons
# - Expensive
# 
# **3. Random Search**
# 
# Alternatively define a range for each hyperparamter (e.g. 0-256) and sample uniformly at random from that range. 
# 
# Pros
# - Can run hundreds of trials in parallel using the cloud
# - Requires less trials than Grid Search to find a good solution
# 
# Cons
# - Expensive (but less so than Grid Search)
# 
# **4. Bayesian Optimization**
# 
# Unlike Grid Search and Random Search, Bayesian Optimization takes into account information from  past trials to select parameters for future trials. The details of how this is done is beyond the scope of this notebook, but if you're interested you can read how it works here [here](https://cloud.google.com/blog/products/gcp/hyperparameter-tuning-cloud-machine-learning-engine-using-bayesian-optimization). 
# 
# Pros
# - Picks values intelligenty based on results from past trials
# - Less expensive because requires fewer trials to get a good result
# 
# Cons
# - Requires sequential trials for best results, takes longer
# 
# **AI Platform HyperTune**
# 
# AI Platform HyperTune, powered by [Google Vizier](https://ai.google/research/pubs/pub46180), uses Bayesian Optimization by default, but [also supports](https://cloud.google.com/ml-engine/docs/tensorflow/hyperparameter-tuning-overview#search_algorithms) Grid Search and Random Search. 
# 
# 
# When tuning just a few hyperparameters (say less than 4), Grid Search and Random Search work well, but when tunining several hyperparameters and the search space is large Bayesian Optimization is best.

# In[ ]:


# Use the chown command to change the ownership of the repository
get_ipython().system('sudo chown -R jupyter:jupyter /home/jupyter/training-data-analyst')


# In[ ]:


# Installing the latest version of the package
get_ipython().system('pip install --user google-cloud-bigquery==1.25.0')


# **Note**: Restart your kernel to use updated packages.

# Kindly ignore the deprecation warnings and incompatibility errors related to google-cloud-storage.

# In[ ]:


# Importing the necessary module
import os

from google.cloud import bigquery


# In[ ]:


# Change with your own bucket and project below:
BUCKET =  "<BUCKET>"
PROJECT = "<PROJECT>"
REGION = "<YOUR REGION>"

OUTDIR = "gs://{bucket}/taxifare/data".format(bucket=BUCKET)

os.environ['BUCKET'] = BUCKET
os.environ['OUTDIR'] = OUTDIR
os.environ['PROJECT'] = PROJECT
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = "2.3"


# In[ ]:


get_ipython().run_cell_magic('bash', '', '# Setting up cloud SDK properties\ngcloud config set project $PROJECT\ngcloud config set compute/region $REGION')


# ## Make code compatible with AI Platform Training Service
# In order to make our code compatible with AI Platform Training Service we need to make the following changes:
# 
# 1. Upload data to Google Cloud Storage 
# 2. Move code into a trainer Python package
# 4. Submit training job with `gcloud` to train on AI Platform

# ## Upload data to Google Cloud Storage (GCS)
# 
# Cloud services don't have access to our local files, so we need to upload them to a location the Cloud servers can read from. In this case we'll use GCS.

# ## Create BigQuery tables

# If you haven not already created a BigQuery dataset for our data, run the following cell:

# In[ ]:


bq = bigquery.Client(project = PROJECT)
dataset = bigquery.Dataset(bq.dataset("taxifare"))

# Creating a dataset
try:
    bq.create_dataset(dataset)
    print("Dataset created")
except:
    print("Dataset already exists")


# Let's create a table with 1 million examples.
# 
# Note that the order of columns is exactly what was in our CSV files.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nCREATE OR REPLACE TABLE taxifare.feateng_training_data AS\n\nSELECT\n    (tolls_amount + fare_amount) AS fare_amount,\n    pickup_datetime,\n    pickup_longitude AS pickuplon,\n    pickup_latitude AS pickuplat,\n    dropoff_longitude AS dropofflon,\n    dropoff_latitude AS dropofflat,\n    passenger_count*1.0 AS passengers,\n    'unused' AS key\nFROM `nyc-tlc.yellow.trips`\nWHERE ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 1000)) = 1\nAND\n    trip_distance > 0\n    AND fare_amount >= 2.5\n    AND pickup_longitude > -78\n    AND pickup_longitude < -70\n    AND dropoff_longitude > -78\n    AND dropoff_longitude < -70\n    AND pickup_latitude > 37\n    AND pickup_latitude < 45\n    AND dropoff_latitude > 37\n    AND dropoff_latitude < 45\n    AND passenger_count > 0")


# Make the validation dataset be 1/10 the size of the training dataset.

# In[ ]:


get_ipython().run_cell_magic('bigquery', '', "\nCREATE OR REPLACE TABLE taxifare.feateng_valid_data AS\n\nSELECT\n    (tolls_amount + fare_amount) AS fare_amount,\n    pickup_datetime,\n    pickup_longitude AS pickuplon,\n    pickup_latitude AS pickuplat,\n    dropoff_longitude AS dropofflon,\n    dropoff_latitude AS dropofflat,\n    passenger_count*1.0 AS passengers,\n    'unused' AS key\nFROM `nyc-tlc.yellow.trips`\nWHERE ABS(MOD(FARM_FINGERPRINT(CAST(pickup_datetime AS STRING)), 10000)) = 2\nAND\n    trip_distance > 0\n    AND fare_amount >= 2.5\n    AND pickup_longitude > -78\n    AND pickup_longitude < -70\n    AND dropoff_longitude > -78\n    AND dropoff_longitude < -70\n    AND pickup_latitude > 37\n    AND pickup_latitude < 45\n    AND dropoff_latitude > 37\n    AND dropoff_latitude < 45\n    AND passenger_count > 0")


# ## Export the tables as CSV files

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\necho "Deleting current contents of $OUTDIR"\ngsutil -m -q rm -rf $OUTDIR\n\necho "Extracting training data to $OUTDIR"\nbq --location=US extract \\\n   --destination_format CSV  \\\n   --field_delimiter "," --noprint_header \\\n   taxifare.feateng_training_data \\\n   $OUTDIR/taxi-train-*.csv\n\necho "Extracting validation data to $OUTDIR"\nbq --location=US extract \\\n   --destination_format CSV  \\\n   --field_delimiter "," --noprint_header \\\n   taxifare.feateng_valid_data \\\n   $OUTDIR/taxi-valid-*.csv\n\n# List the files of the bucket\ngsutil ls -l $OUTDIR')


# In[ ]:


# Here, it shows the short header for each object
get_ipython().system('gsutil cat gs://$BUCKET/taxifare/data/taxi-train-000000000000.csv | head -2')


# If all ran smoothly, you should be able to list the data bucket by running the following command:

# In[ ]:


# List the files of the bucket
get_ipython().system('gsutil ls gs://$BUCKET/taxifare/data')


# ## Move code into python package
# 
# Here, we moved our code into a python package for training on Cloud AI Platform. Let's just check that the files are there. You should see the following files in the `taxifare/trainer` directory:
#  - `__init__.py`
#  - `model.py`
#  - `task.py`

# In[ ]:


# It will list all the files in the mentioned directory with a long listing format
get_ipython().system('ls -la taxifare/trainer')


# To use hyperparameter tuning in your training job you must perform the following steps:
# 
#  1. Specify the hyperparameter tuning configuration for your training job by including a HyperparameterSpec in your TrainingInput object.
# 
#  2. Include the following code in your training application:
# 
#   - Parse the command-line arguments representing the hyperparameters you want to tune, and use the values to set the hyperparameters for your training trial.
# Add your hyperparameter metric to the summary for your graph.
# 
#   - To submit a hyperparameter tuning job, we must modify `model.py` and `task.py` to expose any variables we want to tune as command line arguments.

# ### Modify model.py

# In[ ]:


get_ipython().run_cell_magic('writefile', './taxifare/trainer/model.py', '\n# Importing the necessary modules\nimport datetime\nimport hypertune\nimport logging\nimport os\nimport shutil\n\nimport numpy as np\nimport tensorflow as tf\n\nfrom tensorflow.keras import activations\nfrom tensorflow.keras import callbacks\nfrom tensorflow.keras import layers\nfrom tensorflow.keras import models\n\nfrom tensorflow import feature_column as fc\n\nlogging.info(tf.version.VERSION)\n\n\nCSV_COLUMNS = [\n        \'fare_amount\',\n        \'pickup_datetime\',\n        \'pickup_longitude\',\n        \'pickup_latitude\',\n        \'dropoff_longitude\',\n        \'dropoff_latitude\',\n        \'passenger_count\',\n        \'key\',\n]\nLABEL_COLUMN = \'fare_amount\'\nDEFAULTS = [[0.0], [\'na\'], [0.0], [0.0], [0.0], [0.0], [0.0], [\'na\']]\nDAYS = [\'Sun\', \'Mon\', \'Tue\', \'Wed\', \'Thu\', \'Fri\', \'Sat\']\n\n\n# Splits features and labels from feature dictionary\ndef features_and_labels(row_data):\n    for unwanted_col in [\'key\']:\n        row_data.pop(unwanted_col)\n    label = row_data.pop(LABEL_COLUMN)\n    return row_data, label\n\n\n# Loads dataset using the tf.data API from CSV files\ndef load_dataset(pattern, batch_size, num_repeat):\n    dataset = tf.data.experimental.make_csv_dataset(\n        file_pattern=pattern,\n        batch_size=batch_size,\n        column_names=CSV_COLUMNS,\n        column_defaults=DEFAULTS,\n        num_epochs=num_repeat,\n    )\n    return dataset.map(features_and_labels)\n\n\n# Prefetch overlaps the preprocessing and model execution of a training step\ndef create_train_dataset(pattern, batch_size):\n    dataset = load_dataset(pattern, batch_size, num_repeat=None)\n    return dataset.prefetch(1)\n\n\ndef create_eval_dataset(pattern, batch_size):\n    dataset = load_dataset(pattern, batch_size, num_repeat=1)\n    return dataset.prefetch(1)\n\n\n# Parse a string and return a datetime.datetime\ndef parse_datetime(s):\n    if type(s) is not str:\n        s = s.numpy().decode(\'utf-8\')\n    return datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S %Z")\n\n\n# Here, tf.sqrt Computes element-wise square root of the input tensor\ndef euclidean(params):\n    lon1, lat1, lon2, lat2 = params\n    londiff = lon2 - lon1\n    latdiff = lat2 - lat1\n    return tf.sqrt(londiff*londiff + latdiff*latdiff)\n\n\n# Timestamp.weekday() function return the day of the week represented by the date in the given Timestamp object\ndef get_dayofweek(s):\n    ts = parse_datetime(s)\n    return DAYS[ts.weekday()]\n\n\n# It wraps a python function into a TensorFlow op that executes it eagerly\n@tf.function\ndef dayofweek(ts_in):\n    return tf.map_fn(\n        lambda s: tf.py_function(get_dayofweek, inp=[s], Tout=tf.string),\n        ts_in\n    )\n\ndef transform(inputs, NUMERIC_COLS, STRING_COLS, nbuckets):\n    # Pass-through columns\n    transformed = inputs.copy()\n    del transformed[\'pickup_datetime\']\n\n    feature_columns = {\n        colname: fc.numeric_column(colname)\n        for colname in NUMERIC_COLS\n    }\n\n    # Scaling longitude from range [-70, -78] to [0, 1]\n    for lon_col in [\'pickup_longitude\', \'dropoff_longitude\']:\n        transformed[lon_col] = layers.Lambda(\n            lambda x: (x + 78)/8.0,\n            name=\'scale_{}\'.format(lon_col)\n        )(inputs[lon_col])\n\n    # Scaling latitude from range [37, 45] to [0, 1]\n    for lat_col in [\'pickup_latitude\', \'dropoff_latitude\']:\n        transformed[lat_col] = layers.Lambda(\n            lambda x: (x - 37)/8.0,\n            name=\'scale_{}\'.format(lat_col)\n        )(inputs[lat_col])\n\n    # Adding Euclidean dist (no need to be accurate: NN will calibrate it)\n    transformed[\'euclidean\'] = layers.Lambda(euclidean, name=\'euclidean\')([\n        inputs[\'pickup_longitude\'],\n        inputs[\'pickup_latitude\'],\n        inputs[\'dropoff_longitude\'],\n        inputs[\'dropoff_latitude\']\n    ])\n    feature_columns[\'euclidean\'] = fc.numeric_column(\'euclidean\')\n\n    # hour of day from timestamp of form \'2010-02-08 09:17:00+00:00\'\n    transformed[\'hourofday\'] = layers.Lambda(\n        lambda x: tf.strings.to_number(\n            tf.strings.substr(x, 11, 2), out_type=tf.dtypes.int32),\n        name=\'hourofday\'\n    )(inputs[\'pickup_datetime\'])\n    feature_columns[\'hourofday\'] = fc.indicator_column(\n        fc.categorical_column_with_identity(\n            \'hourofday\', num_buckets=24))\n\n    latbuckets = np.linspace(0, 1, nbuckets).tolist()\n    lonbuckets = np.linspace(0, 1, nbuckets).tolist()\n    b_plat = fc.bucketized_column(\n        feature_columns[\'pickup_latitude\'], latbuckets)\n    b_dlat = fc.bucketized_column(\n            feature_columns[\'dropoff_latitude\'], latbuckets)\n    b_plon = fc.bucketized_column(\n            feature_columns[\'pickup_longitude\'], lonbuckets)\n    b_dlon = fc.bucketized_column(\n            feature_columns[\'dropoff_longitude\'], lonbuckets)\n    ploc = fc.crossed_column(\n            [b_plat, b_plon], nbuckets * nbuckets)\n    dloc = fc.crossed_column(\n            [b_dlat, b_dlon], nbuckets * nbuckets)\n    pd_pair = fc.crossed_column([ploc, dloc], nbuckets ** 4)\n    feature_columns[\'pickup_and_dropoff\'] = fc.embedding_column(\n            pd_pair, 100)\n\n    return transformed, feature_columns\n\n\n# Here, tf.sqrt Computes element-wise square root of the input tensor\ndef rmse(y_true, y_pred):\n    return tf.sqrt(tf.reduce_mean(tf.square(y_pred - y_true)))\n\n\ndef build_dnn_model(nbuckets, nnsize, lr):\n    # input layer is all float except for pickup_datetime which is a string\n    STRING_COLS = [\'pickup_datetime\']\n    NUMERIC_COLS = (\n            set(CSV_COLUMNS) - set([LABEL_COLUMN, \'key\']) - set(STRING_COLS)\n    )\n    inputs = {\n        colname: layers.Input(name=colname, shape=(), dtype=\'float32\')\n        for colname in NUMERIC_COLS\n    }\n    inputs.update({\n        colname: layers.Input(name=colname, shape=(), dtype=\'string\')\n        for colname in STRING_COLS\n    })\n\n    # transforms\n    transformed, feature_columns = transform(\n        inputs, NUMERIC_COLS, STRING_COLS, nbuckets=nbuckets)\n    dnn_inputs = layers.DenseFeatures(feature_columns.values())(transformed)\n\n    x = dnn_inputs\n    for layer, nodes in enumerate(nnsize):\n        x = layers.Dense(nodes, activation=\'relu\', name=\'h{}\'.format(layer))(x)\n    output = layers.Dense(1, name=\'fare\')(x)\n    \n    model = models.Model(inputs, output)\n    lr_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)\n    model.compile(optimizer=lr_optimizer, loss=\'mse\', metrics=[rmse, \'mse\'])\n    \n    return model\n\n\n# Define train and evaluate method to evaluate performance of the model\ndef train_and_evaluate(hparams):\n    batch_size = hparams[\'batch_size\']\n    eval_data_path = hparams[\'eval_data_path\']\n    nnsize = hparams[\'nnsize\']\n    nbuckets = hparams[\'nbuckets\']\n    lr = hparams[\'lr\']\n    num_evals = hparams[\'num_evals\']\n    num_examples_to_train_on = hparams[\'num_examples_to_train_on\']\n    output_dir = hparams[\'output_dir\']\n    train_data_path = hparams[\'train_data_path\']\n\n    if tf.io.gfile.exists(output_dir):\n        tf.io.gfile.rmtree(output_dir)\n    \n    timestamp = datetime.datetime.now().strftime(\'%Y%m%d%H%M%S\')\n    savedmodel_dir = os.path.join(output_dir, \'savedmodel\')\n    model_export_path = os.path.join(savedmodel_dir, timestamp)\n    checkpoint_path = os.path.join(output_dir, \'checkpoints\')\n    tensorboard_path = os.path.join(output_dir, \'tensorboard\')    \n\n    dnn_model = build_dnn_model(nbuckets, nnsize, lr)\n    logging.info(dnn_model.summary())\n\n    trainds = create_train_dataset(train_data_path, batch_size)\n    evalds = create_eval_dataset(eval_data_path, batch_size)\n\n    steps_per_epoch = num_examples_to_train_on // (batch_size * num_evals)\n\n    checkpoint_cb = callbacks.ModelCheckpoint(checkpoint_path,\n                                              save_weights_only=True,\n                                              verbose=1)\n\n    tensorboard_cb = callbacks.TensorBoard(tensorboard_path,\n                                           histogram_freq=1)\n\n    history = dnn_model.fit(\n        trainds,\n        validation_data=evalds,\n        epochs=num_evals,\n        steps_per_epoch=max(1, steps_per_epoch),\n        verbose=2,  # 0=silent, 1=progress bar, 2=one line per epoch\n        callbacks=[checkpoint_cb, tensorboard_cb]\n    )\n\n    # Exporting the model with default serving function.\n    tf.saved_model.save(dnn_model, model_export_path)\n    \n    # TODO 1\n    hp_metric = history.history[\'val_rmse\'][num_evals-1]\n    \n    # TODO 1\n    hpt = hypertune.HyperTune()\n    hpt.report_hyperparameter_tuning_metric(\n        hyperparameter_metric_tag=\'rmse\',\n        metric_value=hp_metric,\n        global_step=num_evals\n    )\n\n    return history')


# ### Modify task.py

# In[ ]:


get_ipython().run_cell_magic('writefile', 'taxifare/trainer/task.py', '# Importing the necessary module\nimport argparse\nimport json\nimport os\n\nfrom trainer import model\n\n\nif __name__ == \'__main__\':\n    parser = argparse.ArgumentParser()\n    parser.add_argument(\n        "--batch_size",\n        help = "Batch size for training steps",\n        type = int,\n        default = 32\n    )\n    parser.add_argument(\n        "--eval_data_path",\n        help = "GCS location pattern of eval files",\n        required = True\n    )\n    parser.add_argument(\n        "--nnsize",\n        help = "Hidden layer sizes (provide space-separated sizes)",\n        nargs = "+",\n        type = int,\n        default=[32, 8]\n    )\n    parser.add_argument(\n        "--nbuckets",\n        help = "Number of buckets to divide lat and lon with",\n        type = int,\n        default = 10\n    )\n    parser.add_argument(\n        "--lr",\n        help = "learning rate for optimizer",\n        type = float,\n        default = 0.001\n    )\n    parser.add_argument(\n        "--num_evals",\n        help = "Number of times to evaluate model on eval data training.",\n        type = int,\n        default = 5\n    )\n    parser.add_argument(\n        "--num_examples_to_train_on",\n        help = "Number of examples to train on.",\n        type = int,\n        default = 100\n    )\n    parser.add_argument(\n    "--output_dir",\n        help = "GCS location to write checkpoints and export models",\n        required = True\n    )\n    parser.add_argument(\n        "--train_data_path",\n        help = "GCS location pattern of train files containing eval URLs",\n        required = True\n    )\n    parser.add_argument(\n        "--job-dir",\n        help = "this model ignores this field, but it is required by gcloud",\n        default = "junk"\n    )\n\n    args, _  = parser.parse_known_args()\n        \n    hparams = args.__dict__\n    hparams["output_dir"] = os.path.join(\n        hparams["output_dir"],\n        json.loads(\n            os.environ.get("TF_CONFIG", "{}")\n        ).get("task", {}).get("trial", "")\n    )\n    print("output_dir", hparams["output_dir"])\n    model.train_and_evaluate(hparams)')


# ### Create config.yaml file
# 
# Specify the hyperparameter tuning configuration for your training job
# Create a HyperparameterSpec object to hold the hyperparameter tuning configuration for your training job, and add the HyperparameterSpec as the hyperparameters object in your TrainingInput object.
# 
# In your HyperparameterSpec, set the hyperparameterMetricTag to a value representing your chosen metric. If you don't specify a hyperparameterMetricTag, AI Platform Training looks for a metric with the name training/hptuning/metric. The following example shows how to create a configuration for a metric named metric1:

# In[ ]:


get_ipython().run_cell_magic('writefile', 'hptuning_config.yaml', '# Setting parameters for hptuning_config.yaml\ntrainingInput:\n  scaleTier: BASIC\n  hyperparameters:\n    goal: MINIMIZE\n    maxTrials: 10 # TODO 2\n    maxParallelTrials: 2 # TODO 2\n    hyperparameterMetricTag: rmse # TODO 2\n    enableTrialEarlyStopping: True\n    params:\n    - parameterName: lr\n      # TODO 2\n      type: DOUBLE\n      minValue: 0.0001\n      maxValue: 0.1\n      scaleType: UNIT_LOG_SCALE\n    - parameterName: nbuckets\n      # TODO 2\n      type: INTEGER\n      minValue: 10\n      maxValue: 25\n      scaleType: UNIT_LINEAR_SCALE\n    - parameterName: batch_size\n      # TODO 2\n      type: DISCRETE\n      discreteValues:\n      - 15\n      - 30\n      - 50\n    ')


# #### Report your hyperparameter metric to AI Platform Training
# 
# The way to report your hyperparameter metric to the AI Platform Training service depends on whether you are using TensorFlow for training or not. It also depends on whether you are using a runtime version or a custom container for training.
# 
# We recommend that your training code reports your hyperparameter metric to AI Platform Training frequently in order to take advantage of early stopping.
# 
# TensorFlow with a runtime version
# If you use an AI Platform Training runtime version and train with TensorFlow, then you can report your hyperparameter metric to AI Platform Training by writing the metric to a TensorFlow summary. Use one of the following functions.
# 
# You may need to install `cloudml-hypertune` on your machine to run this code locally.

# In[ ]:


# Installing the latest version of the package
get_ipython().system('pip install cloudml-hypertune')


# Kindly ignore, if you get the version warnings related to pip install command.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\n# Testing our training code locally\nEVAL_DATA_PATH=./taxifare/tests/data/taxi-valid*\nTRAIN_DATA_PATH=./taxifare/tests/data/taxi-train*\nOUTPUT_DIR=./taxifare-model\n\nrm -rf ${OUTDIR}\nexport PYTHONPATH=${PYTHONPATH}:${PWD}/taxifare\n    \npython3 -m trainer.task \\\n--eval_data_path $EVAL_DATA_PATH \\\n--output_dir $OUTPUT_DIR \\\n--train_data_path $TRAIN_DATA_PATH \\\n--batch_size 5 \\\n--num_examples_to_train_on 100 \\\n--num_evals 1 \\\n--nbuckets 10 \\\n--lr 0.001 \\\n--nnsize 32 8')


# In[ ]:


ls taxifare-model/tensorboard


# The below hyperparameter training job step will take **upto 45 minutes** to complete.

# In[ ]:


get_ipython().run_cell_magic('bash', '', '\nPROJECT_ID=$(gcloud config list project --format "value(core.project)")\nBUCKET=$PROJECT_ID\nREGION="us-central1"\nTFVERSION="2.1"\n\n# Output directory and jobID\nOUTDIR=gs://${BUCKET}/taxifare/trained_model_$(date -u +%y%m%d_%H%M%S)\nJOBID=taxifare_$(date -u +%y%m%d_%H%M%S)\necho ${OUTDIR} ${REGION} ${JOBID}\ngsutil -m rm -rf ${OUTDIR}\n\n# Model and training hyperparameters\nBATCH_SIZE=15\nNUM_EXAMPLES_TO_TRAIN_ON=100\nNUM_EVALS=10\nNBUCKETS=10\nLR=0.001\nNNSIZE="32 8"\n\n# GCS paths\nGCS_PROJECT_PATH=gs://$BUCKET/taxifare\nDATA_PATH=$GCS_PROJECT_PATH/data\nTRAIN_DATA_PATH=$DATA_PATH/taxi-train*\nEVAL_DATA_PATH=$DATA_PATH/taxi-valid*\n\n# TODO 3\ngcloud ai-platform jobs submit training $JOBID \\\n    --module-name=trainer.task \\\n    --package-path=taxifare/trainer \\\n    --staging-bucket=gs://${BUCKET} \\\n    --config=hptuning_config.yaml \\\n    --python-version=3.7 \\\n    --runtime-version=${TFVERSION} \\\n    --region=${REGION} \\\n    -- \\\n    --eval_data_path $EVAL_DATA_PATH \\\n    --output_dir $OUTDIR \\\n    --train_data_path $TRAIN_DATA_PATH \\\n    --batch_size $BATCH_SIZE \\\n    --num_examples_to_train_on $NUM_EXAMPLES_TO_TRAIN_ON \\\n    --num_evals $NUM_EVALS \\\n    --nbuckets $NBUCKETS \\\n    --lr $LR \\\n    --nnsize $NNSIZE ')


# Copyright 2020 Google Inc. Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License
