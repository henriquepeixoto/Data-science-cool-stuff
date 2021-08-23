#!/usr/bin/env python
# coding: utf-8

# In[35]:


from google.cloud import storage
import gcsfs
from PIL import Image
import base64
import io
import cv2
from google.cloud import aiplatform
import googleapiclient.discovery
from google.cloud.aiplatform.gapic.schema import predict
import pandas as pd
import numpy as np


# In[42]:


def predict_image_classification(
    project: str,
    endpoint_id: str,
    filename: str,
    location: str = "us-central1",
    api_endpoint: str = "us-central1-aiplatform.googleapis.com",
):
    
    
    #Getting the image:
    fs = gcsfs.GCSFileSystem(project='inspecao-de-embalagens')

    with fs.open(filename, 'rb') as f:
        jpeg_data = base64.b64encode(f.read())
    #resize image
    buffer = io.BytesIO()
    imgdata = base64.b64decode(jpeg_data)
    img = Image.open(io.BytesIO(imgdata))
    new_img = img.resize((800,800))
    new_img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue())
    
    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)


    instance = predict.instance.ImageClassificationPredictionInstance(content=img_b64.decode()
    ).to_value()
    instances = [instance]

    # See gs://google-cloud-aiplatform/schema/predict/params/image_classification_1.0.0.yaml for the format of the parameters.
    parameters = predict.params.ImageClassificationPredictionParams(confidence_threshold=0.5, max_predictions=1).to_value()
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )

    # See gs://google-cloud-aiplatform/schema/predict/prediction/classification.yaml for the format of the predictions.
    predictions = response.predictions
    for prediction in predictions:
        pred = dict(prediction)
    conf = pred['confidences'][0]
    name = pred['displayNames'][0]
    
    if name == 'ProdutoConforme':
        name = 'Produto conforme'
        return name
    else:
        endpoint = client.endpoint_path(
        project=project, location=location, endpoint='1918164005352898560')
        response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters)
        predictions = response.predictions
        for prediction in predictions:
            pred = dict(prediction)
        conf = pred['confidences'][0]
        name = pred['displayNames'][0]
        
        if name == 'Sujidade':
            name = 'Sujidade na solda'
        elif name == 'Prega':
            name = 'Prega na solda'
        print('Confiança:')
        print(conf)
    return name


# In[43]:


predict_image_classification('414034571314', '7536897121706835968', 'gs://imagens-para-treinamento-do-modelo/2021-08-08heic_Prega na solda_20210720_155524.jpeg')


# In[6]:





# In[7]:





# In[ ]:




