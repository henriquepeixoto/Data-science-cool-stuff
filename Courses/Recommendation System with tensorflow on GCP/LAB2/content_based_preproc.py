#!/usr/bin/env python
# coding: utf-8

# ## Create datasets for the Content-based Filter
# 
# This notebook builds the data we will use for creating our content based model. We'll collect the data via a collection of SQL queries from the publicly avialable Kurier.at dataset in BigQuery.
# Kurier.at is an Austrian newsite. The goal of these labs is to recommend an article for a visitor to the site. In this lab we collect the data for training, in the subsequent notebook we train the recommender model. 
# 
# This notebook illustrates
# * how to pull data from BigQuery table and write to local files
# * how to make reproducible train and test splits 

# In[ ]:


import os
import tensorflow as tf
import numpy as np
from google.cloud import bigquery 

PROJECT = 'cloud-training-demos' # REPLACE WITH YOUR PROJECT ID
BUCKET = 'cloud-training-demos-ml' # REPLACE WITH YOUR BUCKET NAME
REGION = 'us-central1' # REPLACE WITH YOUR BUCKET REGION e.g. us-central1

# do not change these
os.environ['PROJECT'] = PROJECT
os.environ['BUCKET'] = BUCKET
os.environ['REGION'] = REGION
os.environ['TFVERSION'] = '2.1'


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'gcloud  config  set project $PROJECT\ngcloud config set compute/region $REGION')


# We will use this helper funciton to write lists containing article ids, categories, and authors for each article in our database to local file. 

# In[ ]:


def write_list_to_disk(my_list, filename):
  with open(filename, 'w') as f:
    for item in my_list:
        line = "%s\n" % item
        f.write(line)


# ### Pull data from BigQuery
# 
# The cell below creates a local text file containing all the article ids (i.e. 'content ids') in the dataset. 
# 
# Have a look at the original dataset in [BigQuery](https://console.cloud.google.com/bigquery?p=cloud-training-demos&d=GA360_test&t=ga_sessions_sample). Then read through the query below and make sure you understand what it is doing. 

# In[ ]:


sql="""
#standardSQL

SELECT  
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id 
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY
  content_id
  
"""

content_ids_list = bigquery.Client().query(sql).to_dataframe()['content_id'].tolist()
write_list_to_disk(content_ids_list, "content_ids.txt")
print("Some sample content IDs {}".format(content_ids_list[:3]))
print("The total number of articles is {}".format(len(content_ids_list)))


# There should be 15,634 articles in the database.  
# Next, we'll create a local file which contains a list of article categories and a list of article authors.
# 
# Note the change in the index when pulling the article category or author information. Also, we are using the first author of the article to create our author list.  
# Refer back to the original dataset, use the `hits.customDimensions.index` field to verify the correct index.	 

# In[ ]:


sql="""
#standardSQL
SELECT  
  (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category  
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY   
  category
"""
categories_list = bigquery.Client().query(sql).to_dataframe()['category'].tolist()
write_list_to_disk(categories_list, "categories.txt")
print(categories_list)


# The categories are 'News', 'Stars & Kultur', and 'Lifestyle'.  
# When creating the author list, we'll only use the first author information for each article.  

# In[ ]:


sql="""
#standardSQL
SELECT
  REGEXP_EXTRACT((SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)), r"^[^,]+")  AS first_author  
FROM `cloud-training-demos.GA360_test.ga_sessions_sample`,   
  UNNEST(hits) AS hits
WHERE 
  # only include hits on pages
  hits.type = "PAGE"
  AND (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
GROUP BY   
  first_author
"""
authors_list = bigquery.Client().query(sql).to_dataframe()['first_author'].tolist()
write_list_to_disk(authors_list, "authors.txt")
print("Some sample authors {}".format(authors_list[:10]))
print("The total number of authors is {}".format(len(authors_list)))


# There should be 385 authors in the database. 

# ### Create train and test sets.
# 
# In this section, we will create the train/test split of our data for training our model. We use the concatenated values for visitor id and content id to create a farm fingerprint, taking approximately 90% of the data for the training set and 10% for the test set.

# In[ ]:


sql="""
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category, 
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM 
    `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND ABS(MOD(FARM_FINGERPRINT(CONCAT(visitor_id, content_id)), 10)) < 9
"""
training_set_df = bigquery.Client().query(sql).to_dataframe()
training_set_df.to_csv('training_set.csv', header=False, index=False, encoding='utf-8')
training_set_df.head()


# In[ ]:


sql="""
WITH site_history as (
  SELECT
      fullVisitorId as visitor_id,
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) AS content_id,
      (SELECT MAX(IF(index=7, value, NULL)) FROM UNNEST(hits.customDimensions)) AS category, 
      (SELECT MAX(IF(index=6, value, NULL)) FROM UNNEST(hits.customDimensions)) AS title,
      (SELECT MAX(IF(index=2, value, NULL)) FROM UNNEST(hits.customDimensions)) AS author_list,
      SPLIT(RPAD((SELECT MAX(IF(index=4, value, NULL)) FROM UNNEST(hits.customDimensions)), 7), '.') as year_month_array,
      LEAD(hits.customDimensions, 1) OVER (PARTITION BY fullVisitorId ORDER BY hits.time ASC) as nextCustomDimensions
  FROM 
    `cloud-training-demos.GA360_test.ga_sessions_sample`,   
     UNNEST(hits) AS hits
   WHERE 
     # only include hits on pages
      hits.type = "PAGE"
      AND
      fullVisitorId IS NOT NULL
      AND
      hits.time != 0
      AND
      hits.time IS NOT NULL
      AND
      (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(hits.customDimensions)) IS NOT NULL
)
SELECT
  visitor_id,
  content_id,
  category,
  REGEXP_REPLACE(title, r",", "") as title,
  REGEXP_EXTRACT(author_list, r"^[^,]+") as author,
  DATE_DIFF(DATE(CAST(year_month_array[OFFSET(0)] AS INT64), CAST(year_month_array[OFFSET(1)] AS INT64), 1), DATE(1970,1,1), MONTH) as months_since_epoch,
  (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) as next_content_id
FROM
  site_history
WHERE (SELECT MAX(IF(index=10, value, NULL)) FROM UNNEST(nextCustomDimensions)) IS NOT NULL
      AND ABS(MOD(FARM_FINGERPRINT(CONCAT(visitor_id, content_id)), 10)) >= 9
"""
test_set_df = bigquery.Client().query(sql).to_dataframe()
test_set_df.to_csv('test_set.csv', header=False, index=False, encoding='utf-8')
test_set_df.head()


# Let's have a look at the two csv files we just created containing the training and test set. We'll also do a line count of both files to confirm that we have achieved an approximate 90/10 train/test split.  
# In the next notebook, **Content Based Filtering** we will build a model to recommend an article given information about the current article being read, such as the category, title, author, and publish date. 

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'wc -l *_set.csv')


# In[ ]:


get_ipython().system('head *_set.csv')

