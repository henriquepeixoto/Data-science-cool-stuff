#!/usr/bin/env python
# coding: utf-8

# ## Content Based Filtering by hand
# 
# This lab illustrates how to implement a content based filter using low level Tensorflow operations.  
# The code here follows the technique explained in Module 2 of Recommendation Engines: Content Based Filtering.
# 
# 

# In[ ]:


get_ipython().system('pip install tensorflow==2.1')


# Make sure to restart your kernel to ensure this change has taken place.

# In[ ]:


import numpy as np
import tensorflow as tf

print(tf.__version__)


# To start, we'll create our list of users, movies and features. While the users and movies represent elements in our database, for a content-based filtering method the features of the movies are likely hand-engineered and rely on domain knowledge to provide the best embedding space. Here we use the categories of Action, Sci-Fi, Comedy, Cartoon, and Drama to describe our movies (and thus our users).
# 
# In this example, we will assume our database consists of four users and six movies, listed below.  

# In[ ]:


users = ['Ryan', 'Danielle',  'Vijay', 'Chris']
movies = ['Star Wars', 'The Dark Knight', 'Shrek', 'The Incredibles', 'Bleu', 'Memento']
features = ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']

num_users = len(users)
num_movies = len(movies)
num_feats = len(features)
num_recommendations = 2


# ### Initialize our users, movie ratings and features
# 
# We'll need to enter the user's movie ratings and the k-hot encoded movie features matrix. Each row of the users_movies matrix represents a single user's rating (from 1 to 10) for each movie. A zero indicates that the user has not seen/rated that movie. The movies_feats matrix contains the features for each of the given movies. Each row represents one of the six movies, the columns represent the five categories. A one indicates that a movie fits within a given genre/category. 

# In[ ]:


# each row represents a user's rating for the different movies
users_movies = tf.constant([
                [4,  6,  8,  0, 0, 0],
                [0,  0, 10,  0, 8, 3],
                [0,  6,  0,  0, 3, 7],
                [10, 9,  0,  5, 0, 2]],dtype=tf.float32)

# features of the movies one-hot encoded
# e.g. columns could represent ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']
movies_feats = tf.constant([
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1]],dtype=tf.float32)


# ### Computing the user feature matrix
# 
# We will compute the user feature matrix; that is, a matrix containing each user's embedding in the five-dimensional feature space. 

# In[ ]:


users_feats = tf.matmul(users_movies,movies_feats)
users_feats


# Next we normalize each user feature vector to sum to 1. Normalizing isn't strictly neccesary, but it makes it so that rating magnitudes will be comparable between users.

# In[ ]:


users_feats = users_feats/tf.reduce_sum(users_feats,axis=1,keepdims=True)
users_feats


# #### Ranking feature relevance for each user
# 
# We can use the users_feats computed above to represent the relative importance of each movie category for each user. 

# In[ ]:


top_users_features = tf.nn.top_k(users_feats, num_feats)[1]
top_users_features


# In[ ]:


for i in range(num_users):
    feature_names = [features[int(index)] for index in top_users_features[i]]
    print('{}: {}'.format(users[i],feature_names))


# ### Determining movie recommendations. 
# 
# We'll now use the `users_feats` tensor we computed above to determine the movie ratings and recommendations for each user.
# 
# To compute the projected ratings for each movie, we compute the similarity measure between the user's feature vector and the corresponding movie feature vector.  
# 
# We will use the dot product as our similarity measure. In essence, this is a weighted movie average for each user.

# In[ ]:


users_ratings = tf.matmul(users_feats,tf.transpose(movies_feats))
users_ratings


# The computation above finds the similarity measure between each user and each movie in our database. To focus only on the ratings for new movies, we apply a mask to the all_users_ratings matrix.  
# 
# If a user has already rated a movie, we ignore that rating. This way, we only focus on ratings for previously unseen/unrated movies.

# In[ ]:


users_ratings_new = tf.where(tf.equal(users_movies, tf.zeros_like(users_movies)),
                                  users_ratings,
                                  tf.zeros_like(tf.cast(users_movies, tf.float32)))
users_ratings_new


# Finally let's grab and print out the top 2 rated movies for each user

# In[ ]:


top_movies = tf.nn.top_k(users_ratings_new, num_recommendations)[1]
top_movies


# In[ ]:


for i in range(num_users):
    movie_names = [movies[index] for index in top_movies[i]]
    print('{}: {}'.format(users[i],movie_names))

