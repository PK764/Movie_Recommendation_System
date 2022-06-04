#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval


# # Reading datasets

# In[10]:


credits_dataframe=pd.read_csv("C:/Users/konda/Downloads/Movie_recommendation_system/tmdb_5000_credits.csv")
movies_dataframe=pd.read_csv("C:/Users/konda/Downloads/Movie_recommendation_system/tmdb_5000_movies.csv")


# In[11]:


movies_dataframe.head(5)


# In[12]:


credits_dataframe.head(6)


# # Merging movies and credits datasets

# In[13]:


credits_dataframe.columns=['id','title','cast','crew']
movies_dataframe=movies_dataframe.merge(credits_dataframe,on='id')


# In[17]:


movies_dataframe.info


# In[53]:


movies_dataframe.dropna(inplace=True)
movies_dataframe["genres"].head(5)


# In[58]:


movies_dataframe["cast"].head(5)


# In[59]:


def get_director(x):
    for i in x:
        if i["job"] == "Director":
            return i["name"]
    return np.nan


# In[60]:


def get_list(x):
    if isinstance(x, list):
        names = [i["name"] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []


# In[62]:


movies_dataframe["director"] = movies_dataframe["crew"].apply(get_director)

features = ["cast", "keywords", "genres"]
for feature in features:
    movies_dataframe[feature] = movies_dataframe[feature].apply(get_list)


# In[66]:


movies_dataframe[['cast', 'director', 'keywords', 'genres']].head()


# In[68]:


def clean_data(row):
    if isinstance(row, list):
        return [str.lower(i.replace(" ", "")) for i in row]
    else:
        if isinstance(row, str):
            return str.lower(row.replace(" ", ""))
        else:
            return ""

features = ['cast', 'keywords', 'director', 'genres']
for feature in features:
    movies_dataframe[feature] = movies_dataframe[feature].apply(clean_data)


# In[70]:


def create_soup(features):
    return ' '.join(features['keywords']) + ' ' + ' '.join(features['cast']) + ' ' + features['director'] + ' ' + ' '.join(features['genres'])


movies_dataframe["soup"] = movies_dataframe.apply(create_soup, axis=1)
print(movies_dataframe["soup"].head())


# In[75]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

count_vectorizer = CountVectorizer(stop_words="english")
count_matrix = count_vectorizer.fit_transform(movies_dataframe["soup"])

print(count_matrix.shape)

cosine_sim2 = cosine_similarity(count_matrix, count_matrix) 
print(cosine_sim2.shape)

movies_dataframe = movies_dataframe.reset_index()
indices = pd.Series(movies_dataframe.index, index=movies_dataframe['original_title'])


# In[78]:


indices = pd.Series(movies_dataframe.index, index=movies_dataframe['original_title']).drop_duplicates()

print(indices.head())


# In[80]:


def get_recommendations(title, cosine_sim=cosine_sim):
    idx = indices[title]
    similarity_scores = list(enumerate(cosine_sim[idx]))
    similarity_scores= sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores= sim_scores[1:11]
    # (a, b) where a is id of movie, b is similarity_scores

    movies_indices = [ind[0] for ind in similarity_scores]
    movies = movies_dataframe["title"].iloc[movies_indices]
    return movies


print("################ Content Based System #############")
print("Recommendations for The Dark Knight Rises")
print(get_recommendations("The Dark Knight Rises", cosine_sim2))
print()
print("Recommendations for Avengers")
print(get_recommendations("The Avengers", cosine_sim2))


# In[ ]:




