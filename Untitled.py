#!/usr/bin/env python
# coding: utf-8

# ## The Olympic Games

# ![image.png](attachment:image.png)

# The Olympic Games are considered the world's foremost sports competition with more than 200 nations participating. The Olympic Games are normally held every four years, alternating between the Summer and Winter Olympics every two years in the four-year period.
# They have sent 2 datasets: 
# Athletes, this datasset is about the information of the athletes that took part in the final of all the olympics games since 1896 including the columns of: athlete_url, the url of the athlete; athlete_full_name, the name of the finalist; first_game, the first olympic game the athlete has ever been on; athlete_year_birth, the athlete´s year of birth; athlete_medals, the athlete´s medals; games_participations, how many times have the athlete participate; and bio, the information of the athlete.
# Medals: dataset includes a row for every Olympic Medal (Athlete or Team) that has won a medal including the columns of: discipline_title, the name of the discipline; slug_game, the id of the olympic game; event_title, the name of the event; event_gender, the gender of the players of the event; medal_type, The kind of medal the team pr athlete has earn; participant_type, if the medal is from a team or an athlete; participant_title, where is the player from; athlete_url, the url of the player; athlete_full_name, the name of the athletes; country_name, the name of the athletes´ country; country_code, the universal code of the country; country_3_letter_code, the universal 3 digits code of the country.
#     

# This Olympic Games organization is requiering me as a data science for fixing their information on the olympic games thrue the years and figuring out what kind of settings can it be done.

# ## Table Processing

# In[42]:


import pandas as pd
import sklearn.model_selection
import sklearn.tree
import sklearn.metrics
import matplotlib.pyplot as plt


# ## Load and Show the Datasets

# In[22]:


athletes= pd.read_csv("olympic_athletes.csv")
athletes.head()


# In[44]:


medals= pd.read_csv("olympic_medals.csv")
medals.head()


# 

# ## Data explorations

# In[40]:


athletes.describe()


# In[43]:


athletes.boxplot()


# In[48]:


athletes.hist()


# In[26]:


athletes.isnull().sum()


# In[45]:


medals.describe()


# In[27]:


medals.isnull().sum()


# ## Data Preprocessing

# As it is notable there is a problem of too much data related to the country of the athlete, the columns participant_title, country_name, country_code and country_3_letter_code; for this reason, there will be reduce to only one column call athlete_country that will be filled with the country of the athlete that earned the medal with the information of country_name because it is the fullest column.

# In[62]:


medals['athlete_country'] = medals['country_name']


# In[63]:


medals=medals.drop(columns=['participant_title', 'country_name',
             'country_code', 'country_3_letter_code'])


# In[64]:


medals.isnull().sum()


# In this part we will merge the two columns with the athlete full name

# In[133]:


df= medals.merge(athletes, how='left', on='athlete_full_name')
df.head()


# There is a lot of inconsistancy in the column of bio will afect on the analysis because most of the data from there is missing and almost never filled when they insert the new data.

# In[134]:


df= df.drop(columns=['bio'])


# To fill the values that are null in the columns of games_participations and athlete_medals, the null values will be 0.

# In[135]:


df.loc[df['games_participations'].isnull(),'games_participations']=0


# In[136]:


df.loc[df['athlete_medals'].isnull(),'athlete_medals']=0


# The null values on the athlete_url, first_game and athlete_full_name columns will be change to NA

# In[137]:


df['athlete_url']=(df['athlete_url_y'])
df.loc[df['athlete_url'].isnull(),'athlete_url']=df['athlete_url_x']
df.loc[df['athlete_url'].isnull(),'athlete_url']= 'NA'
df= df.drop(columns=['athlete_url_x', 'athlete_url_y'])


# In[138]:


df.loc[df['athlete_full_name'].isnull(),'athlete_full_name']= 'NA'


# In[139]:


df.loc[df['first_game'].isnull(),'first_game']= 'NA'


# The null values on the athlete_year_birth column will be deleted because it can affect the analysis

# In[140]:


df = df.dropna(subset=['athlete_year_birth'])


# In[141]:


df.isnull().sum()


# Other problem of the dataset is that the column athlete_medals has the simmbol \n, for that it will be deleted from that column.

# In[142]:


df['athlete_medals']=df['athlete_medals'].replace(to_replace=r'\n', value='', regex=True)


# ## Business Questions

# How many medals of each has every country?

# It is important to know how many medals have every country for classifying the top. In the result we first filter by country and then by kind of medal.

# In[168]:


df.groupby(['athlete_country','medal_type']).size()


# How many medals of each has every country in every discipline?

# It is important to know how many medals have every country by discipline for classifying the top. In the result we first filter by country and then by kind of medal.

# In[170]:


df.groupby(['athlete_country','discipline_title','medal_type']).size()


# Which country has the most medals?

# This is the most important question because it can be known which country has invest more in sports and in this case as the output of the code says is the United States of America.

# In[160]:


df.groupby(['athlete_country']).size().idxmax()


# How many medals of each has every country in game?

# It is important to know how many medals have every country by discipline for classifying the top. In the result we first filter by game, then country and by kind of medal.

# In[162]:


df.groupby(['slug_game','athlete_country','medal_type']).size()


# Which country has the most gold medals?

# This is an important question because it can be known which country has invest the most in sports and in this case as the output of the code says is the United States of America.

# In[165]:


df[df['medal_type']=='GOLD'].groupby(['athlete_country']).size().idxmax()


# Which country has the most silver medals?

# This is an important question because it can be known which country has invest the most in sports and in this case as the output of the code says is the United States of America.

# In[166]:


df[df['medal_type']=='SILVER'].groupby(['athlete_country']).size().idxmax()


# Which country has the most bronze medals?

# This will show which country has earned the most of the third places on the olympics game and as it says it is the United States of America. 

# In[167]:


df[df['medal_type']=='BRONZE'].groupby(['athlete_country']).size().idxmax()


# ## Conclusions

# The first 2 datasets given where really empty and they had to be fixed, the data was ambiguous and some data had to be changed as well as well as the null values, but on the rigth sigth they were easy to merge to create a bigger dataset for the analysis.
# This data was easy to analyze, the data is really simple and easy to clean, also.
# It is recommended for the client to upload the data without errors and no empty values.

# In[ ]:




