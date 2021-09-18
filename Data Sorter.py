#!/usr/bin/env python
# coding: utf-8

# In[202]:


import pandas as pd
import csv
import datetime


# In[203]:


df = pd.read_csv('./Downloads/120321-V8/tweetid_userid_keyword_topics_sentiments_emotions_timestamp (usa sample).csv')
df.shape
df.head()


# In[204]:


df['date_stamp'] = pd.to_datetime(df['date_stamp'])


# In[205]:


formatted_df = df["date_stamp"].dt.strftime("%d/%m/%Y")
df['date_stamp'] = formatted_df
df.head()


# In[206]:


start = datetime.datetime.strptime("01/03/2020", '%d/%m/%Y')
end = datetime.datetime.strptime("31/08/2020", '%d/%m/%Y')


# In[207]:


df['date_stamp']= pd.to_datetime(df['date_stamp'])


# In[208]:


type(df['date_stamp'][1])


# In[209]:


df['date_stamp'][1]


# In[210]:


df.shape


# In[211]:


after = df['date_stamp'] >= start


# In[212]:


df['date_stamp'].value_counts()


# In[213]:


df_new = df[(df['date_stamp'] >= start) & (df['date_stamp'] <= end)]


# In[214]:


df_new.to_csv('./Downloads/out_usa.csv', index=False)
#df[mask].to_csv('./Downloads/out2.csv', index=False)


# In[ ]:


df_new.shape


# In[ ]:


df_new.head(100)


# In[ ]:




