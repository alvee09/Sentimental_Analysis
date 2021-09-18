#!/usr/bin/env python
# coding: utf-8

# In[118]:


import pandas as pd
data = pd.read_csv("./Data/tweetid_userid_keyword_topics_sentiments_emotions Part 3.csv")


# In[119]:


#data2 = pd.read_csv("./Data//tweetid_userid_keyword_topics_sentiments_emotions updated.csv")


# In[139]:


data4= pd.read_csv("./Data/2021_march28_march29.csv", names = ["tweet_ID", "Geo"])


# In[154]:


data4.shape


# In[141]:


#test = data4.head()


# In[122]:


#data3 = data2.head(500000)


# In[123]:


#data3.shape


# In[116]:


#data3.loc[data3["tweet_ID"] == 1224743166844320000, "tweet"] = "tweet nai"
#data3.head()


# In[146]:


from twarc import Twarc

consumer_key="ruQ0WMdFl5x5juM3jyTyr1Sow"
consumer_secret="nJNpEQmD6ZuyTM4npQ35FziJKi7ywRwJ4ihAPirwcdJEr44t1l"
access_token="AAAAAAAAAAAAAAAAAAAAANUbTAEAAAAAQqGu0FE29f64VmVwCtsPNbZijbg%3D8xa2lH50Y10Le7UVb2w7N6tuSHvwr2DSOdb1KsizaUwjhoxWYQ"
access_token_secret=""

t = Twarc(consumer_key, consumer_secret, access_token, access_token_secret)
rowcount = 0
for tweet in t.hydrate(data4["tweet_ID"]):    
    #print(tweet)
    data4.loc[data4["tweet_ID"] == tweet["id"], "tweet"] = tweet['full_text']
    data4.loc[data4["tweet_ID"] == tweet["id"], "Country"] = tweet['place']['country']
    #data3.loc[rowcount]['id_from_twitter'] = tweet['id']
    print(tweet['full_text'])
    #print(tweet.keys())
        
    
    rowcount+=1
    
    #print(tweet['id'])
    
    #if tweet['place']:
     #   print(tweet['place']['country'])


# In[147]:


data4.to_csv('./Downloads/with_tweet.csv', index=False)


# In[152]:


data5= data4.loc[data4["Country"] == "Australia"]


# In[153]:


data5


# In[ ]:




