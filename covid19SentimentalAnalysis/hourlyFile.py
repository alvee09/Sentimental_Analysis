import nltk
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('punkt')
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from pandas import Series
import re, string, random
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

df = pd.read_csv('Hourly_updates.csv', index_col=False)

df = df[['Sentiment_Label', 'Tweet_text']]

positive_tweets = df.loc[df['Sentiment_Label']== 'positive']
negative_tweets = df.loc[df['Sentiment_Label']== 'negative']


positive_tweets = positive_tweets[['Tweet_text']]
negative_tweets = negative_tweets[['Tweet_text']]

positive_tweet_tokens = []
negative_tweet_tokens = []

for index, row in positive_tweets.iterrows():
    # print(row['Tweet_text'])
    positive_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))

for index, row in negative_tweets.iterrows():
    # print(row['Tweet_text'])
    negative_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))

# print(positive_tweet_tokens[0])
# print(negative_tweets_tokens[0])

# type(positive_tweets_with_label)
# print(positive_tweets_with_label.iloc[0,1])

# positive_tweets = positive_tweets_with_label.Tweet_text
# negative_tweets = negative_tweets_with_label.Tweet_text
# print(type(positive_tweets))
# print(type(positive_tweets))
# print(type(positive_tweets))

#
# p_tweets = ""
# i = 0;
# for index, value in positive_tweets.items():
#     # print(index, " ", value)
#     p_tweets += value
#     ++i

# print(p_tweets[0,:])

# Convert to string
# positive_tweets = Series.to_string(positive_tweets)

# print(positive_tweets)
# negative_tweets = Series.to_string(negative_tweets)


# print(positive_tweets)
# print(type(positive_tweets))


# Tokenizer
# positive_tweet_tokens01 = tknzr.tokenize(positive_tweets[:])
# print(type(positive_tweet_tokens01))
# positive_tweet_tokens02 = word_tokenize(positive_tweets)





