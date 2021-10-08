from numpy import negative, positive
import pandas as pd

data = pd.read_csv('Hourly_updates.csv')
# print(data.columns)
sentimentData = data[['Sentiment_Label', 'Tweet_text']]

positive_tweets = sentimentData.loc[sentimentData['Sentiment_Label']== 'positive']
negative_tweets = sentimentData.loc[sentimentData['Sentiment_Label']== 'negative']

positive_tweets = positive_tweets[['Tweet_text']]
print(type(positive_tweets.head(1)))

print(positive_tweets)
text = data[['Tweet_text']]

# print(positiveSentiment.head())
# print(negativeSentiment.head())
# print(text.head())