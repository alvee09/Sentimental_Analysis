import pandas as pd
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
import re, string, random

tknzr = TweetTokenizer()


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)
        token = re.sub("Covid", "", token)
        token = re.sub("COVID", "", token)
        token = re.sub("lockdown", "", token)
        token = re.sub("nan", "", token)
        token = re.sub("️", "", token)
        token = re.sub("get", "", token)
        token = re.sub("australia", "", token)
        token = re.sub("go", "", token)
        token = re.sub("vaccine", "", token)
        token = re.sub("pandemic", "", token)
        token = re.sub("^\s+", "", token)  # remove the front
        token = re.sub("\s+\Z", "", token)  # remove the back
        token = re.sub("19", "", token)
        token = re.sub("Australia", "", token)
        token = re.sub("AUSTRALIA", "", token)
        token = re.sub("sydney", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(
                token) > 0 and token not in string.punctuation and token.lower() not in stop_words and token != "..." and token != "’" and token != 'covid':
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


df = pd.read_csv('./Data_output/August/mergedfile.csv', index_col=False)
august_tweet = df['Tweet_text']

for tweet in august_tweet:
    custom_tokens = remove_noise(word_tokenize(tweet))
    print(custom_tokens)
    # print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))


# august_tweet_tokens = []
#
# for index, row in august_tweet.iterrows():
#     august_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))
# print(august_tweet_tokens)
# august_cleaned_tokens_list = []
#
# stop_words = stopwords.words('english')
# for tokens in august_tweet_tokens:
#     august_cleaned_tokens_list.append(remove_noise(tokens, stop_words))
#
# august_tokens_for_model = get_tweets_for_model(august_cleaned_tokens_list)
#
# print(august_tokens_for_model)

# custom_tokens = remove_noise(word_tokenize(custom_tweet))

# print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
