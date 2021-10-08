# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import nltk
# nltk.download('twitter_samples')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
import pandas as pd
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from pandas import Series
import re, string, random
from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []

    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == "__main__":
    # --------------------------------------------
    df = pd.read_csv('mergedfile.csv', index_col=False)

    df = df[['Sentiment_Label', 'Tweet_text']]

    positive_tweets = df.loc[df['Sentiment_Label'] == 'positive']
    negative_tweets = df.loc[df['Sentiment_Label'] == 'negative']
    neutral_tweets = df.loc[df['Sentiment_Label'] == 'neutral']

    positive_tweets = positive_tweets[['Tweet_text']]
    negative_tweets = negative_tweets[['Tweet_text']]
    neutral_tweets = neutral_tweets[['Tweet_text']]

    positive_tweet_tokens = []
    negative_tweet_tokens = []
    neutral_tweet_tokens = []

    for index, row in positive_tweets.iterrows():
        # print(row['Tweet_text'])
        positive_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))

    for index, row in negative_tweets.iterrows():
        # print(row['Tweet_text'])
        negative_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))

    for index, row in neutral_tweets.iterrows():
        # print(row['Tweet_text'])
        neutral_tweet_tokens.append(tknzr.tokenize(row['Tweet_text']))

    #---------------------------------------------

    # positive_tweets = twitter_samples.strings('positive_tweets.json')
    # negative_tweets = twitter_samples.strings('negative_tweets.json')

    # text = twitter_samples.strings('tweets.20150430-223406.json')
    # tweet_tokens = twitter_samples.tokenized('positive_tweets.json')[0]

    stop_words = stopwords.words('english')

    # positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    # negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    neutral_cleaned_tokens_list = []

    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in neutral_tweet_tokens:
        neutral_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    print("Most common occuring words in positive tweets")
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)
    print(freq_dist_pos.most_common(10))

    print("Most common occuring words in negative tweets")
    all_neg_words = get_all_words(negative_cleaned_tokens_list)
    freq_dist_neg = FreqDist(all_neg_words)
    print(freq_dist_neg.most_common(10))

    print("Most common occuring words in neutral tweets")
    all_neu_words = get_all_words(neutral_cleaned_tokens_list)
    freq_dist_neu = FreqDist(all_neu_words)
    print(freq_dist_neu.most_common(10))

    positive_tokens_for_model = get_tweets_for_model(positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(negative_cleaned_tokens_list)
    neutral_tokens_for_model = get_tweets_for_model(neutral_cleaned_tokens_list)

    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    neutral_dataset = [(tweet_dict, "Neutral")
                        for tweet_dict in neutral_tokens_for_model]

    dataset = positive_dataset + negative_dataset + neutral_dataset

    random.shuffle(dataset)
    print(len(dataset))
    train_data = dataset[:3000]
    test_data = dataset[3000:]

    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    # ALl other classifiers
    classifiers = {
        "BernoulliNB": BernoulliNB(),
        "ComplementNB": ComplementNB(),
        "MultinomialNB": MultinomialNB(),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(),
        "MLPClassifier": MLPClassifier(max_iter=1000),
        "AdaBoostClassifier": AdaBoostClassifier(),
    }

    train_count = 3000

    for name, sklearn_classifier in classifiers.items():

        classifier = nltk.classify.SklearnClassifier(sklearn_classifier)

        classifier.train(train_data)

        accuracy = nltk.classify.accuracy(classifier, test_data)

        print(F"{accuracy:.2%} - {name}")

    custom_tweet = "@Vince34359049 @normanswan Please see our letter "

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))