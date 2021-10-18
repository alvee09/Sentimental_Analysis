# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import nltk
import csv
# nltk.download('twitter_samples')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('punkt')
import pandas as pd
import os
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import twitter_samples, stopwords
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier
from pandas import Series
import re, string, random
from nltk.tokenize import TweetTokenizer
import collections
from nltk.metrics.scores import precision
from nltk.metrics.scores import recall

tknzr = TweetTokenizer()
from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from nltk.metrics import f_measure

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
        token = re.sub("\s+\Z","",token) #remove the back
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

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words and token != "..." and token != "’" and token != 'covid':
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

    # Read from general dataset with positive/negative sentiments

    # file = open('dataset.csv')
    # csvreader = csv.reader(file)
    # df = pd.read_csv(file)
    # print(df.head())
    # df = df.sample(frac=1).reset_index(drop=True)
    # print(df.head())
    # df = df.iloc[:50000]
    # print(df.head())

    # ---------------------------------------------
    df = pd.read_csv('allMerged.csv', index_col=False)

    # Merge hastags with Tweet------------
    df['Tweet_text_merged'] = df.Tweet_text.astype(str).str.cat(df.hashtags.astype(str), sep=' ')
    # ----------------------
    # df = df[['Sentiment_Label', 'Tweet_text']]
    df = df[['Sentiment_Label', 'Tweet_text_merged']]
    print("1")
    positive_tweets = df.loc[df['Sentiment_Label'] == 'positive']
    negative_tweets = df.loc[df['Sentiment_Label'] == 'negative']
    neutral_tweets = df.loc[df['Sentiment_Label'] == 'neutral']

    positive_tweets = positive_tweets[['Tweet_text_merged']]
    negative_tweets = negative_tweets[['Tweet_text_merged']]
    neutral_tweets = neutral_tweets[['Tweet_text_merged']]

    positive_tweet_tokens = []
    negative_tweet_tokens = []
    neutral_tweet_tokens = []

    for index, row in positive_tweets.iterrows():
        positive_tweet_tokens.append(tknzr.tokenize(row['Tweet_text_merged']))
        # positive_tweet_tokens.append(nltk.word_tokenize(row['Tweet_text']))

    for index, row in negative_tweets.iterrows():
        negative_tweet_tokens.append(tknzr.tokenize(row['Tweet_text_merged']))
        # negative_tweet_tokens.append(nltk.word_tokenize(row['Tweet_text']))

    for index, row in neutral_tweets.iterrows():
        neutral_tweet_tokens.append(tknzr.tokenize(row['Tweet_text_merged']))
        # neutral_tweet_tokens.append(nltk.word_tokenize(row['Tweet_text']))

    # ---------------------------------------------
    stop_words = stopwords.words('english')

    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    neutral_cleaned_tokens_list = []
    print("2")
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
    train_data = dataset[:31000]
    print(len(train_data))
    test_data = dataset[31000:]
    print(len(test_data))
    # print("3")
    classifier = NaiveBayesClassifier.train(train_data)

    print("Accuracy is:", classify.accuracy(classifier, test_data))

    print(classifier.show_most_informative_features(10))

    # Precision and recall
    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    for i, (feats, label) in enumerate(test_data):
        refsets[label].add(i)
        observed = classifier.classify(feats)
        testsets[observed].add(i)

    print("Naive Bayes Classifier")
    print('Precision Positive:', precision(refsets['Positive'], testsets['Positive']))
    print('Recall Positive:', recall(refsets['Positive'], testsets['Positive']))
    print('F-measure Positive: ', f_measure(refsets['Positive'], testsets['Positive']))

    print('Precision Negative:', precision(refsets['Negative'], testsets['Negative']))
    print('Recall Negative:', recall(refsets['Negative'], testsets['Negative']))
    print('F-measure Negative: ', f_measure(refsets['Negative'], testsets['Negative']))

    print('Precision Neutral:', precision(refsets['Neutral'], testsets['Neutral']))
    print('Recall Neutral:', recall(refsets['Neutral'], testsets['Neutral']))
    print('F-measure Neutral: ', f_measure(refsets['Neutral'], testsets['Neutral']))
    print("")

    # ALl other classifiers
    classifiers = {
        "SGDClassifier": SGDClassifier(max_iter=1000),
        # "MultinomialNB": MultinomialNB(),
        "LinearSVC": LinearSVC(),
        # "BernoulliNB": BernoulliNB(),
        "ComplementNB": ComplementNB(),

        # "KNeighborsClassifier": KNeighborsClassifier(),
        # "DecisionTreeClassifier": DecisionTreeClassifier(),
        "RandomForestClassifier": RandomForestClassifier(),
        "LogisticRegression": LogisticRegression(max_iter=10000),
        "MLPClassifier": MLPClassifier(),
        # "AdaBoostClassifier": AdaBoostClassifier(),
    }



    train_count = 31000
    # print("4")
    for name, sklearn_classifier in classifiers.items():
        classifier = nltk.classify.SklearnClassifier(sklearn_classifier)

        classifier.train(train_data)

        accuracy = nltk.classify.accuracy(classifier, test_data)

        print(F"{accuracy:.2%} - {name}")

        refsets = collections.defaultdict(set)
        testsets = collections.defaultdict(set)

        for i, (feats, label) in enumerate(test_data):
            refsets[label].add(i)
            observed = classifier.classify(feats)
            testsets[observed].add(i)


   
        print('Precision Positive:', precision(refsets['Positive'], testsets['Positive']))
        print('Recall Positive:', recall(refsets['Positive'], testsets['Positive']))
        print('F-measure Positive: ', f_measure(refsets['Positive'], testsets['Positive']))

        print('Precision Negative:', precision(refsets['Negative'], testsets['Negative']))
        print('Recall Negative:', recall(refsets['Negative'], testsets['Negative']))
        print('F-measure Negative: ', f_measure(refsets['Negative'], testsets['Negative']))

        print('Precision Neutral:', precision(refsets['Neutral'], testsets['Neutral']))
        print('Recall Neutral:', recall(refsets['Neutral'], testsets['Neutral']))
        print('F-measure Neutral: ', f_measure(refsets['Neutral'], testsets['Neutral']))
        print("")

    custom_tweet = "This is bad and wrong"
    # print("5")
    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(dict([token, True] for token in custom_tokens)))
