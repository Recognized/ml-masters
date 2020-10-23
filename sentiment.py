import numpy as np
import pandas
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline, FeatureUnion
from xgboost import XGBClassifier

import regexes
from dictionary import countWords


def read_dataset(file):
    dataframe = pandas.read_csv(file)
    dataframe['TweetText'] = dataframe['TweetText'].apply(preprocess_text)
    dataframe = dataframe[dataframe['Sentiment'] != 'irrelevant']
    dataframe['Sentiment'] = dataframe['Sentiment'].apply(transform_sentiment)
    return dataframe.sample(frac=1).reset_index(drop=True)


def transform_sentiment(x):
    if x == 'positive':
        return 1
    elif x == 'negative':
        return -1
    else:
        return 0


def organization(df):
    return df['TweetText'], df['Topic']


def preprocess_text(s):
    return regexes.preprocess(s)


def train_org(df, df_test):
    def read(frame):
        frame = frame[frame['TweetText'].apply(lambda x: len(x) > 15)]
        return frame['TweetText'].to_numpy(), frame['Topic'].to_numpy()

    def get_model(n_features, n_estimators):
        return make_pipeline(
            TfidfVectorizer(max_features=n_features),
            XGBClassifier(n_estimators=n_estimators),
        )

    x, y = read(df)
    x_true, y_true = read(df_test)

    best = None
    np.set_printoptions(threshold=np.inf)
    features = None
    bestBound = None
    for feature in [100, 200, 500, 1000]:
        for estimators in [100, 200, 500, 1000]:
            kf = KFold(n_splits=3)
            results = []
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                # print([sum([1 if x == org else 0 for x in y_train]) for org in ['apple', 'microsoft', 'google', 'twitter']])
                model = get_model(feature, estimators)
                model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append(f1_score(y_test, y_pred, average='weighted'))
            average = sum(results) / len(results)
            print("%s %s %s" % (feature, estimators, average))
            if best is None or average > best:
                best = average
                bestBound = estimators
                features = feature

    model = get_model(features, bestBound)
    model = model.fit(x, y)

    print("[Train organization] best f1 score -- %s" % best)
    print("[Train organization] best params -- %s %s" % (feature, bestBound))
    print("[Test organization] f1 score -- %s" % f1_score(y_true, model.predict(x_true), average='weighted'))

    return model


def train_sentiment(df, df_test, org):
    def read(frame):
        return frame['TweetText'].to_numpy(), frame['Sentiment'].to_numpy()

    x, y = read(df)
    x_true, y_true = read(df_test)

    best = None
    features = None
    bestBound = None
    for feature in [100, 200, 500, 1000]:
        for estimators in [100, 200, 500, 1000]:
            kf = KFold(n_splits=3)
            results = []
            for train_index, test_index in kf.split(x):
                X_train, X_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model = RoundLogisticRegression(estimators, feature, org)
                model = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                results.append(f1_score(y_test, y_pred, average='weighted'))
            average = sum(results) / len(results)
            print("%s %s %s" % (feature, estimators, average))
            if best is None or average > best:
                best = average
                bestBound = estimators
                features = feature

    model = RoundLogisticRegression(bestBound, features, org)
    model = model.fit(x, y)

    print("[Train sentiment] best f1 score -- %s" % best)
    print("[Train sentiment] best params -- %s" % bestBound)
    print("[Test sentiment] f1 score -- %s" % f1_score(y_true, model.predict(x_true), average='weighted'))

    return model


class RoundLogisticRegression:
    def __init__(self, n_estimators, features, org, **kwargs):
        self.pipe = make_pipeline(
            FeatureUnion([('s', TfidfVectorizer(max_features=features)), ('t', AddOrganizationPrediction(org)),
                          ('k', AddDictionaryFeatures())]),
            XGBClassifier(n_estimators=n_estimators),
        )

    def fit(self, X, y, **kwargs):
        self.pipe = self.pipe.fit(X, y, **kwargs)
        return self

    def predict(self, X):
        return self.pipe.predict(X)


class AddOrganizationPrediction(BaseEstimator, TransformerMixin):
    def __init__(self, org):
        self.org = org

    def fit(self, X, y, **kwargs):
        return self

    def transform(self, X):
        return self.org.predict_proba(X)


class AddDictionaryFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y, **kwargs):
        return self

    def f(self, x):
        t = countWords(x)
        return [v for (k, v) in t[0].items()] + [t[3], t[4]]

    def transform(self, X):
        return [self.f(x) for x in X]


def int_cast(x):
    if x:
        return 1
    else:
        return 0


if __name__ == '__main__':
    test_set = read_dataset("data/Test.csv")
    train_set = read_dataset("data/Train.csv")

    organization_predictor = train_org(train_set, test_set)
    sentiment_predictor = train_sentiment(train_set, test_set, organization_predictor)

    while True:
        tweet = preprocess_text(input("Enter tweet: "))
        check = input("What to classify (org, sentiment): ")
        if check == "org":
            result = organization_predictor.predict([tweet])
        elif check == "sentiment":
            result = sentiment_predictor.predict([tweet])
        else:
            continue

        print(result)
