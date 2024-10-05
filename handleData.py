import numpy as np
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import HashingVectorizer
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import pickle
import os
from sklearn.preprocessing import OrdinalEncoder

stop = stopwords.words('english')
porter = PorterStemmer()


def tokenizer(text):
    text = str(text)
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def createHashingVectorizer():
    vect = HashingVectorizer(decode_error='ignore',
                             n_features=2 ** 21,
                             preprocessor=None,
                             tokenizer=tokenizer,
                             analyzer='word',
                             lowercase=False)

    data = pd.read_csv('data/data.csv', usecols=[6], header=None)

def loadDataSet():
    columns_names = ['title', 'episodes', 'rating', 'studio', 'genre1', 'genre2']
    data = pd.read_csv('data/data.csv', usecols=list(range(6)), header=None)
    data.columns = columns_names

    return data


def createEncoder():
    data = loadDataSet()
    data = data.drop(columns=['title'], axis=1)
    ratingEncoder = OrdinalEncoder()
    studioEncoder = OrdinalEncoder()
    genre1Encoder = OrdinalEncoder()
    genre2Encoder = OrdinalEncoder()

    ratingEncoder.fit(data.iloc[:, [1]])
    studioEncoder.fit(data.iloc[:, [2]])
    genre1Encoder.fit(data.iloc[:, [3]])
    genre2Encoder.fit(data.iloc[:, [4]])

    pickle.dump(ratingEncoder, open(os.path.join('data', 'rating_encoder.pkl'), 'wb'), protocol=4)
    pickle.dump(studioEncoder, open(os.path.join('data', 'studio_encoder.pkl'), 'wb'), protocol=4)
    pickle.dump(genre1Encoder, open(os.path.join('data', 'genre1_encoder.pkl'), 'wb'), protocol=4)
    pickle.dump(genre2Encoder, open(os.path.join('data', 'genre2_encoder.pkl'), 'wb'), protocol=4)

def loadHashingVectorizer():
    return pickle.load(open(os.path.join('data', 'HashingVectorizer.pkl'), 'rb'))

def loadEncoder(name):
    return pickle.load(open(os.path.join('data', str(name) + '_encoder.pkl'), 'rb'))


def encodeData(data):
    rating_encoder = loadEncoder('rating')
    studioEncoder = loadEncoder('studio')
    genre1Encoder = loadEncoder('genre1')
    genre2Encoder = loadEncoder('genre2')
    rating = rating_encoder.transform(data.iloc[:, [2]])
    studio = studioEncoder.transform(data.iloc[:, [3]])
    genre1 = genre1Encoder.transform(data.iloc[:, [4]])
    genre2 = genre2Encoder.transform(data.iloc[:, [5]])

    data['rating'] = rating
    data['studio'] = studio
    data['genre1'] = genre1
    data['genre2'] = genre2

    return data


def loadTitle():
    columns_names = ['title']
    data = pd.read_csv('data/data.csv', usecols=[0])
    data.columns = columns_names

    return data


def decodeData(encoded_data):
    rating_encoder = loadEncoder('rating')
    studioEncoder = loadEncoder('studio')
    genre1Encoder = loadEncoder('genre1')
    genre2Encoder = loadEncoder('genre2')
    rating = rating_encoder.transform(encoded_data.iloc[:, [2]])
    studio = studioEncoder.transform(encoded_data.iloc[:, [3]])
    genre1 = genre1Encoder.transform(encoded_data.iloc[:, [4]])
    genre2 = genre2Encoder.transform(encoded_data.iloc[:, [5]])

    encoded_data['rating'] = rating
    encoded_data['studio'] = studio
    encoded_data['genre1'] = genre1
    encoded_data['genre2'] = genre2

    return encoded_data

def handleNan():
    data = encodeData(loadDataSet())
    data = data.interpolate()

    return data

def loadY():
    data = handleNan()['rating']
    print(data.head())
    return data

def loadSynopsis():
    data = pd.read_csv('data/data.csv', usecols=[6], header=None)
    data = data.fillna(method='ffill')
    print(data.isna().sum())
    return data

def loadXandY():
    data = handleNan().drop('title', axis=1)
    X = data.drop('rating', axis=1)
    y = data['rating']

    return X, y

def show_data():
    data = loadDataSet()
    data.dropna(inplace=True)

    genres = data['genre1'].unique()
    rates = data['rating'].unique()
    studios = []
    studios_count = []
    for studio in data['studio'].unique():
        count = len(data[data['studio'] == studio])
        if count > 9:
            studios.append(studio)
            studios_count.append(count)

    genres_count = []
    rates_count = []

    for genre in genres:
        genres_count.append(len(data[data['genre1'] == genre]))

    for rate in rates:
        rates_count.append(len(data[data['rating'] == rate]))

    plt.figure(figsize=(25,25))
    plt.pie(genres_count, labels=genres, autopct='%1.1f%%', radius=1)
    plt.show()
    plt.close()

    plt.pie(rates_count, labels=rates, autopct='%1.1f%%', radius=1)
    plt.show()
    plt.close()

    plt.figure(figsize=(25, 25))
    plt.pie(studios_count, labels=studios, autopct='%1.1f%%', radius=1)
    plt.show()
    plt.close()

show_data()