import random
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

stop = stopwords.words('english')
stemmer = PorterStemmer()

df = pd.read_csv('data/data.csv', encoding='utf', usecols=[0, 2, 6], header=None)

df.columns = ['title', 'rating', 'synopsys']
df['rating'].replace({'PG - Children': 'G - All Ages', 'R - 17+ (violence & profanity)': 'PG-13 - Teens 13 or older', 'R+ - Mild Nudity':'PG-13 - Teens 13 or older'}, inplace=True)

df.dropna(inplace=True)
'''
df.synopsys = df.synopsys.str.lower()

print(df.synopsys[0])

synopsys = df['synopsys']

#tokenazation
synopsys_tokenized = []

for text in synopsys[:10]:
    sents = sent_tokenize(text)
    words = []
    for sent in sents:
        words.append(word_tokenize(sent))
    synopsys_tokenized.append(words)

print(synopsys_tokenized[0])

#stop word removal
for i in range(len(synopsys_tokenized)):
    for j in range(len(synopsys_tokenized[i])):
        synopsys_tokenized[i][j] = [word for word in synopsys_tokenized[i][j] if word not in stop]

print(synopsys_tokenized[0])

#stemming

for i in range(len(synopsys_tokenized)):
    for j in range(len(synopsys_tokenized[i])):
        synopsys_tokenized[i][j] = [stemmer.stem(word) for word in synopsys_tokenized[i][j]]

print(synopsys_tokenized[0])
'''
print(df['rating'].value_counts())
print(df['synopsys'][:10].values.tolist()[0])
print(len(df.title))
print(len(df.isnull()))

def stemming_tokenizer(str):
    words = re.sub(r"[^A-Za-z0-9\-\[\]]|\[.*\]", " ", str).lower().split()
    stem = [stemmer.stem(word) for word in words if word not in stop]
    return stem

vectorizer = TfidfVectorizer(norm=None, tokenizer=stemming_tokenizer, min_df=5)

x = vectorizer.fit_transform(df['synopsys'].values.tolist())
y = df['rating'].replace({'PG-13 - Teens 13 or older':1, 'G - All Ages':0,  'Rx - Hentai':2})

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

mb = MultinomialNB()

mb.fit(x_train, y_train)

print(mb.score(x_test, y_test))

index = random.randint(0, len(y) - 1)
print(mb.predict(vectorizer.transform(['the kid wants a new toy'])))

print(y[index])

print(df['synopsys'][index])

report = classification_report(y_test, mb.predict(x_test))

print(report)