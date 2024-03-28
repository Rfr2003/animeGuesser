from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

from handleData import loadHashingVectorizer
from handleData import loadY
from handleData import loadSynopsis
from handleData import loadXandY
from handleData import loadEncoder

from matplotlib import pyplot as plt

import pandas as pd


def trainSypnosis():
    vect = loadHashingVectorizer()
    X = loadSynopsis().values.ravel()
    print(X.shape)
    y = loadY().astype('int')
    print(y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = vect.transform(X_train)
    clf = SGDClassifier(loss='log_loss', penalty='l2', random_state=1, max_iter=1)

    clf.fit(X_train, y_train)
    print(clf.score(X_train, y_train))

    X_test = vect.transform(X_test)
    print(clf.score(X_test, y_test))


def trainMLP():
    X, y = loadXandY()
    print(X.head(10))
    y = y.astype('int')
    print(y.head(10))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate': ['constant', 'adaptive'],
    }

    mlp = MLPClassifier(max_iter=1000, random_state=1)

    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(grid_search.score(X_train, y_train))


def trainSVM():
    X, y = loadXandY()
    print(X.head(10))
    y = y.astype('int')
    print(y.head(10))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'C': [0.1, 1, 10],               # Paramètre de régularisation
        'loss': ['hinge', 'squared_hinge'],  # Fonction de perte : charnière ou charnière quadratique
    }

    lg = LinearSVC(max_iter=3000, dual=True)

    grid_search = GridSearchCV(lg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)

    print(grid_search.score(X_train, y_train))



def trainTree():
    X, y = loadXandY()
    print(X.head(10))
    y = y.astype('int')
    print(y.head(10))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2'],
    }

    tree = DecisionTreeClassifier(criterion='entropy', max_depth=10, max_features='sqrt', min_samples_leaf=4,
                                  min_samples_split=10)

    tree.fit(X_train, y_train)

    print(tree.score(X_train, y_train))

    print(tree.score(X_test, y_test))

    encoder = loadEncoder('rating')

    for i in range(10):
        x = X.iloc[[i]]
        print(x)
        Y = y.loc[i]

        print("Label prédit : " + encoder.inverse_transform(pd.DataFrame(data=tree.predict(x), columns=['rating'])))
        print("Label réel : " + encoder.inverse_transform(Y.reshape(-1, 1)))
        print("\n")

#trainTree()

def trainForest():
    X, y = loadXandY()
    print(X.head(10))
    y = y.astype('int')
    print(y.head(10))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 300],  # Nombre d'arbres dans la forêt
        'max_depth': [None, 10, 20, 30],  # Profondeur maximale de chaque arbre
        'min_samples_split': [2, 5, 10],  # Nombre minimum d'échantillons requis pour diviser un nœud interne
        'min_samples_leaf': [1, 2, 4]  # Nombre minimum d'échantillons requis pour être une feuille
    }

    forest = RandomForestClassifier(max_depth=30, min_samples_split=4, min_samples_leaf=2, n_estimators=100)

    #grid_search = GridSearchCV(forest, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    forest.fit(X_train, y_train)


    #print(grid_search.best_params_)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))

    encoder = loadEncoder('rating')

    for i in range(10):
        x = X.iloc[[i]]
        print(x)
        Y = y.loc[i]

        print("Label prédit : " + encoder.inverse_transform(pd.DataFrame(data=forest.predict(x), columns=['rating'])))
        print("Label réel : " + encoder.inverse_transform(Y.reshape(-1, 1)))
        print("\n")

trainForest()
