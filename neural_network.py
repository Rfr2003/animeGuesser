from torch import nn
import torch
from torch import optim
from handleData import loadDataSet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

df = loadDataSet().drop(columns=['title'], axis=1)
df.dropna(inplace=True)
dfX = pd.get_dummies(df.drop(['rating'], axis=1), columns=['studio', 'genre1', 'genre2'], dtype=int)
dfy = df['rating'].replace({'PG-13 - Teens 13 or older':2, 'G - All Ages':0, 'PG - Children':1,'R - 17+ (violence & profanity)':3, 'R+ - Mild Nudity':4, 'Rx - Hentai':5})
print(dfy.value_counts())
X = torch.tensor(dfX.values, dtype=torch.float)
y = torch.tensor(dfy.values, dtype=torch.long)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = nn.Sequential(
    nn.Linear(355, 110),
    nn.ReLU(),
    nn.Linear(110, 100),
    nn.ReLU(),
    nn.Linear(100, 6)
)

Closs = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

epochs = 3000

for epoch in range(epochs):
    predictions = model(X_train)
    loss = Closs(predictions, y_train)
    loss.backward()
    opt.step()
    opt.zero_grad()

    if epoch % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch + 1}/{epochs}], CELoss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


model.eval()
with torch.no_grad():
    ## YOUR SOLUTION HERE ##
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

print(f'Accuracy: {accuracy:.4f}')
print(report)