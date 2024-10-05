from torch import nn
import torch
from torch import optim
from handleData import loadDataSet
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler

df = loadDataSet().drop(columns=['title'], axis=1)
df.dropna(inplace=True)
scaler = MinMaxScaler()
df['episodes'] = scaler.fit_transform(df['episodes'].values.reshape(-1,1))
print(df['episodes'].head())
dfX = pd.get_dummies(df.drop(['rating'], axis=1), columns=['studio', 'genre1', 'genre2'], dtype=int)
df['rating'].replace({'PG - Children': 'G - All Ages', 'R - 17+ (violence & profanity)': 'PG-13 - Teens 13 or older', 'R+ - Mild Nudity':'PG-13 - Teens 13 or older'}, inplace=True)
dfy = df['rating'].replace({'PG-13 - Teens 13 or older':1, 'G - All Ages':0,  'Rx - Hentai':2})
print(dfy.value_counts())
X = torch.tensor(dfX.values, dtype=torch.float)
y = torch.tensor(dfy.values, dtype=torch.long)
print(X.shape)
print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = nn.Sequential(
    nn.Linear(X.shape[1], 350),
    nn.ReLU(),
    nn.Linear(350, 3)
)

Closs = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)

epochs = 5000

for epoch in range(epochs):
    predictions = model(X_train)
    loss = Closs(predictions, y_train)
    loss.backward()
    opt.step()
    opt.zero_grad()

    if epoch % 100 == 0:
        predicted_labels = torch.argmax(predictions, dim=1)
        accuracy = accuracy_score(y_train, predicted_labels)
        print(f'Epoch [{epoch}/{epochs}], CELoss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')


model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted_labels = torch.argmax(predictions, dim=1)
    accuracy = accuracy_score(y_test, predicted_labels)
    report = classification_report(y_test, predicted_labels)

print(f'Accuracy: {accuracy:.4f}')
print(report)