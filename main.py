from NeuralNetwork import NeuralNetwork, featureScale
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

data = pd.read_csv('data_banknote_authentication.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X = featureScale(X)
m = np.shape(y)
y = np.reshape(y, (m[0], 1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = NeuralNetwork()

j = model.train(X_train, y_train, alpha=0.03, hiddenUnits=10, hiddenLayers=2, Lambda=0)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0, 0] + cm[1, 1]) / sum(cm)
plt.plot(j)
plt.show()
