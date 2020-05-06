from NeuralNetwork import NeuralNetwork, featureScale
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder,Normalizer

data = pd.read_csv('ionosphere.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
m,n=np.shape(X)
scaler=Normalizer().fit(X)
X=scaler.transform(X)

enc=LabelEncoder()
y=enc.fit_transform(y)
y=np.reshape(y,(np.shape(y)[0],1))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model = NeuralNetwork()
model.add_input_layer(n)
model.add_hidden_layers(10,1,activation='sigmoid')
model.add_output_layer(1,activation='sigmoid')

j = model.train(X_train, y_train, alpha=0.03, Lambda=0.1, iters=100000)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy = (cm[0, 0] + cm[1, 1]) / sum(sum(cm))

print("accuracy: ", accuracy)
plt.plot(j)
plt.show()
