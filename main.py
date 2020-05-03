from NeuralNetwork import NeuralNetwork,featureScale
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import multilayer_perceptron
data=pd.read_csv('data.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
X=featureScale(X)
y=np.reshape(y,(99,1))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

model=NeuralNetwork()

j=model.train(X_train,y_train,6,alpha=0.1,hiddenLayers=2,iters=10000)
y_pred=model.predict(X_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
plt.plot(j)
plt.show()