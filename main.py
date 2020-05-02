from NeuralNetwork import NeuralNetwork,featureScale
import numpy as np, pandas as pd
data=pd.read_csv('data.csv')
X=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

model=NeuralNetwork()
y=np.reshape(y,(99,1))
model.train(X,y,5,hiddenLayers=3)
