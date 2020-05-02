import numpy as np
import random


def featureScale(x):
    return (x - x.mean()) / x.std()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def gradSigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


class NeuralNetwork:
    def __init__(self):
        self.inputToHiddenTheta = 0
        self.hiddenTheta = []
        self.hiddenToOutputTheta = 0
        return

    def costFunction(self, X, y, Lambda,hiddenUnits,hiddenLayers):
        m, n = np.shape(X)
        dummy, outputSize = np.shape(y)
        # apply feedforward propagation to calculate activation units
        activIn = X

        z=X.dot(self.inputToHiddenTheta.T)
        a=sigmoid(z)
        zHidden=[z]
        activHidden=[a]
        for i in range(1,hiddenLayers-1):
            z=activHidden[i-1].dot(self.hiddenTheta[i])
            a=sigmoid(a)
            zHidden.append(z)
            activHidden.append(a)
        a = np.append(np.ones((m, 1)), a, axis=1)
        zOut=a.dot(self.hiddenToOutputTheta.T)
        activOut = sigmoid(zOut)

        # calculate cost function J
        J=(-1/m)*np.sum(y*np.log(activOut)+(1-y)*np.log(1-activOut))
        if Lambda!=0:
            regInputToHiddenTheta=self.inputToHiddenTheta[:,1:]
            regHiddenTheta=[]
            for i in range(hiddenLayers-1):
                regHiddenTheta.append(self.hiddenTheta[i][:,1:])
            regHiddenToOutputTheta=self.hiddenToOutputTheta[:,1:]
            error=(Lambda/(2*m))*(np.sum(np.square(regInputToHiddenTheta))+np.sum(np.square(regHiddenToOutputTheta)))
            for i in range(hiddenLayers-1):
                error+=np.sum(np.square(regHiddenTheta[i]))
            J=J+error
        print(J)
        # TODO:apply backpropagation to calculate deltas
        #initializing deltas to zeros
        delOutTheta=np.zeros((outputSize, hiddenUnits + 1))
        deltaHiddenTheta=[]
        for i in range(hiddenLayers - 1):
            deltaHiddenTheta.append(np.zeros((hiddenUnits, hiddenUnits + 1)))
        aHidden=[None]*(hiddenLayers-1)
        dHid=[None]*(hiddenLayers-1)
        for i in range(m):


            continue

        return

    def train(self, X, y, hiddenUnits, alpha=0.01, iters=10000, Lambda=0, hiddenLayers=1):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        dummy, outputSize = np.shape(y)
        n=n+1
        # initialize inputToHiddenTheta with random values of size hiddenUnits*n+1
        self.inputToHiddenTheta = np.random.rand(hiddenUnits, n)
        # initialize hiddenTheta as a cubic matrix with dimensions hiddenLayers*hiddenUnits*hiddenUnits+1
        for i in range(hiddenLayers - 1):
            self.hiddenTheta.append(np.random.rand(hiddenUnits, hiddenUnits + 1))
        # initialize hiddenToOutputTheta with random values of size hiddenUnits*outputSize
        self.hiddenToOutputTheta = np.random.rand(outputSize, hiddenUnits + 1)
        J=self.costFunction(X,y,Lambda,hiddenUnits,hiddenLayers)

        # TODO:apply gradient descent algorithm to minimize thetas
        return J

    def predict(self, X):
        return