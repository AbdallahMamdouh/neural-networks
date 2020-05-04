import numpy as np
import random


# TODO: add more activation functions


def activate(x, fn='sigmoid', grad=False):
    if fn == 'sigmoid':
        y = sigmoid(x)
        if not grad:
            return y
        else:
            return y * (1 - y)
    elif fn == 'reLu':
        if not grad:
            return reLu(x)
        else:
            return gradReLu(x)




def featureScale(x):
    return (x - x.mean()) / x.std()


def gradReLu(x):
    y=1 if x>0 else 0


def reLu(x):
    y=x if x>0 else 0


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def gradSigmoid(x):
    return x * (1 - x)


class NeuralNetwork:
    def __init__(self):
        self.activations = []
        self.hiddenLayers = 0
        self.hiddenUnits = 0
        self.inputUnits = 0
        self.outputUnits = 0
        self.network = []
        np.random.seed(1)
        self.activation='sigmoid'
        return

    def __forwardPropagate(self, X):
        m, n = np.shape(X)
        self.activations[0] = X
        for i in range(1, self.hiddenLayers + 2):
            if i != 1:
                self.activations[i - 1] = np.append(np.ones((m, 1)), self.activations[i - 1], axis=1)
            self.activations[i] = activate(self.activations[i - 1].dot(self.network[i-1].T),self.activation)

    def __backpropagate(self, y, Lambda):
        deltaNetwork = [0] * (self.hiddenLayers + 1)
        deltas = [0] * (self.hiddenLayers + 1)
        deltas[-1] = (self.activations[-1] - y) * activate(self.activations[-1],self.activation,True)
        for i in reversed(range(self.hiddenLayers)):
            deltas[i] = np.dot(deltas[i + 1], self.network[i + 1]) * activate(self.activations[i + 1],self.activation,True)
            deltas[i] = deltas[i][:, 1:]
        for i in range(self.hiddenLayers+1):
            deltaNetwork[i] = deltas[i].T.dot(self.activations[i])
        error = (-1 / self.m) * (np.sum(y * np.log(self.activations[-1]) + (1 - y) * np.log(1 - self.activations[-1])))
        if Lambda != 0:
            regTerm = 0
            regNetwork = self.network
            for i in range(self.hiddenLayers + 1):
                regNetwork[i][:, 0] = 0
                deltaNetwork[i] += regNetwork[i] * Lambda
                regTerm += np.sum(np.square(regNetwork[i][:, 1:]))
                error += (Lambda / (2 * self.m)) * regTerm
        for i in range(self.hiddenLayers + 1):
            deltaNetwork[i] /= self.m
        return deltaNetwork, error

    def __initializeNetwork(self):
        self.activations=[0]*(self.hiddenLayers+2)
        inputToHiddenTheta = 6 * np.random.rand(self.hiddenUnits, self.inputUnits) - 3
        self.network.append(inputToHiddenTheta)
        hiddenTheta = [None] * (self.hiddenLayers - 1)
        for i in range(self.hiddenLayers - 1):
            hiddenTheta[i] = 6 * np.random.rand(self.hiddenUnits, self.hiddenUnits + 1) - 3
        self.network += hiddenTheta
        hiddenToOutputTheta = 6 * np.random.rand(self.outputUnits, self.hiddenUnits + 1) - 3
        self.network.append(hiddenToOutputTheta)
        return

    def train(self, X, y, hiddenUnits, alpha=0.01, iters=10000, Lambda=0, hiddenLayers=1,activation='sigmoid'):
        self.activation=activation
        y = np.array(y)
        self.m, n = np.shape(X)
        X = np.append(np.ones((self.m, 1)), X, axis=1)
        m,self.outputUnits=np.shape(y)
        n += 1
        self.inputUnits = n
        self.hiddenUnits = hiddenUnits
        self.hiddenLayers = hiddenLayers
        # initialize network
        self.__initializeNetwork()
        stage = iters / 100
        Jvec = np.zeros(iters)
        for iter in range(iters):
            self.__forwardPropagate(X)
            deltaNetwork, J = self.__backpropagate(y, Lambda)
            for j in range(self.hiddenLayers + 1):
                self.network[j] -= alpha * deltaNetwork[j]
            if iter%stage==0:
                print("error: ",J)
            Jvec[iter] = J
        return Jvec

    def predict(self, X):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        self.__forwardPropagate(X)
        return np.round(self.activations[-1])
