import numpy as np
import random


# TODO: add more activation functions


def activate(x, fn='sigmoid', grad=False):
    if fn == 'sigmoid':
        y = sigmoid(x)
        if grad == False:
            return y
        else:
            return y * (1 - y)


def featureScale(x):
    return (x - x.mean()) / x.std()


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
        return

    def __forwardPropagate(self, X):
        m, n = np.shape(X)
        activations = [0] * (self.hiddenLayers + 2)
        activations[0] = X
        for i in range(1, self.hiddenLayers + 2):
            if i != 1:
                activations[i - 1] = np.append(np.ones((m, 1)), activations[i - 1], axis=1)
            activations[i] = sigmoid(activations[i - 1].dot(self.network[i - 1].T))
        self.activations = activations

    def __backpropagate(self, y, Lambda):
        deltaNetwork = list()
        temp = np.zeros((self.hiddenUnits, self.inputUnits))
        deltaNetwork.append(temp)
        temp = []
        for i in range(1, self.hiddenLayers):
            temp.append(np.zeros((self.hiddenUnits, self.hiddenUnits + 1)))
        deltaNetwork += temp
        temp = np.zeros((self.outputUnits, self.hiddenUnits + 1))
        deltaNetwork.append(temp)
        for i in range(self.m):
            yi = y[i, :]
            deltas = [0] * (self.hiddenLayers + 1)
            deltas[-1] = (self.activations[-1][i, :] - yi).reshape(self.outputUnits, 1) * gradSigmoid(
                self.activations[-1][i, :])
            for j in reversed(range(self.hiddenLayers)):
                deltas[j] = np.dot(deltas[j + 1], self.network[j + 1]) * sigmoid(self.activations[j + 1][i, :].T)
                deltas[j] = deltas[j][:, 1:]
            for j in range(self.hiddenLayers + 1):
                n = np.shape(self.activations[j][i, :])
                # x = np.matrix(self.activations[j][i, :].reshape(n))
                deltaNetwork[j] += deltas[j].T.dot(self.activations[j][i, :].reshape((1, n[0])))

        if Lambda != 0:
            regNetwork = self.network
            for j in range(self.hiddenLayers + 1):
                regNetwork[j][:, 0] = 0
                deltaNetwork[j] += regNetwork[j] * Lambda
        for j in range(self.hiddenLayers + 1):
            deltaNetwork[j] /= self.m
        return deltaNetwork

    def __initializeNetwork(self):
        inputToHiddenTheta = 6 * np.random.rand(self.hiddenUnits, self.inputUnits) - 3
        self.network.append(inputToHiddenTheta)
        hiddenTheta = [None] * (self.hiddenLayers - 1)
        for i in range(self.hiddenLayers - 1):
            hiddenTheta[i] = 6 * np.random.rand(self.hiddenUnits, self.hiddenUnits + 1) - 3
        self.network += hiddenTheta
        hiddenToOutputTheta = 6 * np.random.rand(self.outputUnits, self.hiddenUnits + 1) - 3
        self.network.append(hiddenToOutputTheta)
        return

    def __costFunction(self, y, Lambda):
        error = (-1 / self.m) * (np.sum(y * np.log(self.activations[-1]) + (1 - y) * np.log(1 - self.activations[-1])))
        if Lambda != 0:
            regNetwrok = self.network
            regTerm = 0
            for i in range(self.hiddenLayers + 1):
                regTerm += np.square(np.sum(regNetwrok[i][:, 1:]))
            error = (Lambda / (2 * self.m)) * regTerm
        return error

    def train(self, X, y, hiddenUnits, alpha=0.01, iters=10000, Lambda=0, hiddenLayers=1):
        y = np.array(y)
        self.m, n = np.shape(X)
        m, self.outputUnits = np.shape(y)

        X = np.append(np.ones((self.m, 1)), X, axis=1)
        n += 1
        self.inputUnits = n
        self.hiddenUnits = hiddenUnits
        self.hiddenLayers = hiddenLayers
        # initialize network
        self.__initializeNetwork()
        stage = iters / 100
        Jvec = np.zeros(iters)
        for iter in range(iters):
            J = 0
            self.__forwardPropagate(X)
            J = self.__costFunction(y, Lambda)
            deltaNetwork = self.__backpropagate(y, Lambda)
            for j in range(self.hiddenLayers + 1):
                self.network[j] -= alpha * deltaNetwork[j]
            Jvec[iter] = J
            if iter % stage == 0:
                print(J)
        return Jvec

    def predict(self, X):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        self.__forwardPropagate(X)
        return np.round(self.activations[-1])
