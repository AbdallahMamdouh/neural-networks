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
    x[x<0]=0
    x[x>=0]=1
    return x


def reLu(x):
    x[x<0]=0
    return x


def tanH(x):
    return np.tanh(x)


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y


def gradSigmoid(x):
    return x * (1 - x)


class neural_layer:
    def __init__(self):
        self.units=0
        self.activation='sigmoid'
        self.layer=0
class NeuralNetwork:
    def __init__(self):
        self.activations = []
        self.layers = 0
        self.inputUnits = 0
        self.network = []
        np.random.seed(1)
        self.lastOutput = 0
        self.added_input = 0
        self.added_output = 0
        self.trainSize = 0
        return

    def load(self,fileName):
        return

    def add_input_layer(self, inputUnits):
        if not self.added_input:
            self.inputUnits = inputUnits + 1
            self.lastOutput = inputUnits
            self.activations.append(0)
            self.added_input = 1
        else:
            print("input layer already added")

    def add_hidden_layers(self, hiddenUnits, hiddenLayers=1,activation='sigmoid'):
        if self.added_input:
            for i in range(hiddenLayers):
                layer=neural_layer()
                layer.units=hiddenUnits
                layer.activation=activation
                layer.layer=6 * np.random.rand(hiddenUnits, self.lastOutput + 1) - 3
                self.network.append(layer)
                self.activations.append(0)
                self.lastOutput = hiddenUnits
            self.layers += hiddenLayers
        else:
            print("add input layer first")

    def add_output_layer(self, outputUnits,activation='sigmoid'):
        if not self.added_output and self.added_input:
            layer=neural_layer()
            layer.layer=6 * np.random.rand(outputUnits, self.lastOutput + 1) - 3
            layer.activation=activation
            layer.units=outputUnits
            self.activations.append(0)
            self.network.append(layer)
            self.added_output = 1
            self.layers+=1
        else:
            if self.added_output:
                print("output layer already added")
            else:
                print("add input layer first")

    def __forwardPropagate(self, X):
        m, n = np.shape(X)
        self.activations[0] = X
        for i in range(1, self.layers+1):
            if i != 1:
                self.activations[i - 1] = np.append(np.ones((m, 1)), self.activations[i - 1], axis=1)
            self.activations[i] = activate(self.activations[i - 1].dot(self.network[i - 1].layer.T), self.network[i-1].activation)

    def __backpropagate(self, y, Lambda):
        deltaNetwork = [0] * self.layers
        deltas = [0] * self.layers
        deltas[-1] = (self.activations[-1] - y) * activate(self.activations[-1], self.network[-1].activation, True)
        for i in reversed(range(self.layers-1)):
            deltas[i] = np.dot(deltas[i + 1], self.network[i + 1].layer) * activate(self.activations[i + 1], self.network[i+1].activation,
                                                                              True)
            deltas[i] = deltas[i][:, 1:]
        for i in range(self.layers):
            deltaNetwork[i] = deltas[i].T.dot(self.activations[i])
        error = (-1 / self.trainSize) * (
            np.sum(y * np.log(self.activations[-1]) + (1 - y) * np.log(1 - self.activations[-1])))
        if Lambda != 0:
            regTerm = 0
            regNetwork = self.network
            for i in range(self.layers):
                regNetwork[i].layer[:, 0] = 0
                deltaNetwork[i] += regNetwork[i].layer * Lambda
                regTerm += np.sum(np.square(regNetwork[i].layer))
            error += (Lambda / (2 * self.trainSize)) * regTerm
        for i in range(self.layers):
            deltaNetwork[i] /= self.trainSize
        return deltaNetwork, error

    def train(self, X, y, alpha=0.01, iters=10000, Lambda=0.01):
        self.trainSize, n = np.shape(X)
        X = np.append(np.ones((self.trainSize, 1)), X, axis=1)
        stage = iters / 100
        Jvec = np.zeros(iters)
        for iter in range(iters):
            self.__forwardPropagate(X)
            deltaNetwork, J = self.__backpropagate(y, Lambda)
            for i in range(self.layers):
                self.network[i].layer -= alpha * deltaNetwork[i]
            if iter % stage == 0:
                print("error: ", J)
            Jvec[iter] = J
        return Jvec

    def predict(self, X):
        m, n = np.shape(X)
        X = np.append(np.ones((m, 1)), X, axis=1)
        self.__forwardPropagate(X)
        return np.round(self.activations[-1])

    def save(self,fileName='model.txt'):
        file=open(fileName,'w+')

        return
