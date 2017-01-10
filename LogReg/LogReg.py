import numpy as np
import math


class LogReg:
    __feature = None
    __class = 10
    __basisFct = None
    __weights = None
    __bias = 0
    __dataSet = None
    __prior = None
    __error = 0
    __learningRate = 0

    def __init__(self, dataSet, learningRate=0.01):
        self.__dataSet = dataSet
        self.__learningRate = learningRate
        np.random.seed(0)
        self.__weights = np.random.rand(self.__class + 1)

    @property
    def feature(self):
        return self.__feature

    @feature.setter
    def feature(self, value):
        self.__feature = value

    @property
    def classes(self):
        return self.__class

    @classes.setter
    def classes(self, value):
        self.__class = value

    @property
    def basisFct(self):
        return self.__basisFct

    @basisFct.setter
    def basisFct(self, value):
        self.__basisFct = value

    @property
    def weights(self):
        return self.weights

    @weights.setter
    def weights(self, value):
        self.__weights = value

    @property
    def bias(self):
        return self.__bias

    @bias.setter
    def bias(self, value):
        self.__bias = value

    @property
    def dataSet(self):
        return self.__dataSet

    @dataSet.setter
    def dataSet(self, value):
        self.__dataSet = value

    @property
    def prior(self):
        return self.__prior

    @prior.setter
    def prior(self, value):
        self.__prior = value

    @property
    def error(self):
        return self.__error

    @error.setter
    def error(self, value):
        self.__error = value

    def __getPrior(self):
        self.__prior = np.zeros(self.__class)
        for t in self.__dataSet.flattenDataArray.target:
            self.__prior[t] += 1
        self.__prior /= self.__dataSet.flattenDataArray.target.size

    def __softmax(self, index, W, X):
        numerator = np.dot(W[:, index], X)
        denominator = 0
        for i in xrange(10):
            denominator += np.dot(W[:, i], X)

        return math.exp(numerator) / math.exp(denominator)

    def __cost(self, predicts, targets, fct=0):
        if fct == 0:
            cost = 0
            for predict, target in zip(predicts, targets):
                cost += (1 / 2) * math.pow((predict - target), 2)
            return (1 / targets.size) * cost
