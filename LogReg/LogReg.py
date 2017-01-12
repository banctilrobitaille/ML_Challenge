import numpy as np
import math


class LogReg:
    __class = 10
    __basisFct = None
    __weights = None
    __bias = 0
    __features = None
    __prior = None
    __error = 0
    __learningRate = 0
    __targets = None

    def __init__(self, dataSet, learningRate=0.01):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        self.__learningRate = learningRate
        np.random.seed(0)
        self.__weights = np.random.rand(len(dataSet.data_instances[0].features))

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
        return self.__weights

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
    def features(self):
        return self.__features

    @features.setter
    def features(self, value):
        self.__features = value

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

    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, value):
        self.__targets = value

    def __getPrior(self):
        self.__prior = np.zeros(self.__class)
        for t in self.__features.flattenDataArray.target:
            self.__prior[t] += 1
        self.__prior /= self.__features.flattenDataArray.target.size

    def __softmax(self, W, X):
        numerator = np.dot(W, X)
        denominator = 0
        for i in xrange(10):
            denominator += np.dot(W[:, i], X)

        return np.exp(numerator) / np.exp(denominator)

    def __cost(self, predicts, targets, fct=0):
        cost = 0
        for predict, target in zip(predicts, targets):
            cost += (1 / 2) * math.pow((predict - target), 2)
        if fct == 0:
            return (1 / targets.size) * cost
        elif fct == 1:
            return cost

    def __updateWeights(self, W, X, cost):
        tmpW = W - self.__learningRate * cost * X
        self.__weights = tmpW

    def __maxProb(self, prob):
        return np.argmax(prob)

    def train(self):
        # ToDo Pass the feature from the dataSet
        prob = self.__softmax(self.__weights, self.__features)
        predict = self.__maxProb(prob)
        # ToDo Pass the target from the dataSet
        self.__error = self.__cost(predict, self.__features)
        self.__updateWeights(self.__weights, self.__features, self.__error)

    def __getFeatures(self, dataSet):
        listOfList = []
        for instance in dataSet.data_instances:
            list = instance.features.values()
            listOfList.append(list)
        features = np.array(listOfList)
        return features

    def __getLabel(self, dataSet):
        list = []
        for instance in dataSet.data_instances:
            list.append(instance.label)
        target = np.array(list)
        return target