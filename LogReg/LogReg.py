import numpy as np
import math


class LogReg:
    __class = 10
    __basisFct = None
    __weights = None
    __bias = 0
    __features = None
    __error = 0
    __cost = None
    __learningRate = 0
    __targets = None

    def __init__(self, dataSet, learningRate=0.01):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        self.__learningRate = learningRate
        np.random.seed(0)
        self.__weights = np.random.rand(len(dataSet.data_instances[0].features)+1, self.__class)

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
    def error(self):
        return self.__error

    @error.setter
    def error(self, value):
        self.__error = value

    @property
    def learningRate(self):
        return self.__learningRate

    @learningRate.setter
    def learningRate(self, value):
        self.__learningRate = value

    @property
    def targets(self):
        return self.__targets

    @targets.setter
    def targets(self, value):
        self.__targets = value

    def __softmax(self, W, X):
        numerator = np.dot(X,W)
        denominator = 0
        for i in xrange(10):
            denominator += np.dot(W[:, i], X)

        return np.exp(numerator) / np.exp(denominator)

    def __updateWeights(self, W, grad, target):
        self.__weights[:, target] = W[:, target] - self.__learningRate * grad

    def __maxProb(self, prob):
        return np.argmax(prob)

    def train(self):
        for feature, target in zip(self.__features, self.__targets):
            prob = self.__softmax(self.__weights, feature)
            self.__error = self.__logLikelihood(prob, target)
            print self.__error
            self.__updateWeights(self.__weights,self.__grad(prob,target,feature),target)

    def __getFeatures(self, dataSet):
        listOfList = []
        for instance in dataSet.data_instances:
            valueList = instance.features.values()
            valueList.append(1)
            listOfList.append(valueList)
        features = np.array(listOfList)
        return features

    def __getLabel(self, dataSet):
        list = []
        for instance in dataSet.data_instances:
            list.append(instance.label)
        target = np.array(list)
        return target

    def __logLikelihood(self, prob, target):
        return -math.log(1-prob[target])

    def __grad(self, prob, target, feature):
        return (prob[target]-target) * feature
