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

    def __init__(self, dataSet, learningRate=0.0001):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        self.__learningRate = learningRate
        np.random.seed()
        self.__weights = np.random.randn(len(dataSet.data_instances[0].features)+1, self.__class)

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
        numerator -= np.max(numerator)
        return np.exp(numerator) / np.sum(np.exp(numerator))

    def __updateWeights(self, prob, target, feature):
        index = np.argmax(target)
        self.__weights[:, index] -= self.__learningRate * self.__grad(prob,target,feature)


    def __maxProb(self, prob):
        return np.argmax(prob)

    def train(self):
        i = 1
        print "Training ...\n\n"
        for epoch in range(0, 10):
            for feature, target in zip(self.__features, self.__targets):
                prob = self.__softmax(self.__weights, feature)
                self.__error += (1/i) * self.__logLikelihood(target, prob)
                print self.__error
                if self.__error != 0:
                    self.__updateWeights(prob, target, feature)
                i += 1
        print "Training finished"
        print "------------------------------"

    def __getFeatures(self, dataSet):
        listOfList = []
        for instance in dataSet.data_instances:
            valueList = instance.features.values()
            valueList.append(1)
            listOfList.append(valueList)
        features = np.array(listOfList)
        return features

    def __getLabel(self, dataSet):
        target = np.zeros((len(dataSet.data_instances),self.__class))
        i = 0
        for instance in dataSet.data_instances:
            target[i,instance.label] = 1
            i += 1
        return target

    def __logLikelihood(self, target, prob):
        return -np.dot(target,np.log(prob))

    def __grad(self, prob, target, feature):
        return -(target[np.argmax(target)] - prob[np.argmax(prob)]) * feature

    def __predict(self, prob=None, feature=None):
        if prob is None:
            prob = self.__softmax(self.__weights,feature)
        index = np.argmax(prob)
        prediction = np.zeros(self.__class)
        prediction[index] = 1
        return prediction

    def test(self, dataSet):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        goodPred = 0
        badPred = 0
        print "Testing ..."
        for feature, target in zip(self.__features, self.__targets):
            prob = self.__predict(feature=feature)
            predict = np.argmax(prob)
            if predict == np.argmax(target):
                goodPred += 1
            else:
                badPred += 1
        pred = ((1.0*goodPred)/(goodPred+badPred)) * 100.0
        print "Percentage of good prediction is: " + str(pred)
