import numpy as np
#from models.learning.learning_optimization import

class LogReg(object):
    __class = 10
    __basisFct = None
    __weights = None
    __bias = 0
    __features = None
    __error = 0
    __cost = None
    __learningRate = 0
    __targets = None

    def __init__(self, dataSet, learningRate=0.1):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        self.__learningRate = learningRate
        np.random.seed(300)
        self.__weights = np.random.rand(len(dataSet.data_instances[0].features) + 1, self.__class)

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
        Values = np.dot(X, W)
        Values = np.apply_along_axis(self.__minArray,1,Values)
        Values = np.apply_along_axis(self.__classProb,1,Values)
        return Values

    def __updateWeights(self, prob, target, feature):
        self.__weights -= self.__learningRate * self.__grad(prob, target, feature)

    def __maxProb(self, prob):
        return np.argmax(prob)

    def train(self):
        print "Training ...\n\n"
        for epoch in range(0, 10000):
            prob = self.__softmax(self.__weights, self.__features)
            self.__error = self.__logLikelihood(self.__targets, prob)
            print self.__error
            #if self.__error != 0:
            #self.__updateWeights(prob, self.__targets, self.__features)
            grad = self.__grad(prob, self.__targets, self.__features)
            self.__weights -= self.__learningRate * self.__grad(prob, self.__targets, self.__features)#map(lambda x,y: x * y, self.__grad(prob, self.__targets, self.__features),self.__learningRate)
            if self.__error < 0.5:
                break
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
        target = np.zeros((len(dataSet.data_instances), self.__class))
        i = 0
        for instance in dataSet.data_instances:
            target[i, instance.label] = 1
            i += 1
        return target

    def __logLikelihood(self, target, prob):
        return (-(np.log(prob)*target).sum(1)).mean()
        #sum = 0
        #for i in xrange(0,target.shape[0]):
        #    sum += -target[i]*np.log(prob[i])
        #return sum

    def __grad(self, prob, target, feature):
        return -(np.dot(feature.T,(target - prob)))/feature.shape[0]

    def __predict(self, prob=None, feature=None):
        if prob is None:
            prob = self.__softmax(self.__weights, feature)
        prediction = np.apply_along_axis(self.__classPredict,1,prob)
        return prediction

    def test(self, dataSet):
        self.__features = self.__getFeatures(dataSet)
        self.__targets = self.__getLabel(dataSet)
        accuracy = 0.0
        print "Testing ..."
        predict = self.__predict(feature=self.__features)
        for p,t in zip(predict, self.__targets):
            if np.array_equal(p,t):
                accuracy += 1.0
        accuracy = (accuracy/self.__features.shape[0])*100
        print "Percentage of good prediction is: " + str(accuracy)

    def __minArray(self,X):
        X -= np.max(X)
        return X

    def __classProb(self,X):
        X =  np.exp(X) / np.sum(np.exp(X))
        return X

    def __classPredict(self,X):
        P = np.zeros(self.__class)
        P[np.argmax(X)] = 1
        return P

    def __classCompare(self,X,T):
        if np.array_equal(X,T):
            return 1
        else:
            return 0
