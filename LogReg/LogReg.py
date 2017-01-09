import numpy as np

class LogReg:
    __feature = None
    __class = 10
    __basisFct = None
    __weights = None
    __bias = 0
    __dataSet = None
    __prior = None

    def __init__(self, dataSet):
        self.__dataSet = dataSet

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

    def __getPrior(self):
        self.__prior = np.zeros(self.__class)
        for t in self.__dataSet.flattenDataArray.target:
            self.__prior[t] += 1
        self.__prior /= self.__dataSet.flattenDataArray.target.size
