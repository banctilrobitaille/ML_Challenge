import abc
import numpy as np


class Neuron(object):
    __metaclass__ = abc.ABCMeta

    __bias = 0
    __weight = 0

    def __init__(self):
        pass

    def initialize(self):
        self.__bias = np.random.randn()
        self.__weight = np.random.randn()


class SigmoidNeuron(Neuron):
    def __init__(self):
        super(SigmoidNeuron, self).__init__()
        self.initialize()
