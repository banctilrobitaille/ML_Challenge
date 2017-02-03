import abc
import numpy as np
from collections import OrderedDict


class DataInstance(object):
    __metaclass__ = abc.ABCMeta
    __label = None
    __features_values = None

    def __init__(self, features, label):
        self.__label = label
        self.__features_values = features

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label):
        self.__label = label

    @property
    def features_values_vector(self):
        return np.array(self.__features_values)
