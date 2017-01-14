import abc
import numpy as np
from collections import OrderedDict


class DataInstance(object):
    __metaclass__ = abc.ABCMeta
    __features = {}
    __label = None
    __features_values_vector = None

    def __init__(self, features, label):
        self.__label = label
        self.__features = features
        self.__features_values_vector = np.array(self.__features.values())

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features):
        self.__features = features

    @property
    def label(self):
        return self.__label

    @label.setter
    def label(self, label):
        self.__label = label

    @property
    def features_values_vector(self):
        return self.__features_values_vector
