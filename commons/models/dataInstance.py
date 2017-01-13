import abc
from collections import OrderedDict


class DataInstance(object):
    __metaclass__ = abc.ABCMeta
    __features = {}
    __label = None

    def __init__(self, features, label):
        self.__label = label
        self.__features = features

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
        return OrderedDict(sorted(self.__features.items(), key=lambda t: t[0])).values()
