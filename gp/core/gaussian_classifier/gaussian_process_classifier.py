import numpy as np
from gp.helper.data_tools.data_normalizer import DataNormalizerFactory
from gp.core.gaussian_process.gaussian_process import GaussianProcessFactory
from logreg.helper.data_extractor import DataExtractor



class GaussianProcessClassifier(object):
    def __init__(self, dataset, data_transformation=0):
        self.__data = self.__data_class(dataset)
        self.__gp = self.__create_gps()

    @staticmethod
    def __data_class(dataset):
        data_list = []
        for i in range(0, 10):
            data = DataExtractor.data_extraction(dataset.get_instances_with_label(i), "feature")
            data_list.append(data)
        return data_list

    def __create_gps(self):
        gp_list = []
        for i in range(0, 10):
            gp = GaussianProcessFactory.create_gaussian_process(self.__data[i])
            gp_list.append(gp)
        return gp_list

    def classify(self, test_set, target):
        pass