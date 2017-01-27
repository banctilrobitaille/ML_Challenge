import numpy as np


class DataType(object):
    FEATURE = "feature"
    LABEL = "label"


class DataExtractor(object):
    @staticmethod
    def data_extraction(dataset, data_type):
        if data_type == DataType.FEATURE:
            return DataExtractor.feature_extraction(dataset)
        elif data_type == DataType.LABEL:
            return DataExtractor.label_extraction(dataset)

    @classmethod
    def feature_extraction(cls, dataset):
        list_of_list = []
        for instance in dataset.data_instances:
            value_list = instance.features.values()
            value_list.append(1)
            list_of_list.append(value_list)
        features = np.array(list_of_list)
        return features

    @classmethod
    def label_extraction(cls, dataset):
        # TODO: Find a way to now how many class
        target = np.zeros((len(dataset.data_instances), 10))
        i = 0
        for instance in dataset.data_instances:
            target[i, instance.label] = 1
            i += 1
        return target
