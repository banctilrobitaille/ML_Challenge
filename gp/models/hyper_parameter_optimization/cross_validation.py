import numpy as np


class CrossValidation(object):
    def __init__(self, data_matrix, k):
        self.__data_matrix = CrossValidation.separate_data(data_matrix, k)
        self.__k = k

    @classmethod
    def separate_data(cls, data_matrix, k):
        nbr_per_fold = data_matrix.shape[0] / k
        data_matrix = np.reshape(data_matrix, (k, nbr_per_fold, data_matrix.shape[1]))
        return data_matrix

