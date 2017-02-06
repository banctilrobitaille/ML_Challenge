import numpy as np


class DataNormalizer(object):
    @staticmethod
    def __normalize_vector_0_1(vector):
        return (vector - np.min(vector)) / (np.max(vector) - np.min(vector))

    @staticmethod
    def __scale_vector(vector, min_, max_):
        return ((vector - np.min(vector)) * (max_ - min_)) / (np.max(vector) - np.min(vector))

    @staticmethod
    def __standardize_vector(vector):
        return (vector - np.mean(vector)) / np.std(vector)

    @staticmethod
    def normalize_data(X, min_=0, max_=1):
        if min_ == 0 and max_ == 1:
            X = np.apply_along_axis(DataNormalizer.__normalize_vector_0_1, 1, X)
            return X
        else:
            X = np.apply_along_axis(DataNormalizer.__scale_vector, 1, vector=X, min_=min_, max_=max_)
            return X

    @staticmethod
    def standardize_data(X):
        X = np.apply_along_axis(DataNormalizer.__standardize_vector, 1, X)
        return X


class DataNormalizerFactory(object):
    @staticmethod
    def create_data_normalizer():
        return DataNormalizer()
