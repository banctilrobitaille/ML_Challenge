import numpy as np


class LabelMapper(object):
    __output_vector = np.identity(10)

    def map_label_to_vector(self, label):
        return self.__output_vector[label, :]

    def map_vector_to_label(self, vector):
        return np.argmax(vector)
