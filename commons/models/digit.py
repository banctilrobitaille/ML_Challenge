from commons.models.dataInstance import DataInstance
from commons.helpers.datasetHelper import DatasetHelper


class Digit(DataInstance):
    def __init__(self, features, label):
        DataInstance.__init__(self, features, label)

    @classmethod
    def from_image(cls, image, label, with_data_normalization, with_threshold, number_of_features):
        return cls(DatasetHelper.extract_features_from_image(image, with_data_normalization, with_threshold,
                                                             number_of_features), label)
