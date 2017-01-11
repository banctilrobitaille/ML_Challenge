from commons.models.dataInstance import DataInstance
from commons.helpers.datasetHelper import DatasetHelper


class Digit(DataInstance):
    def __init__(self, features, label):
        DataInstance.__init__(self)
        self.features = features
        self.label = label

    @classmethod
    def from_image(cls, image, label):
        return cls(DatasetHelper.extract_features_from_image(image), label)
