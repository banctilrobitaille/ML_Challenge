from sklearn.datasets import fetch_mldata
from dataset import Dataset


class DatasetFactory:
    def __init__(self):
        pass

    def createDatasetFromOnlineResource(self):
        return Dataset(flatten_data_array=fetch_mldata('MNIST original'))
