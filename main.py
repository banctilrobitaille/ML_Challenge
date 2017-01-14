from commons.exceptions.unableToLoadDatasetException import UnableToLoadDatasetException
from commons.exceptions.unableToSaveDatasetException import UnableToSaveDatasetException
from commons.helpers.datasetLoader import DatasetLoader
from commons.models.constants.datasetType import DatasetType
from commons.models.datasetFactory import DatasetFactory
from knn.core.classifier import KnnClassifier as knn
from knn.core.classifier import MultiProcessedKnnClassifier as multi_processed_knn
import numpy as np

if __name__ == '__main__':
    training_data_set = None
    test_data_set = None

    try:
        training_data_set = DatasetLoader.load_data_set(DatasetType.TRAINING)
        test_data_set = DatasetLoader.load_data_set(DatasetType.TEST)
    except UnableToLoadDatasetException as e:
        try:
            print(e.message)
            training_data_set = DatasetFactory.create_dataset_from_files(data_set_type=DatasetType.TRAINING)
            test_data_set = DatasetFactory.create_dataset_from_files(data_set_type=DatasetType.TEST)
            DatasetLoader.save_data_set(training_data_set, DatasetType.TRAINING)
            DatasetLoader.save_data_set(test_data_set, DatasetType.TEST)
        except UnableToSaveDatasetException as e:
            print(e.message)

            # knn.classify(training_data_set=training_data_set, test_data_set=test_data_set, number_of_neighbors=10)
    multi_processed_knn(training_data_set=training_data_set).classify(test_data_set=test_data_set,
                                                                      number_of_neighbors=10)
