from commons.models.datasetFactory import DatasetFactory
from knn.core.classifier import KnnClassifier as knn
from knn.core.classifier import MultiProcessedKnnClassifier as multi_processed_knn

if __name__ == '__main__':
    training_data_set = DatasetFactory.create_dataset_from_files(data_type='training')
    test_data_set = DatasetFactory.create_dataset_from_files(data_type="test")

    # knn.classify(training_data_set=training_data_set, test_data_set=test_data_set, number_of_neighbors=10)
    multi_processed_knn(training_data_set=training_data_set).classify(test_data_set=test_data_set,
                                                                      number_of_neighbors=10)
