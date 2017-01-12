from commons.models.datasetFactory import DatasetFactory
from knn.core.classifier import KnnClassifier as knn

if __name__ == '__main__':
    training_dataset = DatasetFactory.create_dataset_from_files(data_type='training')
    test_dataset = DatasetFactory.create_dataset_from_files(data_type="test")

    knn.classify(training_dataset=training_dataset, test_dataset=test_dataset, number_of_neighbors=10)
