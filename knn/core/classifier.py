from commons.models.classificationStats import ClassificationStats
from knn.models.model import Model


class KnnClassifier(object):
    @staticmethod
    def classify(training_dataset, test_dataset, number_of_neighbors):
        print("KNN classification in progess... \n")
        classification_stats = ClassificationStats(ClassificationStats.CLASSIFICATION_METHODS["KNN"])
        knn_model = Model().train_with_dataset(training_dataset)

        for data_instance in test_dataset.data_instances[:5]:
            estimated_label = knn_model.classify(data_instance, number_of_neighbors=number_of_neighbors)
            correctly_classified = data_instance.label == estimated_label
            classification_stats.register_data_instance_classification(label=data_instance.label,
                                                                       correctly_classified=correctly_classified)
        print(classification_stats)
