import multiprocessing
import copy
from multiprocessing import Process

from commons.models.classificationStats import ClassificationStats
from commons.multi_processing.processManager import ProcessManager
from knn.models.model import Model
import numpy as np


class KnnClassifier(object):
    @staticmethod
    def classify(training_data_set, test_data_set, number_of_neighbors):
        print("KNN classification in progess... \n")
        classification_stats = ClassificationStats(ClassificationStats.CLASSIFICATION_METHODS["KNN"])
        knn_model = Model().train_with_dataset(training_data_set)

        classification_stats.set_classification_start_time()
        for data_instance in test_data_set.data_instances[:1000]:
            estimated_label = knn_model.classify(data_instance, number_of_neighbors=number_of_neighbors)
            correctly_classified = data_instance.label == estimated_label
            classification_stats.register_data_instance_classification(label=data_instance.label,
                                                                       correctly_classified=correctly_classified)
        print(classification_stats.to_string())


class MultiProcessedKnnClassifier(object):
    __number_of_neighbors = 0
    __process_manager = None
    __classification_stats = None
    __knn_model = None

    def __init__(self, training_data_set):
        ProcessManager.register('ClassificationStats', ClassificationStats)
        self.__process_manager = ProcessManager()
        self.__process_manager.start()
        self.__classification_stats = self.__process_manager.ClassificationStats(
                ClassificationStats.CLASSIFICATION_METHODS["PKNN"])
        self.__knn_model = Model().train_with_dataset(training_data_set)

    @property
    def knn_model(self):
        return self.__knn_model

    @property
    def classification_stats(self):
        return self.__classification_stats

    @property
    def number_of_neighbors(self):
        return self.__number_of_neighbors

    def classify(self, test_data_set, number_of_neighbors):
        print("".join(["KNN classification in progess with up to ", str(multiprocessing.cpu_count()),
                       " classification processes...\n"]))
        self.__number_of_neighbors = number_of_neighbors

        classification_threads = map(
                lambda data_instances: ClassificationProcess(self.__knn_model, data_instances,
                                                             self.__classification_stats, number_of_neighbors),
                np.array_split(test_data_set.data_instances[:1000], multiprocessing.cpu_count()))

        self.__classification_stats.set_classification_start_time()

        for classification_thread in classification_threads:
            classification_thread.start()

        for classification_thread in classification_threads:
            classification_thread.join()

        print(self.__classification_stats.to_string())


class ClassificationProcess(Process):
    __number_of_neighbors = 0
    __classification_stats = None
    __test_data_instances = None
    __knn_model = None

    def __init__(self, knn_model, test_data_instances, classification_stats, number_of_neighbors):
        super(ClassificationProcess, self).__init__()
        self.__knn_model = knn_model
        self.__classification_stats = classification_stats
        self.__test_data_instances = test_data_instances
        self.__number_of_neighbors = number_of_neighbors

    def run(self):
        for data_instance in self.__test_data_instances:
            estimated_label = self.__knn_model.classify(
                    data_instance,
                    number_of_neighbors=self.__number_of_neighbors)
            correctly_classified = data_instance.label == estimated_label
            self.__classification_stats.register_data_instance_classification(label=data_instance.label,
                                                                              correctly_classified=correctly_classified)
