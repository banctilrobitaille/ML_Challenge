import numpy as np

from logreg.helper.data_extractor import DataExtractor
from logreg.models.learning.learning_optimization import LearningProcessFactory
from commons.models.classificationStats import ClassificationStats
from logreg.models.feature_computers.prediction_computer import ProbabilityComputerFactory


class LogRegClassifier(object):
    def __init__(self, dataset, learning_rate=0.1):
        self.__features = DataExtractor.data_extraction(dataset, "feature")
        self.__targets = DataExtractor.data_extraction(dataset, "label")
        np.random.seed(300)
        self.__weights = np.random.rand(self.__features.shape[1], 10)
        self.__learning_rate = learning_rate

    def train(self, number_epoch=1000, cost_threshold=0.001, debug=False):
        print "Training logistic regression...."
        learning_process = LearningProcessFactory.create_learning_process(self.__learning_rate, number_epoch,
                                                                          cost_threshold, debug)
        self.__weights = learning_process.learn(self.__weights, self.__targets, self.__features)
        print "Training finished"

    def classify(self, dataset):
        features_to_classify = DataExtractor.data_extraction(dataset, "feature")
        targets_to_predict = DataExtractor.data_extraction(dataset, "label")
        prediction_computer = ProbabilityComputerFactory.create_probability_computer("predict")

        stat = ClassificationStats(ClassificationStats.CLASSIFICATION_METHODS["LOREG"])
        stat.set_classification_start_time()
        predictions = prediction_computer.predict(np.dot(features_to_classify, self.__weights))
        for prediction, target in zip(predictions, targets_to_predict):
            stat.register_data_instance_classification(np.argmax(target),
                                                       prediction_computer.prediction_is_true(prediction, target))
        print stat.to_string()
