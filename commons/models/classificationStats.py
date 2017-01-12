from threading import RLock
from datetime import datetime


class ClassificationStats(object):
    CLASSIFICATION_METHODS = {
        "KNN": "K Nearest Neighbors",
        "PKNN": "Multi Processed K Nearest Neighbors",
        "NN": "Neural Networks",
    }

    __write_lock = RLock()
    __classification_method = ""
    __total_number_of_classified_data_instances = 0
    __number_of_correctly_classified_data_instances = 0
    __number_of_correctly_classified_data_instance_by_label = {}
    __number_of_classified_data_instances_by_label = {}
    __success_rate_by_label = {}
    __classification_start_time = None
    __classification_run_time = None

    def __init__(self, classification_method):
        self.__classification_method = classification_method

    def set_classification_start_time(self):
        self.__classification_start_time = datetime.now()

    def register_data_instance_classification(self, label, correctly_classified=False):
        try:
            self.__write_lock.acquire()
            self.__total_number_of_classified_data_instances += 1

            if label not in self.__number_of_classified_data_instances_by_label:
                self.__success_rate_by_label[label] = 0
                self.__number_of_correctly_classified_data_instance_by_label[label] = 0
                self.__number_of_classified_data_instances_by_label[label] = 1
            else:
                self.__number_of_classified_data_instances_by_label[label] += 1

            if correctly_classified:
                self.__number_of_correctly_classified_data_instances += 1
                self.__number_of_correctly_classified_data_instance_by_label[label] += 1

            self.__success_rate_by_label[label] = str(float(
                    self.__number_of_correctly_classified_data_instance_by_label[label]) / float(
                    self.__number_of_classified_data_instances_by_label[label]) * 100) + "%"
        finally:
            self.__write_lock.release()

    def to_string(self):
        return "".join(
                ["--------------------------------CLASSIFICATION RESULTS--------------------------------" + "\n",
                 "Classification method: " + str(self.__classification_method) + "\n\n",
                 "Total number of classified data instances: " + str(
                         self.__total_number_of_classified_data_instances) + "\n\n",
                 "Success rate by label: " + str(self.__success_rate_by_label) + "\n\n",
                 "Global success rate: " + str(
                         float(self.__number_of_correctly_classified_data_instances) / float(
                                 self.__total_number_of_classified_data_instances) * 100) + "%" + "\n\n",
                 "Runtime: " + str(
                         (datetime.now() - self.__classification_start_time).total_seconds() / 60) + "minutes \n",
                 "--------------------------------------------------------------------------------------" + "\n"])
