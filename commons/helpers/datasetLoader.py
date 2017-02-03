import cPickle as pickle
import os

from commons.exceptions.unableToSaveDatasetException import UnableToSaveDatasetException
from commons.models.constants.filePath import FilePath
from commons.models.constants.datasetType import DatasetType

from commons.exceptions.unableToLoadDatasetException import UnableToLoadDatasetException


class DatasetLoader(object):
    @staticmethod
    def load_data_sets_for(classification_method):
        print("".join(["Loading data sets for ", classification_method, "..."]))
        try:
            data_set_file = classification_method + "_"
            return pickle.load(
                    open(os.path.join(FilePath.DATA_FOLDER, data_set_file + FilePath.TRAINING_DATA_SET_FILE_NAME),
                         "rb")), pickle.load(
                    open(os.path.join(FilePath.DATA_FOLDER, data_set_file + FilePath.TEST_DATA_SET_FILE_NAME),
                         "rb"))
        except Exception as e:
            raise UnableToLoadDatasetException(
                    "".join(["Unable to load: ", classification_method, " data set with cause: ", e.message, "\n"]))

    @staticmethod
    def load_data_set(data_set_type, classification_method):
        print("".join(["Loading ", data_set_type, " data set for ", classification_method, "..."]))
        try:
            data_set_file = classification_method + "_"
            if data_set_type == DatasetType.TRAINING:
                data_set_file += FilePath.TRAINING_DATA_SET_FILE_NAME
            else:
                data_set_file += FilePath.TEST_DATA_SET_FILE_NAME
            return pickle.load(open(os.path.join(FilePath.DATA_FOLDER, data_set_file), "rb"))
        except Exception as e:
            raise UnableToLoadDatasetException(
                    "".join(["Unable to load: ", classification_method, " data set with cause: ", e.message, "\n"]))

    @staticmethod
    def save_data_set_with_type(data_set, data_set_type, classification_method):
        print("".join(["Saving ", data_set_type, " data set for future use..."]))
        try:
            data_set_file = classification_method + "_"
            if data_set_type == DatasetType.TRAINING:
                data_set_file += FilePath.TRAINING_DATA_SET_FILE_NAME
            else:
                data_set_file += FilePath.TEST_DATA_SET_FILE_NAME

            pickle.dump(data_set, open(os.path.join(FilePath.DATA_FOLDER, data_set_file), "wb"))
        except Exception as e:
            raise UnableToSaveDatasetException(
                    "".join(["Unable to save: ", classification_method, "data set with cause: ", e.message]))

    @staticmethod
    def save_data_set(training_data_set, test_data_set, classification_method):
        try:
            DatasetLoader.save_data_set_with_type(training_data_set, DatasetType.TRAINING, classification_method)
            DatasetLoader.save_data_set_with_type(test_data_set, DatasetType.TEST, classification_method)
        except UnableToSaveDatasetException as e:
            print(e.message)
