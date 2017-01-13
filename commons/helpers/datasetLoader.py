import cPickle as pickle
import os

from commons.exceptions.unableToSaveDatasetException import UnableToSaveDatasetException
from commons.models.constants.filePath import FilePath
from commons.models.constants.datasetType import DatasetType

from commons.exceptions.unableToLoadDatasetException import UnableToLoadDatasetException


class DatasetLoader(object):
    @staticmethod
    def load_data_set(data_set_type):
        print("".join(["Loading ", data_set_type, " data set..."]))
        try:
            data_set_file = None
            if data_set_type == DatasetType.TRAINING:
                data_set_file = FilePath.TRAINING_DATA_SET_FILE_NAME
            else:
                data_set_file = FilePath.TEST_DATA_SET_FILE_NAME
            return pickle.load(open(os.path.join(FilePath.DATA_FOLDER, data_set_file), "rb"))
        except Exception as e:
            raise UnableToLoadDatasetException(
                    "".join(["Unable to load: ", data_set_file, " data set with cause: ", e.message, "\n"]))

    @staticmethod
    def save_data_set(data_set, data_set_type):
        print("".join(["Saving ", data_set_type, " data set for future use..."]))
        try:
            data_set_file = None
            if data_set_type == DatasetType.TRAINING:
                data_set_file = FilePath.TRAINING_DATA_SET_FILE_NAME
            else:
                data_set_file = FilePath.TEST_DATA_SET_FILE_NAME

            pickle.dump(data_set, open(os.path.join(FilePath.DATA_FOLDER, data_set_file), "wb"))
        except Exception as e:
            raise UnableToSaveDatasetException(
                    "".join(["Unable to save: ", data_set_file, "data set with cause: ", e.message]))
