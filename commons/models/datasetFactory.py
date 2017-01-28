from commons.models.constants.filePath import FilePath
from commons.models.digit import Digit
from dataset import Dataset
import os
import struct
from array import array
import numpy as np


class DatasetFactory(object):
    @classmethod
    def create_dataset_from_files(cls, data_set_type):
        print("Creating " + data_set_type + " dataset from files... \n")
        images = None
        labels = None

        if data_set_type == 'training':
            images, labels = cls.__load(os.path.join(FilePath.DATA_FOLDER, FilePath.TRAINING_IMAGES_FILE_NAME),
                                        os.path.join(FilePath.DATA_FOLDER, FilePath.TRAINING_LABELS_FILE_NAME))
        elif data_set_type == 'test':
            images, labels = cls.__load(os.path.join(FilePath.DATA_FOLDER, FilePath.TEST_IMAGES_FILE_NAME),
                                        os.path.join(FilePath.DATA_FOLDER, FilePath.TEST_LABELS_FILE_NAME))

        labels = np.array(labels)
        images = np.array(images).reshape((len(images), 28, 28))

        return Dataset().with_data_instances(map(lambda image, label: Digit.from_image(image, label), images, labels))

    @classmethod
    def __load(cls, path_img, path_lbl):
        with open(path_lbl, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049,'
                                 'got {}'.format(magic))

            labels = array("B", file.read())

        with open(path_img, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051,'
                                 'got {}'.format(magic))

            image_data = array("B", file.read())

        images = []
        for i in range(size):
            images.append([0] * rows * cols)

        for i in range(size):
            images[i][:] = image_data[i * rows * cols:(i + 1) * rows * cols]

        return images, labels
