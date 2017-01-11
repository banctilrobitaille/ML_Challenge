from commons.models.digit import Digit
from dataset import Dataset
import os
import struct
from array import array
import numpy as np


class DatasetFactory(object):
    TEST_IMAGES_FILE_NAME = 't10k-images.idx3-ubyte'
    TEST_LABELS_FILE_NAME = 't10k-labels.idx1-ubyte'
    TRAINING_IMAGES_FILE_NAME = 'train-images.idx3-ubyte'
    TRAINING_LABELS_FILE_NAME = 'train-labels.idx1-ubyte'
    DATA_DIRECTORY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/data'

    @classmethod
    def create_dataset_from_files(cls, data_type='training'):
        images = None
        labels = None

        if data_type == 'training':
            images, labels = cls.__load(os.path.join(cls.DATA_DIRECTORY, cls.TRAINING_IMAGES_FILE_NAME),
                                        os.path.join(cls.DATA_DIRECTORY, cls.TRAINING_LABELS_FILE_NAME))
        elif data_type == 'test':
            images, labels = cls.__load(os.path.join(cls.DATA_DIRECTORY, cls.TEST_IMAGES_FILE_NAME),
                                        os.path.join(cls.DATA_DIRECTORY, cls.TEST_LABELS_FILE_NAME))

        labels = np.array(labels)
        images = np.array(images).reshape((60000, 28, 28))

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
