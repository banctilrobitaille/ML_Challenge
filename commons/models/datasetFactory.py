from sklearn.datasets import fetch_mldata
from dataset import Dataset
import os
import struct
from array import array
import numpy as np


class DatasetFactory:
    def __init__(self):
        self.__path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) + '/data'
        self.__test_img_fname = 't10k-images.idx3-ubyte'
        self.__test_lbl_fname = 't10k-labels.idx1-ubyte'
        self.__train_img_fname = 'train-images.idx3-ubyte'
        self.__train_lbl_fname = 'train-labels.idx1-ubyte'

    def createDatasetFromOnlineResource(self):
        return Dataset(flatten_data_array=fetch_mldata('MNIST original'))

    def createDatasetFromFiles(self, file='train'):

        if file == 'train':
            tmpImg, tmpLabel = self.load(os.path.join(self.__path, self.__train_img_fname),
                                os.path.join(self.__path, self.__train_lbl_fname))
        elif file == 'test':
            tmpImg, tmpLabel = self.load(os.path.join(self.__path, self.__test_img_fname),
                      os.path.join(self.__path, self.__test_lbl_fname))

        labels = np.array(tmpLabel)
        images = np.array(tmpImg).reshape((60000, 28, 28))

        return images, labels

    @classmethod
    def load(cls, path_img, path_lbl):
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

