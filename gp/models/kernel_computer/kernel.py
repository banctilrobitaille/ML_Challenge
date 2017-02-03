import numpy as np
import math
import abc


class KernelComputerType(object):
    LINEAR = "linear"
    POLYNOMIAL = "polynomial"
    SQUARED_EXPONENTIAL = "squared exponential"


class KernelComputer(object):
    __metaclass__ = abc.ABCMeta

    def compute_kernel(self, vector_1, vector_2):
        raise NotImplementedError


class LinearKernel(KernelComputer):
    def __init__(self, c=0):
        self.__c = c

    def compute_kernel(self, vector_1, vector_2):
        return np.dot(vector_1.T, vector_2) + self.__c


class PolynomialKernel(KernelComputer):
    def __init__(self, c=0, d=1, alpha=1):
        self.__c = c
        self.__d = d
        self.__alpha = alpha

    def compute_kernel(self, vector_1, vector_2):
        return math.pow(self.__alpha * np.dot(vector_1.T, vector_2) + self.__c, self.__d)


class SquaredExponentialKernel(KernelComputer):
    def __init__(self, sigma, l):
        self.__sigma = sigma
        self.__l = l

    def compute_kernel(self, vector_1, vector_2):
        return math.pow(self.__sigma, 2.0) * math.exp(
            (math.pow(np.linalg.norm((vector_1 - vector_2)), 2)) / 2 * math.pow(self.__l, 2))


class KernelComputerFactory(object):
    @staticmethod
    def create_kernel_computer(kernel_type, c=0, d=1, alpha=1, sigma=1, l=1):
        if kernel_type == KernelComputerType.LINEAR:
            return LinearKernel(c)
        elif kernel_type == KernelComputerType.POLYNOMIAL:
            return PolynomialKernel(c, d, alpha)
        elif kernel_type == KernelComputerType.SQUARED_EXPONENTIAL:
            return SquaredExponentialKernel(sigma, l)
