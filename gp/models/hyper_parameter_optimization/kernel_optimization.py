import numpy as np
from gp.models.kernel_computer.kernel import KernelComputerType
import abc
import math


class KernelGradientOptimization(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def compute_gradient(**kwargs):
        raise NotImplementedError


class SQEKernelGradient(KernelGradientOptimization):
    @staticmethod
    def compute_gradient(**kwargs):
        sigma = 0
        l = 0
        X = None
        X_ = None
        gradient_sigma = 0
        gradient_l = 0

        for name, value in kwargs.items():
            if name == "l":
                l = value
            elif name == "sigma":
                sigma = value
            elif name == "X":
                X = value
            elif name == "X_":
                X_ = value

        gradient_l = -((2 * math.pow(sigma, 2)) / math.pow(l, 3)) * math.exp(
            (math.pow(np.linalg.norm((X - X_)), 2)) / 2 * math.pow(l, 2))
        gradient_sigma = -(2 * sigma) * math.exp((math.pow(np.linalg.norm((X - X_)), 2)) / 2 * math.pow(l, 2))

        return gradient_l, gradient_sigma


class KernelGradientFactory(KernelComputerType):
    @staticmethod
    def create_kernel_gradient(kernel_type):
        if kernel_type == KernelComputerType.SQUARED_EXPONENTIAL:
            return SQEKernelGradient()
        else:
            raise NotImplementedError
