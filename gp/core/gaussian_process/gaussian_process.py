import numpy as np
from gp.models.kernel_computer.kernel import KernelComputerType
from gp.models.kernel_computer.kernel import KernelComputerFactory


class GaussianKernelType(KernelComputerType):
    NONE = "none"


class GaussianProcess(object):
    def __init__(self, kernel_type, input_1):
        self.__kernel = KernelComputerFactory.create_kernel_computer(kernel_type, sigma=1, l=1)
        self.__covar_matrix = self.create_covariance_matrix(input_1)
        self.__mean = input_1.mean()

    def create_covariance_matrix(self, input_1):
        covar_matrix = np.zeros(input_1.shape[0], input_1.shape[0])
        for i in range(0, input_1.shape[0]):
            for j in range(0, input_1.shape[0]):
                covar_matrix[i, j] = self.__kernel.compute_kernel(input_1[i], input_1[j],)
        return covar_matrix

    def function_covariance(self, input_):
        pass