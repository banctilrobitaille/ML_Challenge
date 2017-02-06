import numpy as np

from gp.models.kernel_computer.kernel import KernelComputerType
from gp.models.kernel_computer.kernel import KernelComputerFactory
from gp.models.hyper_parameter_optimization.marginal_log import ComputerMarginalLikelihoodFactory
from gp.models.hyper_parameter_optimization.kernel_optimization import KernelGradientFactory


class GaussianKernelType(KernelComputerType):
    NONE = "none"


class GaussianProcess(object):
    def __init__(self, kernel_type, dataset, sigma=1, l=1):
        self.__kernel = KernelComputerFactory.create_kernel_computer(kernel_type, sigma=sigma, l=l)
        self.__x = np.arange(dataset.shape[0])
        self.__fx = dataset
        self.__covar_matrix = self.__create_covariance_matrix(self.__x)


    def __create_covariance_matrix(self, input_):
        covar_matrix = np.zeros(input_.shape[0], input_.shape[0])
        for i in range(0, input_.shape[0]):
            for j in range(0, input_.shape[0]):
                covar_matrix[i, j] = self.__kernel.compute_kernel(input_[i], input_[j])
        return covar_matrix

    def __test_covariance_matrix(self, data, test_data):
        covar_matrix = np.zeros(test_data.shape[0])
        for i in range(0, test_data.shape[0]):
            covar_matrix[i] = self.__kernel.compute_kernel(data[i], test_data[i])
        return covar_matrix

    def fit(self, data, Y, test_data):
        self.__covar_matrix = self.__create_covariance_matrix(data)
        test_covar = self.__test_covariance_matrix(data, test_data)
        L = np.linalg.cholesky(self.__covar_matrix)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
        function = test_covar.T * alpha
        return function

    def __train(self, input_, Y, number_epoch=100, cost_treshold=0.5, debug=False):
        K = None
        likelihood_computer = ComputerMarginalLikelihoodFactory.create_marginal_computer()
        gradient_computer = KernelGradientFactory.create_kernel_gradient("squared exponential")
        i = 0
        for epoch in range(0, number_epoch):
            K = self.__create_covariance_matrix(input_)
            cost = likelihood_computer.compute_marginal_likelihood(K, Y)
            if debug:
                print cost
            l_op, sigma_op = gradient_computer.compute_gradient(input_[i, :], input_[i + 1, :])
            self.__kernel.l -= l_op
            self.__kernel.sigma -= sigma_op
            if cost < cost_treshold:
                break
            if i + 1 == input_.shape[0]:
                i = 0
        self.__covar_matrix = K


class GaussianProcessFactory(object):
    @staticmethod
    def create_gaussian_process(dataset, kernal_type="squared exponential"):
        return GaussianProcess(kernal_type, dataset)
