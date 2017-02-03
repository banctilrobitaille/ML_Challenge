import numpy as np
from gp.models.kernel_computer.kernel import KernelComputerType


class GaussianKernelType(KernelComputerType):
    NONE = "none"



class GaussianProcess(object):
    def __init__(self):
        self