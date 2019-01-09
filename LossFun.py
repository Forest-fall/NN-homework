import numpy as np 
from ActivationFunc import Sigmoid

class Quadratic(object):

    @staticmethod
    def cost_result(z, a, y):
        """return the output_result of this cost function"""
        return 0.5 * (a - y)**2

    @staticmethod
    def output_delta(z, a, y):
        """return the derivative of this cost function"""
        sigmoid_detrivative = Sigmoid(z)
        return (a - y) * sigmoid_detrivative


class LogLikehood(object):

    @staticmethod
    def cost_result(z, a, y):
        """return the output_result of this cost function"""
        return (- y * np.log(a))

    @staticmethod
    def output_delta(z, a, y):
        """return the derivative of this cost function"""
        return (a - y)


class CrossEntropy(object):

    @staticmethod
    def cost_result(z, a, y):
        """return the output_result of this cost function"""
        return (-y * np.log(a) - (1 - y) * np.log(1-a))

    @staticmethod
    def output_delta(z, a, y):
        """return the derivative of this cost function"""
        pass