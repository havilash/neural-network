import numpy as np


class Activation:
    @staticmethod
    def func(x):
        pass

    @staticmethod
    def derivative(x):
        pass


class Sigmoid(Activation):
    @staticmethod
    def func(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.func(x) * (1 - Sigmoid.func(x))


class ReLU(Activation):
    @staticmethod
    def func(x):
        return np.maximum(x, 0)

    @staticmethod
    def derivative(x):
        return (x > 0).astype(int)


class TanH(Activation):
    @staticmethod
    def func(x):
        num = np.exp(2 * x)
        return (num - 1) / (num + 1)

    @staticmethod
    def derivative(x):
        return 1 - TanH.func(x) ** 2


class SiLU(Activation):
    @staticmethod
    def func(x):
        return x / (1 + np.exp(-x))

    @staticmethod
    def derivative(x):
        e_x = np.exp(-x)
        denominator = (1 + e_x) ** 2
        numerator = e_x * (x + 1) + x
        return numerator / denominator
    
    
class Identity(Activation):
    @staticmethod
    def func(x):
        return x

    @staticmethod
    def derivative(x):
        return 1
    

class Softmax(Activation):
    @staticmethod
    def func(x):
        exp_x = np.exp(x)
        return exp_x / np.sum(exp_x)

    @staticmethod
    def derivative(x):
        p = Softmax.func(x)
        return p * (1 - p)