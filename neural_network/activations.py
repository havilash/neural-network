import math


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
        return 1 / (1 + math.exp(-x))

    @staticmethod
    def derivative(x):
        return Sigmoid.func(x) * (1 - Sigmoid.func(x))


class ReLU(Activation):
    @staticmethod
    def func(x):
        return max(x, 0)

    @staticmethod
    def derivative(x):
        if x > 0:
            return 1
        else:
            return 0


class TanH(Activation):
    @staticmethod
    def func(x):
        num = math.exp(2 * x)
        return (num - 1) / (num + 1)

    @staticmethod
    def derivative(x):
        return 1 - TanH.func(x) ** 2


class SiLU(Activation):
    @staticmethod
    def func(x):
        return x / (1 + math.exp(-x))

    @staticmethod
    def derivative(x):
        e_x = math.exp(-x)
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
        return math.exp(x) / (1 + math.exp(x))

    @staticmethod
    def derivative(x):
        p = Softmax.func(x)
        return p * (1 - p)
