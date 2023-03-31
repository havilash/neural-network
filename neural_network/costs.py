import math


class Cost:
    @staticmethod
    def func(output, expected_output):
        pass   

    @staticmethod
    def derivative(output, expected_output):
        pass


class MeanError(Cost):
    @staticmethod
    def func(output, expected_output):
        return abs(output - expected_output)
     
    @staticmethod
    def derivative(output, expected_output):
        return math.copysign(1, output - expected_output)


class MeanSquaredError(Cost):
    @staticmethod
    def func(output, expected_output):
        return (output - expected_output) ** 2
     
    @staticmethod
    def derivative(output, expected_output):
        return 2 * (output - expected_output)


class BinaryCrossEntropy(Cost):
    @staticmethod
    def func(output, expected_output):
        return -(expected_output * math.log(output) + (1 - expected_output) * math.log(1 - output))
     
    @staticmethod
    def derivative(output, expected_output):
        return (output - expected_output) / (output * (1 - output))


class CategoricalCrossEntropy(Cost):
    @staticmethod
    def func(output, expected_output):
        return -sum(expected_output * math.log(output))
     
    @staticmethod
    def derivative(output, expected_output):
        return -expected_output / output


class SparseCategoricalCrossEntropy(Cost):
    @staticmethod
    def func(output, expected_output):
        return -math.log(output[expected_output])
     
    @staticmethod
    def derivative(output, expected_output):
        result = [0] * len(output)
        result[expected_output] = -1 / output[expected_output]
        return result