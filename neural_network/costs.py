import numpy as np


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
        return np.abs(output - expected_output)
     
    @staticmethod
    def derivative(output, expected_output):
        return np.sign(output - expected_output)


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
        return -(expected_output * np.log(output) + (1 - expected_output) * np.log(1 - output))
     
    @staticmethod
    def derivative(output, expected_output):
        return (output - expected_output) / (output * (1 - output))


class CategoricalCrossEntropy(Cost):
    @staticmethod
    def func(output, expected_output):
        return -expected_output * np.log(output)
     
    @staticmethod
    def derivative(output, expected_output):
        return output - expected_output

