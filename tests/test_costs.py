import pytest
import numpy as np
from neural_network.costs import MeanError, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy, SparseCategoricalCrossEntropy

@pytest.mark.parametrize("cost_func, output, expected_output, expected_cost", [
    (MeanError, np.array([1]), np.array([2]), 1),
    (MeanSquaredError, np.array([1]), np.array([2]), 1),
    (BinaryCrossEntropy, np.array([0.5]), np.array([1]), 0.6931471805599453),
    (CategoricalCrossEntropy, np.array([0.5, 0.5]), np.array([1, 0]), 0.6931471805599453),
    (SparseCategoricalCrossEntropy, np.array([0.5, 0.5]), 0, 0.6931471805599453)
])
def test_cost_func(cost_func, output, expected_output, expected_cost):
    assert np.allclose(cost_func.func(output, expected_output), expected_cost)

@pytest.mark.parametrize("cost_func", [MeanError, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy])
def test_cost_derivative(cost_func):
    output = np.random.rand(1)
    expected_output = np.random.rand(1)
    delta = 1e-5
    numerical_derivative = (cost_func.func(output + delta, expected_output) - cost_func.func(output, expected_output)) / delta
    assert np.allclose(numerical_derivative, cost_func.derivative(output, expected_output), rtol=1e-3)

def test_sparse_categorical_crossentropy_derivative():
    cost_func = SparseCategoricalCrossEntropy
    output = np.array([0.1, 0.2, 0.7])
    expected_output = 2
    delta = 1e-5
    numerical_derivative = (cost_func.func(output + delta, expected_output) - cost_func.func(output, expected_output)) / delta
    assert np.allclose(numerical_derivative, cost_func.derivative(output, expected_output)[expected_output], rtol=1e-3)