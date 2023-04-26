import pytest
import numpy as np
from neural_network.costs import MeanError, MeanSquaredError, BinaryCrossEntropy, CategoricalCrossEntropy

@pytest.mark.parametrize("cost, output, expected_output, expected_cost", [
    (MeanError, np.array([1]), np.array([2]), 1),
    (MeanSquaredError, np.array([1]), np.array([2]), 1),
    (BinaryCrossEntropy, np.array([0.5]), np.array([1]), 0.6931471805599453),
    (CategoricalCrossEntropy, np.array([0.5, 0.5]), np.array([1, 0]), 0.6931471805599453),
])
def test_cost_func(cost, output, expected_output, expected_cost):
    assert np.allclose(np.sum(cost.func(output, expected_output)), expected_cost)

@pytest.mark.parametrize("cost", [MeanError, MeanSquaredError, BinaryCrossEntropy])
def test_cost_derivative(cost):
    output = np.random.rand(1)
    expected_output = np.random.rand(1)
    delta = 1e-5
    numerical_derivative = (cost.func(output + delta, expected_output) - cost.func(output, expected_output)) / delta
    assert np.allclose(numerical_derivative, cost.derivative(output, expected_output), rtol=1e-2)
