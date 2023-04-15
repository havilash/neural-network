import pytest
import numpy as np
from neural_network.activations import Identity, Sigmoid, ReLU, TanH, SiLU, Softmax

@pytest.mark.parametrize("activation, input, expected_output", [
    (Identity, np.array([-1, 0, 1]), np.array([-1, 0, 1])),
    (Sigmoid, np.array([-1, 0, 1]), np.array([0.26894142, 0.5, 0.73105858])),
    (ReLU, np.array([-1, 0, 1]), np.array([0, 0, 1])),
    (TanH, np.array([-1, 0, 1]), np.array([-0.76159416, 0., 0.76159416])),
    (SiLU, np.array([-1, 0, 1]), np.array([-0.26894142, 0., 0.73105858])),
    (Softmax, np.array([-1, 0, 1]), np.array([9.00305732e-02, 2.44728471e-01, 6.65240956e-01]))
])
def test_activation_func(activation, input, expected_output):
    assert np.allclose(activation.func(input), expected_output)

@pytest.mark.parametrize("activation", [Identity, Sigmoid, ReLU, TanH, SiLU, Softmax])
def test_activation_derivative(activation):
    x = np.random.rand(1)
    delta = 1e-5
    numerical_derivative = (activation.func(x + delta) - activation.func(x)) / delta
    assert np.allclose(numerical_derivative, activation.derivative(x), rtol=1e-3)