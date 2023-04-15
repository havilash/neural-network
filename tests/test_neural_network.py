import pytest
import numpy as np

from neural_network import activations, layer, nn as neural_network


data = [
    ([-1, -2], [0, 1]),
    ([-2, -1], [1, 0]),
    ([-10, -2], [1, 0]),
]

@pytest.fixture
def nn():
    return neural_network.NeuralNetwork([
        layer.Layer(2, 8, activations.Identity), 
        layer.Layer(8, 2, activations.Sigmoid),
    ])

def test_initial_outputs(nn):
    for d in data:
        outputs = nn.calculate_outputs(d[0])
        assert len(outputs) == 2

def test_learning(nn):
    for _ in range(1000):
        nn.learn(data, .15)
    cost = nn.cost(*data[0])
    assert cost < 0.1
