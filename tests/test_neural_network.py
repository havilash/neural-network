import pytest

from neural_network import activations, nn as neural_network
from neural_network import layers

data = [
    ([-1, -2], [0, 1]),
    ([-2, -1], [1, 0]),
    ([-10, -2], [1, 0]),
]

@pytest.fixture
def nn():
    return neural_network.NeuralNetwork([
        layers.Dense(2, 8, activations.Identity), 
        layers.Dense(8, 2, activations.Sigmoid),
    ])

def test_outputs(nn):
    for d in data:
        outputs = nn.predict(d[0])
        assert len(outputs) == 2

def test_learning(nn):
    for _ in range(1000):
        nn.learn(data, .15)
    _, cost = nn.validate(data)
    assert cost < 0.1
