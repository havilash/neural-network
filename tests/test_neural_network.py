import pytest
import numpy as np

from neural_network import activations, layer, nn


def test_layer_weight_biases():
    l = layer.Layer(3, 2)

    l.biases[0] = 3
    assert np.array_equal([3, 0], l.calculate_outputs([1, 1, 1]))

    l.weights[0][0] = 4
    assert np.array_equal([7, 0], l.calculate_outputs([1, 1, 1]))

    l.weights[2][0] = 4
    assert np.array_equal([11, 0], l.calculate_outputs([1, 1, 1]))

    l.weights[2][1] = 3
    assert np.array_equal([11, 3], l.calculate_outputs([1, 1, 1]))


def test_nn_layer_creation():
    nn = nn.NeuralNetwork([
        layer.Layer(3, 2, activations.Identity), 
    ])
    assert (
        3 == nn.layers[0].num_nodes_in
        and 2 == nn.layers[0].num_nodes_out
        and len(nn.layers) == 1
    )


def test_nn_outputs():
    nn = nn.NeuralNetwork([
        layer.Layer(3, 2, activations.Identity), 
    ])
    nn.layers[0].weights[0][0] = 4
    nn.layers[0].weights[1][0] = 4
    nn.layers[0].biases[0] = 1

    nn.layers[0].weights[2][1] = 2
    nn.layers[0].biases[1] = 5
    assert np.array_equal([9, 7], nn.calculate_outputs([1, 1, 1]))
