import numpy as np

from neural_network import layer, neural_network
from neural_network import activation


def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(1, 2, activation.SiLU), 
    ])

    nn.layers[0].weights[0][0] = 1
    print(nn.calculate_outputs([-1]))
    print(nn.cost(nn.calculate_outputs([-1]), [1]))

if __name__ == "__main__":
    main()
