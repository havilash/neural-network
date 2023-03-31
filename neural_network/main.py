import numpy as np

from neural_network import activations, layer, neural_network
from neural_network import costs


def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(2, 3, activations.Identity), 
        layer.Layer(3, 2, activations.Sigmoid),
    ])
    
    print(nn.calculate_outputs([-1, -2]))
    for _ in range(2000):
        nn.learn([-1, -2], [0, 1], .2)
        nn.learn([-2, -1], [1, 0], .2)
        nn.learn([-10, -2], [1, 0], .2)
        print(nn.calculate_outputs([-1, -2]))
        print(nn.calculate_outputs([-2, -1]))
        print(nn.calculate_outputs([-10, -2]))

    print(nn.cost([-2, -1], [1, 0]))

if __name__ == "__main__":
    main()
