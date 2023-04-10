import numpy as np

from neural_network import activations, layer, neural_network
from neural_network import costs


data = [
    ([-1, -2], [0, 1]),
    ([-2, -1], [1, 0]),
    ([-10, -2], [1, 0]),
]

def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(2, 8, activations.Identity), 
        layer.Layer(8, 2, activations.Sigmoid),
    ])
    
    
    for _ in range(10000):
        nn.learn(data, .15)
    print(nn.cost([-2, -1], [1, 0]))

if __name__ == "__main__":
    main()
