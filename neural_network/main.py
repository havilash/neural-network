import numpy as np
import pickle
import matplotlib.pyplot as plt

from neural_network import activations, costs, layer, nn as neural_network
from neural_network.data import get_mnist_data, get_augmented_mnist_data, train_test_split, create_batches

data = [
    ([-1, -2], [0, 1]),
    ([-2, -1], [1, 0]),
    ([-10, -2], [1, 0]),
]

def test_main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(2, 8, activations.Identity), 
        layer.Layer(8, 8, activations.Sigmoid),
        layer.Layer(8, 2, activations.Sigmoid),
    ])
    
    
    for _ in range(1000):
        nn.learn(data, .15)
        # print(nn.calculate_outputs(data[0][0]))

    for d in data:
        print(nn.calculate_outputs(d[0]))
    print(nn.cost([-2, -1], [1, 0]))

def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(28*28, 128, activations.ReLU), 
        layer.Layer(128, 10, activations.Softmax),
    ])
    
    one_hot = lambda y: np.eye(10)[y]

    x, y = get_mnist_data()
    x = x.reshape(-1, 28*28)
    y = np.array([one_hot(i) for i in y])
    data = np.array(list(zip(x, y)))

    """for epoch in range(5):
        for i, batch in enumerate(create_batches(data, 256)):
            print(i)
            print(nn.calculate_outputs(batch[0][0]), batch[0][1])
            nn.learn((batch), .2)
    
        with open('neural_network.pkl', 'wb') as f:
            pickle.dump(nn, f)"""

    with open('neural_network.pkl', 'rb') as f:
        nn: neural_network = pickle.load(f)

    for _ in range(10):
        i = np.random.randint(len(data))
        plt.imshow(data[i][0].reshape((28, 28)), cmap='gray')
        expected = np.argmax(data[i][1])
        predicted = np.argmax(nn.calculate_outputs(data[i][0]))
        plt.title(f"Expected: {expected}, Predicted: {predicted}")
        plt.show()

if __name__ == "__main__":
    main()
