import numpy as np

from neural_network import activations, costs, layer, neural_network
from neural_network.data import get_mnist_data, get_augmented_mnist_data, train_test_split, create_batches

def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(28*28, 128, activations.Sigmoid), 
        layer.Layer(128, 10, activations.Sigmoid),
    ])
    
    one_hot = lambda y: np.eye(10)[y]

    x, y = get_mnist_data()
    x = x.reshape(-1, 28*28)
    y = np.array([one_hot(i) for i in y])
    data = np.array(list(zip(x, y)))
    data = create_batches(data, 256)

    for epoch in range(10):
        for i, batch in enumerate(data):
            print(i)
            print(nn.calculate_outputs(batch[0][0]), batch[0][1])
            nn.learn((batch), .15)
        print(nn.calculate_outputs(data[0][0]))
        print(data[0])
    # import matplotlib.pyplot as plt
    # plt.imshow(x, cmap='gray')
    # plt.show()

if __name__ == "__main__":
    main()
