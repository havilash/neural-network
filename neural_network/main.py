import numpy as np
import matplotlib.pyplot as plt

from neural_network import activations, costs, layer, nn as neural_network
from neural_network.data import get_mnist_data, get_augmented_mnist_data, train_test_split, create_batches


def main():
    nn = neural_network.NeuralNetwork([
        layer.Layer(28*28, 128, activations.ReLU), 
        layer.Layer(128, 10, activations.Softmax),
    ])
    
    one_hot = lambda y: np.eye(10)[y]

    x, y = get_mnist_data()
    x = x.reshape(-1, 28*28)
    y = np.array([one_hot(i) for i in y])
    x_train, x_test, y_train, y_test = train_test_split(x, y)
    train_data = np.array(list(zip(x_train, y_train)))
    test_data = np.array(list(zip(x_test, y_test)))
    
    """
    nn.train(train_data, test_data, 0.25, cost=costs.CategoricalCrossEntropy, batch_size=256, save=True)
    """

    with open('neural_network.pkl', 'rb') as f:
        nn = neural_network.NeuralNetwork.load(f)

    fig, axes = plt.subplots(3, 3, figsize=(6, 6))
    for i in range(3):
        for j in range(3):
            k = np.random.randint(len(test_data))
            axes[i, j].imshow(test_data[k][0].reshape((28, 28)), cmap='gray')
            expected = np.argmax(test_data[k][1])
            predicted = np.argmax(nn.predict(test_data[k][0]))
            axes[i, j].set_title(f"Expected: {expected}\nPredicted: {predicted}")
            axes[i, j].set_xticklabels([])
            axes[i, j].set_yticklabels([])
            axes[i, j].tick_params(axis='both', which='both', length=0)
    fig.subplots_adjust(hspace=0.5)
    plt.show()

if __name__ == "__main__":
    main()
