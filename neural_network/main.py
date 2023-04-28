import numpy as np
import matplotlib.pyplot as plt
import math

from neural_network import activations, costs, nn as neural_network, layers, gui, constants
from neural_network.data import get_mnist_data, get_augmented_mnist_data, train_test_split
from neural_network.filters import ALL_FILTERS
from PIL import Image, ImageOps


def recognize(image: Image, nn_path = constants.DEFAULT_NN_PATH):
    with open(nn_path, 'rb') as f:
        nn = neural_network.NeuralNetwork.load(f)

    img = image.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS) 
    img = np.array(img)  / 255
    img = img.reshape(28, 28)
    prediction = nn.predict(img)

    # display image and prediction
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(img.reshape(28, 28), cmap='gray')
    ax1.axis('off')

    # display probabilities as bar chart
    classes = range(10)
    probabilities = prediction
    ax2.bar(classes, probabilities)
    ax2.set_xticks(classes)
    ax2.set_title('Probabilities')
    ax2.set_xlabel('Class')
    ax2.set_ylabel('Probability')

    plt.show()
    return prediction
    

def train():    
    nn = neural_network.NeuralNetwork([
        layers.Conv2D(ALL_FILTERS),
        # layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(26*26 * len(ALL_FILTERS), 128, activations.ReLU),
        layers.Dense(128, 10, activations.Softmax),
    ])

    one_hot = lambda y: np.eye(10)[y]

    # x, y = get_mnist_data(2000)
    x, y = get_augmented_mnist_data(10)  # needs some time
    y = np.array([one_hot(i) for i in y])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    train_data = np.array(list(zip(x_train, y_train)), dtype=object)    
    test_data = np.array(list(zip(x_test, y_test)), dtype=object)

    train_accuracies, train_costs = nn.train(
        train_data, 
        test_data, 
        learn_rate=0.2, 
        cost=costs.CategoricalCrossEntropy, 
        batch_size=64,
        epochs=3,
        save=True, 
        file_name="neural_network/neural_network.pkl",
        validate_per_batch=False,
        validate_interval=100,
        learn_method="threading",
    )
    
    #  plot train accuracies, costs
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(train_accuracies)
    ax1.set_title('Accuracy')
    ax1.set_ylim(0, 1)
    ax2.plot(train_costs)
    ax2.set_title('Cost')

    with open('neural_network/neural_network.pkl', 'rb') as f:
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

def main():
    # train()
    gui.GUI(recognize=lambda x: recognize(x, nn_path='neural_network/neural_network_final_with_softmax.pkl'))
    # gui.GUI(recognize=lambda x: recognize(x, nn_path='neural_network/neural_network_final_more_augmentation.pkl'))

if __name__ == "__main__":
    main()
