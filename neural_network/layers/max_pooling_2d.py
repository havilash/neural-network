import numpy as np
from neural_network import filters
from neural_network.layers import Layer
from skimage.measure import block_reduce

class MaxPooling2D(Layer):
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size

    def calculate_outputs(self, inputs):
        pooled_outputs = np.empty((inputs.shape[0] // 2, inputs.shape[1] // 2, inputs.shape[2]))
        for i in range(inputs.shape[2]):
            pooled_outputs[:, :, i] = block_reduce(inputs[:, :, i], (2, 2), np.max)

        return pooled_outputs

if __name__ == '__main__':
    from neural_network.data import get_mnist_data
    from neural_network.layers import Conv2D
    import random
    from matplotlib import pyplot as plt

    x, y = get_mnist_data()
    random_index = random.randint(0, len(x) - 1)
    img = x[random_index]

    conv2d = Conv2D(filters.ALL_FILTERS)
    max_pooling = MaxPooling2D()

    nparay = conv2d.calculate_outputs(img)
    output = max_pooling.calculate_outputs(nparay)

    # Display the original and transformed images side by side
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')

    for i, filter in enumerate(filters.ALL_FILTERS):
        axs[i].imshow(output[:, :, i-1], cmap='gray')
        axs[i].set_title(f'Filter {i}')

    plt.show()