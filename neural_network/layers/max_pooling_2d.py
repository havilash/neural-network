import numpy as np
from neural_network import filters
from neural_network.layers import Layer



class MaxPooling2D(Layer):
    def __init__(self, pool_size: int = 2):
        self.pool_size = pool_size

    def calculate_outputs(self, inputs):
        height, width, channels = inputs.shape
        pooled_height = height // self.pool_size
        pooled_width = width // self.pool_size

        # Reshape the input array to prepare for max pooling
        reshaped_inputs = inputs.reshape(pooled_height, self.pool_size,
                                        pooled_width, self.pool_size,
                                        channels)

        # Perform max pooling along the second and fourth axes
        pooled_outputs = np.max(reshaped_inputs, axis=(1, 3))

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