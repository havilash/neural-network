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

        pooled_outputs = np.zeros((pooled_height, pooled_width, channels))
        for h in range(pooled_height):
            for w in range(pooled_width):
                for c in range(channels):
                    start_h = h * self.pool_size
                    end_h = start_h + self.pool_size
                    start_w = w * self.pool_size
                    end_w = start_w + self.pool_size

                    pooled_outputs[h, w, c] = np.max(inputs[start_h:end_h, start_w:end_w, c])

        return pooled_outputs

if __name__ == '__main__':
    from neural_network.data import get_mnist_data
    from neural_network.layers import Conv2D
    import random
    from matplotlib import pyplot as plt

    x, y = get_mnist_data()
    random_index = random.randint(0, len(x) - 1)
    img = x[random_index].reshape(28, 28)

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