import numpy as np
from neural_network import filters
from neural_network.layers import Layer
import scipy

class Conv2D(Layer):
    def __init__(self, filters: list[np.ndarray] = filters.ALL_FILTERS):
        self.filters = filters

    def apply_conv(self, inputs, filter):
        transformed_image = scipy.signal.convolve2d(inputs, filter, mode='valid')
        transformed_image = np.clip(transformed_image, 0, 255)
        return transformed_image

    def calculate_outputs(self, inputs):
        imgs = []
        for filter in self.filters:
            imgs.append(self.apply_conv(inputs, filter))

        # print(imgs)

        output = np.stack(imgs, axis=-1)
        
        return output


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from neural_network.data import get_mnist_data
    import random

    x, y = get_mnist_data()
    random_index = random.randint(0, len(x) - 1)
    img = x[random_index]

    conv2d = Conv2D(filters.ALL_FILTERS)
    output = conv2d.calculate_outputs(img)

    # Display the original and transformed images side by side
    fig, axs = plt.subplots(3, 3, figsize=(9, 9))
    axs = axs.flatten()
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Base Image')

    for i, filter in enumerate(filters.ALL_FILTERS):
        axs[i+1].imshow(output[:, :, i], cmap='gray')

    plt.show()
