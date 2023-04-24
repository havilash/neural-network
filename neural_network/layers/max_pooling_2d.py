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

    from matplotlib.image import imread
    import matplotlib.pyplot as plt
    from PIL import Image
    from conv_2d import Conv2D

    conv2d = Conv2D(filters.ALL_FILTERS)

    img = Image.open('eye-png-23.png')
    img_grey = img.convert('L')

    max = MaxPooling2D()
    arr = np.asarray(img_grey)
    nparay = conv2d.calculate_outputs(arr)
    output = max.calculate_outputs(nparay)

    # Display the original and transformed images side by side
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    axs[0].imshow(img_grey, cmap='gray')
    axs[0].set_title('Original')

    for i, filter in enumerate(filters.ALL_FILTERS):
        axs[i].imshow(output[:, :, i-1], cmap='gray')
        axs[i].set_title(f'Filter {i}')

    plt.show()