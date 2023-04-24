import numpy as np
from neural_network import filters
from neural_network.layers import Layer

class Conv2D(Layer):
    def __init__(self, filters: [np.ndarray] = filters.ALL_FILTERS):
        self.filters = filters

    def apply_conv(self, inputs, filter):
        transformed_image = np.zeros_like(inputs)
        for x in range(1, inputs.shape[0] - 1):
            for y in range(1, inputs.shape[1] - 1):
                convolution = np.sum(filter * inputs[x - 1:x + 2, y - 1:y + 2])
                convolution = np.clip(convolution, 0, 255)

                transformed_image[x, y] = convolution

        return transformed_image

    def calculate_outputs(self, inputs):
        imgs = []
        for filter in self.filters:
            imgs.append(self.apply_conv(inputs, filter))


        output = np.stack(imgs, axis=-1)
        return output


if __name__ == '__main__':
    from matplotlib.image import imread
    import matplotlib.pyplot as plt
    from PIL import Image

    img = Image.open('eye-png-23.png')
    img_grey = img.convert('L')

    conv2d = Conv2D(filters.ALL_FILTERS)
    nparay = np.asarray(img_grey)
    output = conv2d.calculate_outputs(nparay)

    # Display the original and transformed images side by side
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    axs = axs.flatten()
    axs[0].imshow(img_grey, cmap='gray')
    axs[0].set_title('Original')

    for i, filter in enumerate(filters.ALL_FILTERS):
        axs[i].imshow(output[:, :, i-1], cmap='gray')
        axs[i].set_title(f'Filter {i}')

    plt.show()
