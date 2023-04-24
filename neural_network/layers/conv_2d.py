import numpy as np
import random
import math

from neural_network import activations, costs


# Prewitt Filters
# Sobel Filters
# Laplacian Filter
# Robinson Compass Masks
# Krisch Compass Masks

class Conv2D():
    def __init__(self, input_array: np.ndarray, filters: dict) -> []:
        self.input_array = input_array
        self.size_x = input_array.shape[0]
        self.size_y = input_array.shape[1]
        self.images = []
        self.filters = filters

        for weight, filter in self.filters.items():
            self.apply_conv(weight, filter)

    def apply_conv(self, weight, filter):
        transformed_image = np.copy(self.input_array)
        for x in range(1, self.size_x - 1):
            for y in range(1, self.size_y - 1):
                convolution = np.sum(filter * self.input_array[x - 1:x + 2, y - 1:y + 2])
                convolution *= weight

                if convolution < 0:
                    convolution = 0
                if convolution > 255:
                    convolution = 255

                transformed_image[x, y] = convolution

        self.images.append(self.activate(transformed_image))

    def activate(self, transformed_image):
        pass

    # def apply_conv(self, weight, filter):
    #     transformed_image = np.copy(self.input_array)
    #     for x in range(1, self.size_x - 1):
    #         for y in range(1, self.size_y - 1):
    #             convolution = scipy.signal.convolve2d(filter, self.input_array[x - 1:x + 2, y - 1:y + 2], mode='valid')
    #             convolution *= weight
    #             convolution = np.clip(convolution, 0, 255)
    #             transformed_image[x, y] = convolution
    #
    #     self.images.append(transformed_image)


