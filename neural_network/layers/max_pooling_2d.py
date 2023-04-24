import numpy as np
import random
import math

from neural_network import activations, costs

class MaxPooling2D:
    def __init__(self, input_array: np.ndarray) -> []:
        self.input_array = input_array
        self.size_x = input_array.shape[0]
        self.size_y = input_array.shape[1]
        self.images = []
        self.new_x = int(self.size_x / 2)
        self.new_y = int(self.size_y / 2)

        self.apply_conv()

    def apply_conv(self, weight, filter):
        filter_size_x, filter_size_y = filter.shape
        pad_x = (filter_size_x - 1) // 2
        pad_y = (filter_size_y - 1) // 2
        padded_input = np.pad(self.input_array, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
        transformed_image = np.zeros_like(self.input_array)
        for x in range(self.size_x):
            for y in range(self.size_y):
                convolution = np.sum(padded_input[x:x + filter_size_x, y:y + filter_size_y] * filter) * weight
                convolution = np.clip(convolution, 0, 255)
                transformed_image[x, y] = convolution
        self.images.append(transformed_image)

    #
    # def apply_conv(self):
    #     newImage = np.empty((self.new_x, self.new_y))
    #
    #     for x in range(0, self.size_x, 2):
    #         for y in range(0, self.size_y, 2):
    #             pixels = np.array([self.input_array[x:x+1, y:y+1]])
    #             newImage[int(x / 2), int(y / 2)] = max(pixels)
    #
    #     self.images.append(newImage)