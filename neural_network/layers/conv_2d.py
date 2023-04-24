import numpy as np
import random
import math
import filters

from neural_network import activations, costs



class Conv2D():
    def __init__(self, filters: [np.ndarray] = filters.ALL_FILTERS):
        # self.size_x = input_array.shape[0]
        # self.size_y = input_array.shape[1]
        self.filters = filters

    def apply_conv(self, filter):
        weight = 1/np.sum(filter)
        transformed_image = np.copy(self.input_array)
        for x in range(1, self.size_x - 1):
            for y in range(1, self.size_y - 1):
                convolution = np.sum(filter * self.input_array[x - 1:x + 2, y - 1:y + 2])
                convolution *= weight

                convolution = np.clip(convolution, 0, 255)


                transformed_image[x, y] = convolution

        self.images.append(self.activate(transformed_image))



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


