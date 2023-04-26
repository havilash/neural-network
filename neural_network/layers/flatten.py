from .layer import Layer
import numpy as np

class Flatten(Layer):
    def __init__(self) -> None:
        super().__init__()

    def calculate_outputs(self, inputs):
        return inputs.flatten()
