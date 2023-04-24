from .layer import Layer

class Flatten(Layer):
    def calculate_outputs(self, inputs):
        return inputs.flatten()
