import numpy as np
import threading
import pickle

from neural_network.layer import Layer
from neural_network import costs
from neural_network.data import create_batches


class NeuralNetwork:
    def __init__(self, layers) -> None:
        self.layers: list[Layer] = np.array(layers)

    def __repr__(self):
        return f"NeuralNetwork(\n layers={self.layers}\n)"

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    # Gradient Descent
    def cost(self, inputs, expected_outputs, node_cost: costs.Cost = costs.MeanSquaredError):
        outputs = self.predict(inputs) 
        cost = np.sum(node_cost.func(outputs, expected_outputs))
        return cost

    def reset_gradients(self):
        for layer in self.layers:
            layer.reset_gradients()

    def apply_gradients(self, learn_rate: float):
        for layer in self.layers:
            layer.apply_gradients(learn_rate)

    def update_gradients(self, inputs, expected_outputs, cost: costs.Cost):
        self.predict(inputs)

        output_layer = self.layers[-1]
        node_values = output_layer.caculate_output_layer_node_values(expected_outputs, cost)
        output_layer.update_gradients(node_values)

        for i in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.caculate_hidden_layer_node_values(self.layers[i+1], node_values)
            hidden_layer.update_gradients(node_values)

    def learn(self, train_batch, learn_rate: float, cost: costs.Cost = costs.MeanSquaredError):
        """
        Update the gradients and apply them to the network based on a batch of training data.

        :param training_batch: A list of tuples containing input and target output data. Each tuple represents a single training example with the first element being the input data (features) and the second element being the target output data (label). For example: [(x1, y1), (x2, y2), (x3, y3)] where x1, x2, x3 are the input data and y1, y2, y3 are the target output data.
        :param learn_rate: The learning rate to use when applying the gradients.
        :param cost: The cost function to use when calculating the error between the network's output and the target output.
        """
        
        threads = []
        for data in train_batch:
            thread = threading.Thread(target=self.update_gradients, args=(data[0], data[1], cost))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        self.apply_gradients(learn_rate / len(train_batch))
        self.reset_gradients()

    def train(self, train_data, test_data, learn_rate: float, cost: costs.Cost = costs.MeanSquaredError, batch_size: int = 32, epochs: int = 5, save: bool = False):
        """
        Update the gradients and apply them to the network based on a list of batches of training data.

        :param data: A list of batches of training data. Each batch is a list of tuples containing input and target output data. Each tuple represents a single training example with the first element being the input data (features) and the second element being the target output data (label). For example: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]] where x1, x2, x3, x4 are the input data and y1, y2, y3, y4 are the target output data.
        :param learn_rate: The learning rate to use when applying the gradients.
        :param cost: The cost function to use when calculating the error between the network's output and the target output.
        :param save: Whether to save the model to a file after each epoch.
        """
        
        if save: 
            file = open('neural_network.pkl', 'wb')
        num_batches = len(train_data) // batch_size
        num = (num_batches // 50)
        print()
        print("Training ...")
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            print("Progress: [", end="", flush=True)
            epoch_cost = 0
            for i, batch in enumerate(create_batches(train_data, batch_size)):
                if i % num == 0:
                    print("\u2588", end="", flush=True)
                self.learn(batch, learn_rate)
                epoch_cost += self.cost(batch[0][0], batch[0][1], cost)
            epoch_cost /= num_batches
            accuracy = self.validate(test_data)
            print(f"] Cost: {epoch_cost:.4f} | Accuracy: {accuracy:.4f}")
            if save: self.save(file)
        if save: file.close()

    def save(self, file):
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        obj = pickle.load(file)
        if isinstance(obj, NeuralNetwork):
            return obj
        else:
            raise TypeError(f"Loaded object is not an instance of NeuralNetwork. Got {type(obj)} instead.")

    def validate(self, data):
        """
        Calculate the accuracy of the model on the data.

        :param data: A list of tuples containing input and target output data. Each tuple represents a single example with the first element being the input data (features) and the second element being the target output data (label).
        :return: The accuracy of the model on the data.
        """
        correct_predictions = 0
        for input, expected_output in data:
            prediction = self.predict(input)
            if np.argmax(prediction) == np.argmax(expected_output):
                correct_predictions += 1
        accuracy = correct_predictions / len(data)
        return accuracy