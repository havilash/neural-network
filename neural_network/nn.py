import numpy as np
import math
import threading
from multiprocessing import Pool
import pickle
import time

from neural_network.layers import Layer, Dense
from neural_network import costs
from neural_network.data import create_batches
from neural_network import constants

def thread_error_handler(stop_event, error_message):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                func(*args, **kwargs)
            except Exception as e:
                error_message.append(str(e))
                stop_event.set()
        return wrapper
    return decorator

class NeuralNetwork:
    def __init__(self, layers) -> None:
        self.layers: list[Layer] = np.array(layers)

        for i in range(len(layers)-1):
            if isinstance(layers[i], Dense):
                if layers[i].num_output != layers[i+1].num_input:
                    raise ValueError(f"Input Shape of Layer {i+1} doesn't match the input Shape of Layer {i+2}")

    def __repr__(self):
        return f"NeuralNetwork(\n layers={self.layers}\n)"
    
    def sequential(self, layers: list[Layer | Dense]):
        self.layers = []
        for i in range(1, len(layers)):
            layers[i].set_shape(layers[i-1].output_shape, layers[i].output_shape)

    def predict(self, inputs):
        for layer in self.layers:
            inputs = layer.calculate_outputs(inputs)
        return inputs
    
    def cost(self, inputs, expected_outputs, node_cost: costs.Cost = constants.DEFAULT_COST):
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
        node_values = output_layer.calculate_output_layer_node_values(expected_outputs, cost)
        output_layer.update_gradients(node_values)

        for i in range(len(self.layers) - 2, -1, -1):
            hidden_layer = self.layers[i]
            node_values = hidden_layer.calculate_node_values(self.layers[i + 1], node_values)
            hidden_layer.update_gradients(node_values)

    def learn(self, train_batch, learn_rate: float, cost: costs.Cost = constants.DEFAULT_COST):
        for data in train_batch:
            self.update_gradients(data[0], data[1], cost)
        self.apply_gradients(learn_rate / len(train_batch))
        self.reset_gradients()

    def learn_threading(self, train_batch, learn_rate: float, cost: costs.Cost = constants.DEFAULT_COST):
        threads = []
        stop_event = threading.Event()
        error_message = []
        update_gradients = thread_error_handler(stop_event, error_message)(self.update_gradients)
        for data in train_batch:
            # self.update_gradients(data[0], data[1], cost)
            thread = threading.Thread(target=update_gradients, args=(data[0], data[1], cost))
            thread.start()
            threads.append(thread)
        
        for thread in threads:
            thread.join()

        if stop_event.is_set():
            print('error_message:', error_message)
            raise Exception(error_message[0])
        else:
            self.apply_gradients(learn_rate / len(train_batch))
            self.reset_gradients()

    def learn_multiprocessing(self, train_batch, learn_rate: float, cost: costs.Cost = constants.DEFAULT_COST):
        with Pool() as pool:
            results = pool.starmap(self.update_gradients, [(data[0], data[1], cost) for data in train_batch])
        self.apply_gradients(learn_rate / len(train_batch))
        self.reset_gradients()

    def train(
            self, 
            train_data, 
            test_data, 
            learn_rate: float, 
            cost: costs.Cost = constants.DEFAULT_COST, 
            batch_size: int = 32, 
            epochs: int = 5, 
            save: bool = False, 
            file_name: str = "neural_network.pkl", 
            validate_per_batch: bool = False, 
            learn_method: str = 'threading', 
        ):
        """
        Update the gradients and apply them to the network based on a list of batches of training data.

        :param train_data: A list of batches of training data. Each batch is a list of tuples containing input and target output data. Each tuple represents a single training example with the first element being the input data (features) and the second element being the target output data (label). For example: [[(x1, y1), (x2, y2)], [(x3, y3), (x4, y4)]] where x1, x2, x3, x4 are the input data and y1, y2, y3, y4 are the target output data.
        :param test_data: A list of tuples containing input and target output data for testing the accuracy of the model.
        :param learn_rate: The learning rate to use when applying the gradients.
        :param cost: The cost function to use when calculating the error between the network's output and the target output.
        :param batch_size: The size of each batch of training data.
        :param epochs: The number of times to iterate over the entire training dataset.
        :param save: Whether to save the model to a file after each epoch.
        :param file_name: The name of the file to save the model to if `save` is `True`.
        :param validate_per_batch: Whether to validate the model after each batch of training data.
        :param learn_method: The method to use for learning. Can be 'normal', 'threading', or 'multiprocessing'.
        :return: A tuple containing the list of accuracies and the list of costs.
        """

        # Set learn function
        if learn_method == 'normal':
            learn = self.learn
        elif learn_method == 'threading':
            learn = self.learn_threading
        elif learn_method == 'multiprocessing':
            learn = self.learn_multiprocessing
        else:
            raise ValueError(f"Invalid learn_method: {learn_method}")

        # Open file
        if save:
            file = open(file_name, 'wb')

        # Initialize accuracies, costs
        accuracies, epoch_costs = [], []

        # number of batches, step size progress bar
        num_batches = math.ceil(len(train_data) / batch_size)
        bar_step = (num_batches / 50)

        print()
        print("Training ...")
        for epoch in range(epochs):
            start_epoch = time.time()
            print(f"Epoch {epoch+1}/{epochs}")
            print("- Progress: [", end="", flush=True)

            # Loop over batches
            for i, batch in enumerate(create_batches(train_data, batch_size)):
                # Update progress bar
                if i % bar_step < 1:
                    print("\u2588" * int(max(1 / bar_step, 1)), end="", flush=True)

                # Update gradients using learn function
                learn(batch, learn_rate)

                # Validate after each batch
                if validate_per_batch:
                    acc, cos = self.validate(test_data, cost)
                    accuracies.append(acc)
                    epoch_costs.append(cos)

            # Validate after each epoch
            if not validate_per_batch:
                acc, cos = self.validate(test_data, cost)
                accuracies.append(acc)
                epoch_costs.append(cos)

            delta_time = time.time() - start_epoch
            print(f"] Cost: {cos:.4f} | Accuracy: {acc:.4f}, | Time: {delta_time:.2f}s")

            # Save model
            if save:
                self.save(file)

        # Close file if save is True
        if save:
            file.close()

        return accuracies, epoch_costs

    def save(self, file):
        pickle.dump(self, file)

    @staticmethod
    def load(file):
        try:
            obj = pickle.load(file)
        except EOFError:
            raise RuntimeError("Failed to load object from file: file is either empty or corrupted")
            
        if isinstance(obj, NeuralNetwork):
            return obj
        else:
            raise TypeError(f"Loaded object is not an instance of NeuralNetwork. Got {type(obj)} instead.")

    def validate(self, data, cost: costs.Cost = constants.DEFAULT_COST):
        """
        Calculate the accuracy and average cost of the model on the data.

        :param data: A list of tuples containing input and target output data. Each tuple represents a single example with the first element being the input data (features) and the second element being the target output data (label).
        :param cost: The cost function to use when calculating the average cost of the model on the data.
        :return: A tuple containing the accuracy and average cost of the model on the data.
        """

        correct_predictions = 0
        average_cost = 0
        for inputs, expected_outputs in data:
            prediction = self.predict(inputs)
            if np.argmax(prediction) == np.argmax(expected_outputs):
                correct_predictions += 1
            average_cost += np.sum(cost.func(prediction, expected_outputs))
        accuracy = correct_predictions / len(data)
        average_cost /= len(data)
        return accuracy, average_cost