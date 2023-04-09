from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x, y = np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test))
    x = x.reshape(-1, 28, 28, 1).astype('float32') / 255
    return x, y

def get_augmented_mnist_data(n):
    x, y = get_mnist_data()
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=5,
        horizontal_flip=False,
        vertical_flip=False,
        fill_mode='nearest'
    )
    x = np.repeat(x, n, axis=0)
    y = np.repeat(y, n)
    for i in range(x.shape[0]):
        x[i] = datagen.random_transform(x[i])
    return x, y

def train_test_split(x, y, test_size=0.2, shuffle=True):
    if shuffle:
        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)
        x = x[indices]
        y = y[indices]
    split_index = int((1 - test_size) * x.shape[0])
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return x_train, x_test, y_train, y_test

def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]

if __name__ == "__main__":
    print(len(get_mnist_data()[0]))
    print(len(get_augmented_mnist_data(2)[0]))
    x_train, x_test, y_train, y_test = train_test_split(*get_augmented_mnist_data(2))
    print(len(x_train))