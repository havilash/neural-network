from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = [*x_train, *x_test]
    y = [*y_train, *y_test]
    x = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x = x_train.astype('float32')
    x /= 255
    return x, y

def get_augmented_mnist_data(n):
    x, y = get_mnist_data()
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1)
    datagen.fit(x)
    augmented_features = []
    augmented_labels = []
    for x_batch, y_batch in datagen.flow(x, y, batch_size=n):
        augmented_features.append(x_batch)
        augmented_labels.append(y_batch)
        if len(augmented_features) == n:
            break
    return np.array(augmented_features), np.array(augmented_labels)

def get_train_test_data(data, train_size=0.7, shuffle=False):
    x, y = data
    if shuffle:
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
    split_index = int(train_size * len(X_train))
    return (X_train[:split_index], y_train[:split_index]), (X_train[split_index:], y_train[split_index:]), (X_test, y_test)


if __name__ == "__main__":
    print(get_mnist_data()[0][0])