import numpy as np
from sklearn.datasets import fetch_openml
import albumentations as alb
import random


def get_mnist_data(limit=None):
    mnist = fetch_openml('mnist_784', version=1, parser='auto')
    x, y = np.array(mnist['data']), np.array(mnist['target'])
    x = x.astype('float32') / 255
    if limit is not None:
        x, y = x[:limit], y[:limit]
    return x, y


def get_augmented_mnist_data(n, limit=None):
    x, y = get_mnist_data(limit=limit)
    x = x.reshape(-1, 28, 28)
    augmented_x = []
    transform = alb.Compose([
        alb.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=12),
        alb.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
        alb.GridDistortion(),
        alb.OpticalDistortion(distort_limit=0.03, shift_limit=0.03)
    ])
    for i in range(x.shape[0]):
        for _ in range(n):
            augmented_x.append(transform(image=x[i])['image'])
    return np.array(augmented_x), np.repeat(y, n)


'''
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
'''


def create_batches(data, batch_size: int = 32):
    for i in range(0, len(data), batch_size):
        yield data[i:i+batch_size]


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import random

    n = 100
    x, y = get_mnist_data(n)
    x_augmented, y_augmented = get_augmented_mnist_data(6, n)

    for _ in range(10):
        random_index = random.randint(0, len(x) - 1)
        base_image = x[random_index]
        augmented_images = x_augmented[random_index*6:(random_index+1)*6]

        fig = plt.figure()
        gs = fig.add_gridspec(3, 3)
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = [fig.add_subplot(gs[i // 3 + 1, i % 3]) for i in range(6)]

        ax1.imshow(base_image.reshape(28, 28), cmap='gray')
        ax1.set_title('Base Image')
        for i in range(6):
            ax2[i].imshow(augmented_images[i].reshape(28, 28), cmap='gray')
        plt.show()