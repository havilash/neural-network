import numpy as np

from . import layer


def main():
    l = layer.Layer(3, 2)
    l.biases[0] = 4
    print(l.calculate_outputs([1, 1, 1]))


if __name__ == "__main__":
    main()
