import numpy as np

# Sobel filters
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# Prewitt filters
prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

# Scharr filters
scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])

# Laplacian filter
laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

edge_detection = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

ALL_FILTERS = [sobel_x, sobel_y, prewitt_x, prewitt_y, scharr_x, scharr_y, laplacian, edge_detection]
