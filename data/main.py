import cv2
import numpy as np
import albumentations as A

DEFAULT_SQUARE = []
DEFAULT_SQUARE.extend([[0 for _ in range(20)] for _ in range(3)])
DEFAULT_SQUARE.extend([0, 0, 0] + [1 for _ in range(14)] + [0, 0, 0] for _ in range(2))
DEFAULT_SQUARE.extend([0, 0, 0, 1, 1] + [0 for _ in range(10)] + [1, 1, 0, 0, 0] for _ in range(10))
DEFAULT_SQUARE.extend([0, 0, 0] + [1 for _ in range(14)] + [0, 0, 0] for _ in range(2))
DEFAULT_SQUARE.extend([[0 for _ in range(20)] for _ in range(3)])
DEFAULT_SQUARE = np.array(DEFAULT_SQUARE, dtype="float32")


def square_gen(size=20, prob_cutout=0):
    image = cv2.resize(DEFAULT_SQUARE, (size, size), interpolation=cv2.INTER_LINEAR)

    transformed = A.Compose([
        A.Cutout(p=prob_cutout, num_holes=2, max_h_size=size//4, max_w_size=size//4)
    ])(image=image)
    image = transformed["image"]

    image[image > 0] = 1
    return image
