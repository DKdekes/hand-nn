import numpy as np


def accuracy(predictions: np.array, targets: np.array) -> float:
    assert predictions.shape == targets.shape
    assert predictions.dtype == targets.dtype
    num_correct = 0
    for i in range(predictions.shape[0]):
        if np.array_equal(predictions[i], targets[i]):
            num_correct += 1
    return num_correct / predictions.shape[0]
