import numpy as np
def lens_to_mask(t: np.ndarray, length: int) -> np.ndarray:
    # t: array of lengths, shape (b,)
    # length: maximum sequence length
    # returns: mask of shape (b, length)
    if t.ndim == 0: # Handle single length input
        t = t.reshape(1)
    seq = np.arange(length)
    mask = seq < t[:, None]  # Shape: (b, length)
    return mask