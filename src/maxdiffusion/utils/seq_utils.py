import numpy as np
from typing import Union,List
def lens_to_mask(t: Union[List[int],np.ndarray], length: int) -> np.ndarray:
    # t: array of lengths, shape (b,)
    # length: maximum sequence length
    # returns: mask of shape (b, length)
    if t is not None and isinstance(t, List):
        t = np.array(t)

    if t.ndim == 0: # Handle single length input
        t = t.reshape(1)
    seq = np.arange(length)
    mask = seq < t[:, None]  # Shape: (b, length)
    return mask