import numpy as np


# NOTE: this would be faster with `gensim`
#       since it uses BLAS, but can't get it to
#       work well with numpy2/scipy combo
def unitvec(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)
