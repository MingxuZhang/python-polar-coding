import numpy as np

from ..base import make_hard_decision
from ..fast_ssc import compute_repetition, compute_single_parity_check


def compute_ig_repetition(llr, mask_steps, N):
    """Compute bits for Improved Generalized Repetition node."""
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        beta = compute_repetition(alpha)
        result[i:N:step] = beta

    return result


def compute_irg_parity(llr, mask_steps, first_chunk_type, N):
    """Compute bits for Improved Relaxed Generalized Parity Check node."""
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        beta = compute_single_parity_check(alpha)
        result[i:N:step] = beta

    return result
