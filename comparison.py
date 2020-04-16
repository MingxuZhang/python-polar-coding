from functools import partial

import numpy as np

from python_polar_coding.channels import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.base import make_hard_decision
from python_polar_coding.polar_codes.fast_ssc import (
    FastSSCPolarCodec,
    compute_repetition,
)
from python_polar_coding.polar_codes.g_fast_ssc import compute_g_repetition
from python_polar_coding.simulation.functions import generate_binary_message


def compute_ig_repetition1(llr, mask_steps, N):
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


def compare(code, verified, to_verify, messages=100):
    channel = SimpleBPSKModulationAWGN(code.K/code.N)
    results = 0

    for m in range(messages):
        msg = generate_binary_message(code.K)
        encoded = code.encode(msg)
        transmitted = channel.transmit(encoded, 10)
        one = verified(transmitted, N=code.N)
        two = to_verify(transmitted, N=code.N)
        print(one)
        print(two)

        results += int(np.all(one == two))

    return results


mask = '0000000000000111'
code = FastSSCPolarCodec(N=16, K=3, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=0)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


mask = '0000000000001111'
code = FastSSCPolarCodec(N=16, K=4, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=1)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


def compute_ig_repetition2(llr, mask_steps, N):
    """Compute bits for Improved Generalized Repetition node."""
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        if i == step - 1:
            beta = make_hard_decision(alpha)
        else:
            beta = compute_repetition(alpha)

        result[i:N:step] = beta

    return result


mask = '0000000000010111'
code = FastSSCPolarCodec(N=16, K=4, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=0)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


mask = '0000000100010111'
code = FastSSCPolarCodec(N=16, K=5, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=0)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


mask = '0000000100011111'
code = FastSSCPolarCodec(N=16, K=6, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=0)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


mask = '0001000100010111'
code = FastSSCPolarCodec(N=16, K=6, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=1)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)


mask = '0001000100011111'
code = FastSSCPolarCodec(N=16, K=7, mask=mask)
verified = partial(compute_g_repetition, mask_steps=4, last_chunk_type=1)
to_verify = partial(compute_ig_repetition1, mask_steps=4)
compare(code, verified, to_verify)
