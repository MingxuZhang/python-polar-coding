import numpy as np

from ..base import functions
from .g_fast_ssc_decoder import GeneralizedFastSSCNode, GeneralizedFastSSCDecoder


def splits(start, end):
    while start <= end:
        yield start
        start *= 2


def compute_ig_repetition(llr, mask_steps, N):
    """Compute bits for Improved Generalized Repetition node."""
    step = N // mask_steps  # step is equal to a chunk size
    result = np.zeros(N)

    for i in range(step):
        alpha = np.zeros(mask_steps)
        for j in range(mask_steps):
            alpha[j] = llr[i + j * step]

        if i == step - 1:
            beta = functions.make_hard_decision(alpha)
        else:
            beta = functions.compute_repetition(alpha)

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

        beta = functions.compute_single_parity_check(alpha)
        result[i:N:step] = beta

    return result


class ImprovedFastSSCNode(GeneralizedFastSSCNode):
    """Decoder for Improved Fast SSC code.

    Custom solution

    """
    IG_REPETITION = 'IG-REPETITION'
    IRG_PARITY = 'IRG-PARITY'

    MIN_CHUNKS = 2

    def get_node_type(self):
        ntype = super().get_node_type()
        if ntype != self.OTHER:
            return ntype
        if self._check_is_ig_repetition(self._mask):
            return self.IG_REPETITION
        if self._check_is_irg_parity(self._mask):
            return self.IRG_PARITY
        return self.OTHER

    def _build_decoding_tree(self):
        """Build Generalized Fast SSC decoding tree."""
        if self.is_simplified_node:
            return

        if self._mask.size == self.M:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        cls = self.__class__
        cls(mask=left_mask, name=self.LEFT, N_min=self.M, parent=self,
            AF=self.AF)
        cls(mask=right_mask, name=self.RIGHT, N_min=self.M, parent=self,
            AF=self.AF)

    def _check_is_ig_repetition(self, mask):
        """Check the node is Improved Generalized Repetition node."""
        # 1. Split mask into T chunks, T in range [2, 4, ..., N/2]
        for t in splits(self.MIN_CHUNKS, self.N // 2):
            chunks = np.split(mask, t)

            last = chunks[-1]
            last_ok = (
                (self._check_is_spc(last) and last.size >= self.REPETITION_MIN_SIZE)
                or self._check_is_one(last)
            )
            if not last_ok:
                continue

            zeros = 0
            reps = 0
            for c in chunks[:-1]:
                if self._check_is_zero(c):
                    zeros += 1
                elif c.size >= self.REPETITION_MIN_SIZE and self._check_is_repetition(c):
                    reps += 1

            others_ok = (zeros + reps + 1) == t and zeros <= self.AF
            if not others_ok:
                continue

            self.last_chunk_type = 1 if self._check_is_one(last) else 0
            self.mask_steps = t
            return True

        return False

    def _check_is_irg_parity(self, mask):
        """Check the node is Improved Relaxed Generalized Parity Check node."""
        # 1. Split mask into T chunks, T in range [2, 4, ..., N/2]
        for t in splits(self.MIN_CHUNKS, self.N // 2):
            chunks = np.split(mask, t)

            first = chunks[0]
            first_ok = self._check_is_zero(first) or self._check_is_repetition(first)
            if not first_ok:
                continue

            ones = 0
            spcs = 0
            for c in chunks[1:]:
                if self._check_is_one(c):
                    ones += 1
                elif c.size >= self.SPC_MIN_SIZE and self._check_is_spc(c):
                    spcs += 1

            others_ok = (ones + spcs + 1) == t and spcs <= self.AF
            if not others_ok:
                continue

            self.mask_steps = t
            return True

        return False

    def compute_leaf_beta(self):
        super().compute_leaf_beta()
        klass = self.__class__

        if self._node_type == klass.IG_REPETITION:
            self._beta = functions.compute_g_repetition(
                llr=self.alpha,
                mask_steps=self.mask_steps,
                last_chunk_type=self.last_chunk_type,
                N=self.N,
            )
        if self._node_type == klass.IRG_PARITY:
            self._beta = functions.compute_rg_parity(
                llr=self.alpha,
                mask_steps=self.mask_steps,
                N=self.N,
            )

    @property
    def is_ig_repetition(self):
        return self._node_type == self.IG_REPETITION

    @property
    def is_irg_parity(self):
        return self._node_type == self.IRG_PARITY


class ImprovedFastSSCDecoder(GeneralizedFastSSCDecoder):
    node_class = ImprovedFastSSCNode

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 code_min_size: int = 0,
                 AF: int = 1):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self._decoding_tree = self.node_class(mask=self.mask,
                                              N_min=code_min_size,
                                              AF=AF)
        self._position = 0
