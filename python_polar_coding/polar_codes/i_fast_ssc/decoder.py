import numpy as np

from ..g_fast_ssc import GeneralizedFastSSCDecoder
from .node import ImprovedFastSSCNode


class ImprovedFastSSCDecoder(GeneralizedFastSSCDecoder):
    node_class = ImprovedFastSSCNode

    def __init__(
            self,
            n: int,
            mask: np.array,
            is_systematic: bool = True,
            code_min_size: int = 0,
            AF: int = 1
    ):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self._decoding_tree = self.node_class(
            mask=self.mask,
            N_min=code_min_size,
            AF=AF,
        )
        self._position = 0
