import numba
import numpy as np
from anytree import PreOrderIter

from ..base.functions import function_1, function_2, make_hard_decision
from .fast_ssc_decoder import FastSSCDecoder, FastSSCNode

# LLR = 1000 is high enough to be considered as +∞ for RC-SCAN decoding
INFINITY = 1000


class RCSCANNode(FastSSCNode):

    def compute_leaf_beta(self):
        """Do nothing.

        Unlike SC-based decoders SCAN decoders does not make decisions
        in leaves.

        """

    def initialize_leaf_beta(self):
        """Initialize BETA values on tree building.

        Initialize Leaves following to Section III doi:10.1109/jsac.2014.140515

        """
        if not self.is_leaf:
            return

        if self._node_type == RCSCANNode.ZERO_NODE:
            self._beta = self._compute_zero_node_beta(self.alpha)
        if self._node_type == RCSCANNode.ONE_NODE:
            self._beta = self._compute_one_node_beta(self.alpha)

    def _get_node_type(self):
        """Get the type of RC SCAN Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];

        Or other type.

        """
        if np.all(self._mask == 0):
            return RCSCANNode.ZERO_NODE
        if np.all(self._mask == 1):
            return RCSCANNode.ONE_NODE
        return RCSCANNode.OTHER

    @staticmethod
    @numba.njit
    def _compute_zero_node_beta(llr):
        """Compute beta values for ZERO node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.ones(llr.size, dtype=np.double) * INFINITY

    @staticmethod
    @numba.njit
    def _compute_one_node_beta(llr):
        """Compute beta values for ONE node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.zeros(llr.size, dtype=np.double)


class RCSCANDecoder(FastSSCDecoder):
    """Implements Reduced-complexity SCAN decoding algorithm.

    Based on:
        * https://arxiv.org/pdf/1510.06495.pdf
        * doi:10.1007/s12243-018-0634-7

    """
    node_class = RCSCANNode

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 code_min_size: int = 0,
                 I: int = 1):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic,
                         code_min_size=code_min_size)
        self.I = I

    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self.clean_before_decoding()

        for leaf in self._decoding_tree.leaves:
            leaf.initialize_leaf_beta()

        for _ in range(self.I):
            super().decode_internal(received_llr)

        return self.result

    def clean_before_decoding(self):
        """Reset intermediate BETA values.

        Run this before calling `__call__` method.

        TODO: Add JIT

        """
        for node in PreOrderIter(self._decoding_tree):
            if not node.is_leaf:
                node.beta *= 0

    def compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""
        for node in leaf.path[1:]:
            if node.is_computed or node.is_leaf:
                continue

            node.is_computed = True
            parent_alpha = node.parent.alpha

            if node.is_left:
                right_beta = node.siblings[0].beta
                node.alpha = self.compute_left_alpha(parent_alpha, right_beta)

            if node.is_right:
                left_beta = node.siblings[0].beta
                node.alpha = self.compute_right_alpha(parent_alpha, left_beta)

    def compute_intermediate_beta(self, node):
        """Compute intermediate BETA values."""
        parent = node.parent
        if node.is_left or node.is_root or parent.is_root:
            return

        left = node.siblings[0]
        parent.beta = self.compute_parent_beta(left.beta, node.beta, parent.alpha)  # noqa
        return self.compute_intermediate_beta(parent)

    @property
    def result(self):
        if self.is_systematic:
            return make_hard_decision(self.root.alpha + self._compute_result_beta())

    @staticmethod
    def compute_left_alpha(parent_alpha, beta):
        """Compute LLR for left node."""
        return RCSCANDecoder.compute_alpha(parent_alpha, beta, is_left=True)

    @staticmethod
    def compute_right_alpha(parent_alpha, beta):
        """Compute LLR for right node."""
        return RCSCANDecoder.compute_alpha(parent_alpha, beta, is_left=False)

    @staticmethod
    def compute_alpha(parent_alpha, beta, is_left):
        """Compute ALPHA values for left or right node."""
        N = parent_alpha.size // 2
        left_parent_alpha = parent_alpha[:N]
        right_parent_alpha = parent_alpha[N:]

        if is_left:
            result_alpha = function_1(left_parent_alpha, right_parent_alpha, beta)
        else:
            result_alpha = function_2(left_parent_alpha, beta, right_parent_alpha)
        return result_alpha

    @staticmethod
    def compute_parent_beta(left_beta, right_beta, parent_alpha):
        """Compute bits of a parent Node."""
        N = parent_alpha.size // 2
        left_parent_alpha = parent_alpha[:N]
        right_parent_alpha = parent_alpha[N:]

        parent_beta_left = function_1(left_beta, right_beta, right_parent_alpha)
        parent_beta_right = function_2(left_beta, left_parent_alpha, right_beta)

        return np.append(parent_beta_left, parent_beta_right)

    def _compute_result_beta(self):
        """Compute result BETA values."""
        alpha = self.root.alpha
        if not self.root.children:
            return self.root.beta
        left, right = self.root.children
        return self.compute_parent_beta(left.beta, right.beta, alpha)
