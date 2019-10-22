import numba
import numpy as np

from ..base.functions import function_1 as fun1
from ..base.functions import function_2 as fun2
from .fast_ssc_decoder import FastSSCDecoder, FastSSCNode

# LLR = 100 is high enough to be considered as +∞ for RC-SCAN decoding
INFINITY = 100


class RCSCANNode(FastSSCNode):

    def compute_leaf_beta(self):
        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

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

    def compute_intermediate_beta(self, node):
        """Compute intermediate BIT values."""
        if node.is_left:
            return

        if node.is_root:
            return

        parent = node.parent
        left = node.siblings[0]
        parent.beta = self.compute_parent_beta(left.beta, node.beta, parent.alpha)  # noqa
        return self.compute_intermediate_beta(parent)

    @staticmethod
    def compute_left_alpha(llr):
        """Compute LLR for left node."""
        N = llr.size // 2
        left_parent_alpha = llr[:N]
        right_parent_alpha = llr[N:]

        left_alpha = np.zeros(N)
        for i in range(N):
            left_alpha[i] = fun1(left_parent_alpha[i], right_parent_alpha[i], 0)  # noqa
        return left_alpha

    @staticmethod
    def compute_right_alpha(parent_alpha, left_beta):
        """Compute LLR for right node."""
        N = parent_alpha.size // 2
        left_parent_alpha = parent_alpha[:N]
        right_parent_alpha = parent_alpha[N:]

        right_alpha = np.zeros(N)
        for i in range(N):
            right_alpha[i] = fun2(left_beta[i], left_parent_alpha[i], right_parent_alpha[i])  # noqa
        return right_alpha

    @staticmethod
    def compute_parent_beta(left_beta, right_beta, parent_alpha):
        """Compute bits of a parent Node."""
        N = parent_alpha.size // 2
        left_parent_alpha = parent_alpha[:N]
        right_parent_alpha = parent_alpha[N:]

        left_parent_beta = np.zeros(N)
        for i in range(N):
            left_parent_beta[i] = fun1(left_beta[i], right_beta[i], right_parent_alpha[i])  # noqa

        right_parent_beta = np.zeros(N)
        for i in range(N):
            right_parent_beta[i] = fun2(left_beta[i], right_beta[i], left_parent_alpha[i])  # noqa

        return np.append(left_parent_beta, right_parent_beta)
