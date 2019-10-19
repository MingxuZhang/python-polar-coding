import numba
import numpy as np
from anytree import Node, PreOrderIter

from .sc_decoder import SCDecoder


class RCSCANNode(Node):

    LEFT = 'left_child'
    RIGHT = 'right_child'
    ROOT = 'root'
    NODE_NAMES = (LEFT, RIGHT, ROOT)

    ZERO_NODE = 'ZERO'
    ONE_NODE = 'ONE'
    # SINGLE_PARITY_CHECK = 'SINGLE_PARITY_CHECK'
    # REPETITION = 'REPETITION'
    OTHER = 'OTHER'

    # LLR = 100 is high enough to be considered as +∞
    INFINITY = 100

    # FAST_SSC_NODE_TYPES = (ZERO_NODE, ONE_NODE, SINGLE_PARITY_CHECK, REPETITION)  # noqa
    SCAN_NODE_TYPES = (ZERO_NODE, ONE_NODE, )

    # # Minimal size of Single parity check node
    # SPC_MIN_SIZE = 4
    # # Minimal size of Repetition Fast SSC Node
    # REPETITION_MIN_SIZE = 2

    def __init__(self, mask, name=ROOT, **kwargs):
        """A node of Fast SSC decoder."""
        if name not in RCSCANNode.NODE_NAMES:
            raise ValueError('Wrong SCAN Node type')

        super().__init__(name, **kwargs)

        self._mask = mask
        self._node_type = self._get_node_type()
        self._alpha = np.zeros(self.N, dtype=np.double)
        self._beta = np.zeros(self.N, dtype=np.double)

        self.is_computed = False
        self._build_scan_tree()

    @property
    def N(self):
        return self._mask.size

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of LLR vector')
        self._alpha = np.array(value)

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        if self._mask.size != value.size:
            raise ValueError('Wrong size of Bits vector')
        self._beta = np.array(value)

    @property
    def is_left(self):
        return self.name == RCSCANNode.LEFT

    @property
    def is_right(self):
        return self.name == RCSCANNode.RIGHT

    @property
    def is_scan_node(self):
        return self._node_type in RCSCANNode.SCAN_NODE_TYPES

    def compute_beta_values(self):
        """Compute bate values."""
        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        if self._node_type == RCSCANNode.ZERO_NODE:
            self._beta = self._compute_zero_node_beta(self.alpha)
        if self._node_type == RCSCANNode.ONE_NODE:
            self._beta = self._compute_one_node_beta(self.alpha)

    def _get_node_type(self):
        """Get the type of Fast SSC Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];

        Or other type.

        """
        if np.all(self._mask == 0):
            return RCSCANNode.ZERO_NODE
        if np.all(self._mask == 1):
            return RCSCANNode.ONE_NODE
        return RCSCANNode.OTHER

    def _build_scan_tree(self):
        """Build Fast SSC tree."""
        if self.is_scan_node:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        RCSCANNode(mask=left_mask, name=RCSCANNode.LEFT, parent=self)
        RCSCANNode(mask=right_mask, name=RCSCANNode.RIGHT, parent=self)

    @staticmethod
    @numba.jit
    def _compute_zero_node_beta(llr):
        """Compute beta values for ZERO node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.ones(llr.size, dtype=np.double) * RCSCANNode.INFINITY

    @staticmethod
    @numba.jit
    def _compute_one_node_beta(llr):
        """Compute beta values for ONE node.

        https://arxiv.org/pdf/1510.06495.pdf Section III.C.

        """
        return np.zeros(llr.size, dtype=np.double)