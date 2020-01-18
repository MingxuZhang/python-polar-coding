import numba
import numpy as np
from anytree import Node, PreOrderIter

from ..base.functions import make_hard_decision
from .sc_decoder import SCDecoder


class FastSSCNode(Node):

    LEFT = 'left_child'
    RIGHT = 'right_child'
    ROOT = 'root'
    NODE_NAMES = (LEFT, RIGHT, ROOT)

    ZERO_NODE = 'ZERO'
    ONE_NODE = 'ONE'
    SINGLE_PARITY_CHECK = 'SINGLE_PARITY_CHECK'
    REPETITION = 'REPETITION'
    OTHER = 'OTHER'

    # Minimal size of Single parity check node
    SPC_MIN_SIZE = 4
    # Minimal size of Repetition Fast SSC Node
    REPETITION_MIN_SIZE = 2

    def __init__(self, mask, name=ROOT, code_min_size=None, **kwargs):
        """A node of Fast SSC decoder."""
        if name not in self.__class__.NODE_NAMES:
            raise ValueError('Wrong Fast SSC Node type')

        super().__init__(name, **kwargs)

        self._mask = mask
        self._code_min_size = code_min_size
        self._node_type = self._get_node_type()
        self._alpha = np.zeros(self.N, dtype=np.double)
        self._beta = np.zeros(self.N, dtype=np.int8)

        self.is_computed = False
        self._build_decoding_tree()

    @property
    def N(self):
        return self._mask.size

    @property
    def M(self):
        """Minimal size of component polar code."""
        return self._code_min_size

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
        return self.name == self.__class__.LEFT

    @property
    def is_right(self):
        return self.name == self.__class__.RIGHT

    @property
    def is_simplified_node(self):
        return self._node_type != self.__class__.OTHER

    @property
    def zero_min_size(self):
        return 1 if self.M is None else self.M

    @property
    def one_min_size(self):
        return self.zero_min_size

    @property
    def repetition_min_size(self):
        return self.__class__.REPETITION_MIN_SIZE if self.M is None else self.M

    @property
    def spc_min_size(self):
        return self.__class__.SPC_MIN_SIZE if self.M is None else self.M

    def to_dict(self):
        return {
            'type': self._node_type,
            'mask': self._mask,
        }

    def compute_leaf_beta(self):
        if not self.is_leaf:
            raise TypeError('Cannot make decision in not a leaf node.')

        if self._node_type == FastSSCNode.ZERO_NODE:
            self._beta = np.zeros(self.N, dtype=np.int8)
        if self._node_type == FastSSCNode.ONE_NODE:
            self._beta = make_hard_decision(self.alpha)
        if self._node_type == FastSSCNode.SINGLE_PARITY_CHECK:
            self._beta = self._compute_bits_spc(self.alpha)
        if self._node_type == FastSSCNode.REPETITION:
            self._beta = self._compute_bits_repetition(self.alpha)

    def _initialize_beta(self):
        """Initialize BETA values on tree building."""
        return np.zeros(self.N, dtype=np.int8)

    @staticmethod
    @numba.njit
    def _compute_bits_spc(llr):
        bits = np.array([l < 0 for l in llr], dtype=np.int8)
        parity = np.sum(bits) % 2
        arg_min = np.abs(llr).argmin()
        bits[arg_min] = (bits[arg_min] + parity) % 2
        return bits

    @staticmethod
    @numba.njit
    def _compute_bits_repetition(llr):
        return (
            np.zeros(llr.size, dtype=np.int8) if np.sum(llr) >= 0
            else np.ones(llr.size, dtype=np.int8)
        )

    def _get_node_type(self):
        """Get the type of Fast SSC Node.

        * Zero node - [0, 0, 0, 0, 0, 0, 0, 0];
        * One node - [1, 1, 1, 1, 1, 1, 1, 1];
        * Single parity check node - [0, 1, 1, 1, 1, 1, 1, 1];
        * Repetition node - [0, 0, 0, 0, 0, 0, 0, 1].

        Or other type.

        """
        if np.all(self._mask == 0) and self._mask.size >= self.zero_min_size:
            return FastSSCNode.ZERO_NODE
        if np.all(self._mask == 1) and self._mask.size >= self.one_min_size:
            return FastSSCNode.ONE_NODE
        if (self._mask.size >= self.spc_min_size
                and self._mask[0] == 0 and np.sum(self._mask) == self.N - 1):
            return FastSSCNode.SINGLE_PARITY_CHECK
        if (self._mask.size >= self.repetition_min_size
                and self._mask[-1] == 1 and np.sum(self._mask) == 1):
            return FastSSCNode.REPETITION
        return FastSSCNode.OTHER

    def _build_decoding_tree(self):
        """Build Fast SSC decoding tree."""
        if self.is_simplified_node:
            return

        if self._mask.size == self.M:
            return

        left_mask, right_mask = np.split(self._mask, 2)
        self.__class__(mask=left_mask, name=self.LEFT, code_min_size=self.M,
                       parent=self)
        self.__class__(mask=right_mask, name=self.RIGHT, code_min_size=self.M,
                       parent=self)


class FastSSCDecoder(SCDecoder):
    """Implements Fast SSC decoding algorithm."""
    node_class = FastSSCNode

    def __init__(self, n: int,
                 mask: np.array,
                 is_systematic: bool = True,
                 code_min_size: int = 0):
        super().__init__(n=n, mask=mask, is_systematic=is_systematic)
        self._decoding_tree = self.node_class(
            mask=self.mask,
            code_min_size=code_min_size,
        )
        self._position = 0

    def _set_initial_state(self, received_llr):
        """Initialize decoder with received message."""
        self.current_state = np.zeros(self.n, dtype=np.int8)
        self.previous_state = np.ones(self.n, dtype=np.int8)

        # LLR values at intermediate steps
        self._position = 0
        self._decoding_tree.root.alpha = received_llr

    def decode_internal(self, received_llr: np.array) -> np.array:
        """Implementation of SC decoding method."""
        self._set_initial_state(received_llr)

        # Reset the state of the tree before decoding
        for node in PreOrderIter(self._decoding_tree):
            node.is_computed = False

        for leaf in self._decoding_tree.leaves:
            self._set_decoder_state(self._position)
            self.compute_intermediate_alpha(leaf)
            leaf.compute_leaf_beta()
            self.compute_intermediate_beta(leaf)
            self.set_next_state(leaf.N)

        return self.root.beta

    @property
    def root(self):
        """Returns root node of decoding tree."""
        return self._decoding_tree.root

    @property
    def result(self):
        if self.is_systematic:
            return self.root.beta

    @property
    def M(self):
        return self._decoding_tree.M

    def compute_intermediate_alpha(self, leaf):
        """Compute intermediate Alpha values (LLR)."""
        for node in leaf.path[1:]:
            if node.is_computed:
                continue

            parent_alpha = node.parent.alpha

            if node.is_left:
                node.alpha = self._compute_left_alpha(parent_alpha)
                continue

            left_node = node.siblings[0]
            left_beta = left_node.beta
            node.alpha = self._compute_right_alpha(parent_alpha, left_beta)
            node.is_computed = True

    def compute_intermediate_beta(self, node):
        """Compute intermediate Beta values (BIT)."""
        if node.is_left:
            return

        if node.is_root:
            return

        parent = node.parent
        left = node.siblings[0]
        parent.beta = self.compute_parent_beta(left.beta, node.beta)
        return self.compute_intermediate_beta(parent)

    def set_next_state(self, leaf_size):
        self._position += leaf_size

    @staticmethod
    def compute_parent_beta(left, right):
        """Compute Beta (BITS) of a parent Node."""
        N = left.size
        # append - njit incompatible
        return np.append((left + right) % 2, right)
