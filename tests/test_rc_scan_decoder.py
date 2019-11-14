from unittest import TestCase

import numpy as np

from polar_codes.decoders import RCSCANDecoder
from polar_codes.decoders.rc_scan_decoder import INFINITY


class TestRCSCANDecoder(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.received_llr = np.array([
            -2.7273,
            -8.7327,
            0.1087,
            1.6463,
            0.0506,
            -0.0552,
            -1.5304,
            -2.1233,
        ])
        cls.length = cls.received_llr.size

    def test_zero_node_decoder(self):
        mask = np.zeros(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        decoder.set_initial_state(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder.root.beta,
            np.ones(self.length, dtype=np.double) * INFINITY
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        decoder.set_initial_state(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder.root.beta,
            np.zeros(self.length)
        )

    def test_complex_code(self):
        # TODO update the data with correct computations after implementing
        # correct results computation
        # And use `result` property
        llr = np.array([-2.7273, -8.7327, 0.1087, 1.6463, 0.0506, -0.0552, -1.5304, -2.1233])
        mask = np.array([0, 1, 0, 1, 0, 1, 1, 1, ], dtype=np.int8)
        sub_codes = [
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([1, 1, ], dtype=np.int8),
        ]
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)

        # Check tree structure
        self.assertEqual(len(decoder._decoding_tree.leaves), len(sub_codes))
        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_equal(leaf._mask, sub_codes[i])

        # Iteration 1
        decoder.set_initial_state(llr)
        decoder()

        # Check result
        expected_result_beta = np.array([0.046, 0.0506, 1.535, 0.0075, -0.0598, 0.046, -0.1133, -0.0796])
        np.testing.assert_almost_equal(decoder._compute_result_beta(), expected_result_beta)

        expected_result = np.array([1, 1, 0, 0, 1, 1, 1, 1, ], dtype=np.int8)
        np.testing.assert_equal(decoder.result, expected_result)

        # Iteration 2
        decoder.set_initial_state(llr)
        decoder()

        # Check result
        expected_result_beta = np.array([0.0414, 0.046, 1.1596, -0.0167, -0.069, 0.0368, -0.1179, 0.0075])
        np.testing.assert_almost_equal(decoder._compute_result_beta(), expected_result_beta)

        expected_result = np.array([1, 1, 0, 0, 1, 1, 1, 1, ], dtype=np.int8)
        np.testing.assert_equal(decoder.result, expected_result)
