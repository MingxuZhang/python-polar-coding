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
        decoder.initialize(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder._decoding_tree.root.beta,
            np.ones(self.length, dtype=np.double) * INFINITY
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        decoder.initialize(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(
            decoder._decoding_tree.root.beta,
            np.zeros(self.length)
        )

    def test_zero_one_decoder(self):
        expected = np.array([
            0.0506,
            -0.0552,
            -1.5304,
            -2.1233,
            -2.7273,
            -8.7327,
            0.1087,
            1.6463,
        ])

        mask = np.append(
            np.zeros(self.length // 2, dtype=np.int8),
            np.ones(self.length // 2, dtype=np.int8)
        )
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        decoder.initialize(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 2)
        np.testing.assert_equal(
            decoder._decoding_tree.root.beta,
            expected
        )

    def test_complex_code(self):
        long_msg = np.array([
            2.113,  1.466,  4.842,  2.867,  2.257, -4.791, -2.735,  2.252,
        ])
        mask = np.array([0, 0, 1, 1, 0, 0, 0, 0, ], dtype=np.int8)
        sub_codes = [
            np.array([0, 0, ], dtype=np.int8),
            np.array([1, 1, ], dtype=np.int8),
            np.array([0, 0, 0, 0, ], dtype=np.int8),
        ]
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)

        # Check tree structure
        self.assertEqual(len(decoder._decoding_tree.leaves), len(sub_codes))
        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_equal(leaf._mask, sub_codes[i])

        decoder.initialize(long_msg)

        # Iteration 1
        decoder()

        expected_llr = [
            np.array([2.113, -1.466, ]),
            np.array([-0.622, 0.786, ]),
            np.array([0.144, -3.325, -0.622, 0.786, ]),
        ]

        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_almost_equal(leaf._alpha, expected_llr[i])

        expected_result_beta = np.array([
            -2.735, 2.252, 2.113, -1.466, -0.622, 3.718, 6.955, 1.401,
        ])

        # Check result
        np.testing.assert_almost_equal(
            decoder._decoding_tree.root.beta,
            expected_result_beta
        )

        # Iteration 2
        decoder()

        expected_llr = [
            np.array([-0.478, -1.466, ]),
            np.array([-1.1, -0.68, ]),
            np.array([1.635, -4.005, -3.213, 0.786, ]),
        ]

        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_almost_equal(leaf._alpha, expected_llr[i])

        expected_result_beta = np.array([
            -0.622, 0.786, -0.478, -1.466, 1.491, 2.252, 4.364, 1.401,
        ])

        # Check result
        np.testing.assert_almost_equal(
            decoder._decoding_tree.root.beta,
            expected_result_beta
        )
