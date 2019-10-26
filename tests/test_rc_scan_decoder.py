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
            decoder.result,
            np.ones(self.length, dtype=np.double) * INFINITY
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        decoder.initialize(self.received_llr)
        decoder()

        self.assertEqual(len(decoder._decoding_tree.leaves), 1)
        np.testing.assert_equal(decoder.result, np.zeros(self.length))

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
        np.testing.assert_equal(decoder.result, expected)

    def test_complex_code(self):
        long_msg = np.array([
             0.1139, 1.4662,  2.8427,  0.8675,  1.2576, -1.1791, 0.7535,  2.2528,
            -0.3653, 0.6884, -0.9574, -0.2793, -0.8862, -1.7831, 1.7425, -3.0953,
        ])
        mask = np.array(
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, ], dtype=np.int8)
        sub_codes = [
            np.array([1, 1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, 0, 0, 0, ], dtype=np.int8),
            np.array([1, ], dtype=np.int8),
            np.array([0, ], dtype=np.int8),
            np.array([1, 1, ], dtype=np.int8),
            np.array([1, 1, 1, 1, ], dtype=np.int8),
        ]
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)

        # Check tree structure
        self.assertEqual(len(decoder._decoding_tree.leaves), len(sub_codes))
        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_equal(leaf._mask, sub_codes[i])

        decoder.initialize(long_msg)
        decoder()

        # TODO: compute expected LLR
        expected_llr = [
            np.array([-0.1139, 0.2793, ]),
            np.array([-0.2793, ]),
            np.array([-0.4744, ]),
            np.array([-0.8862, 1.1791, 0.6400, -2.5321, ]),
            np.array([-0.4792, ]),
            np.array([-1.0378, ]),
            np.array([1.9148, 0.0794, ]),
            np.array([-1.7724, -2.483, 2.9752, -4.7895, ]),
        ]

        for i, leaf in enumerate(decoder._decoding_tree.leaves):
            np.testing.assert_almost_equal(leaf._llr, expected_llr[i])

        expected_result = np.array([
            0,       0,      -0.1139, 0.66864,
            0.1139, -0.4093, -0.8435,  1.5644,
            0.1139,  1.4662,  2.9566, 1.5539,
            1.3715, -0.4907, -0.0900, 2.8114,
        ])

        # Check result
        np.testing.assert_equal(decoder.result, expected_result)
