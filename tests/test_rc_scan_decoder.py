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
