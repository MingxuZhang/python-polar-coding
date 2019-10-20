from unittest import TestCase

import numpy as np

from polar_codes.decoders import RCSCANDecoder
from polar_codes.decoders.rc_scan_decoder import RCSCANNode


class TestFastSSCDecoder(TestCase):
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
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.initialize(self.received_llr)
        decoder()
        np.testing.assert_equal(
            decoder.result,
            np.ones(self.length, dtype=np.double) * RCSCANNode.INFINITY
        )

    def test_one_node_decoder(self):
        mask = np.ones(self.length, dtype=np.int8)
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        self.assertEqual(len(decoder._decoding_tree.leaves), 1)

        decoder.initialize(self.received_llr)
        decoder()
        np.testing.assert_equal(
            decoder.result,
            np.zeros(self.length)
        )

    def test_zero_one_decoder(self):
        mask = np.append(
            np.zeros(self.length // 2, dtype=np.int8),
            np.ones(self.length // 2, dtype=np.int8)
        )
        decoder = RCSCANDecoder(mask=mask, is_systematic=True)
        self.assertEqual(len(decoder._decoding_tree.leaves), 2)

        decoder.initialize(self.received_llr)
        decoder()
        np.testing.assert_equal(
            decoder.result,
            np.zeros(self.length)
        )
