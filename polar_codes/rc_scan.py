from .base import BasicPolarCode
from .decoders.rc_scan_decoder import RCSCANDecoder
from .base.functions import make_hard_decision


class RCSCANPolarCode(BasicPolarCode):
    """Polar code with RC-SCAN decoding algorithm."""

    def __init__(self, iterations=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._iterations = iterations
        self.decoder = RCSCANDecoder(
            mask=self.polar_mask,
            is_systematic=self.is_systematic
        )

    def decode(self, received_message):
        """Decode Polar code with Fast SSC decoding algorithm."""
        return self._rc_scan_decode(received_message)

    def _rc_scan_decode(self, llr_estimated_message):
        """RC SCAN decoding."""
        llr_message = llr_estimated_message

        for i in range(self._iterations):
            self.decoder.initialize(llr_message)
            self.decoder()
            llr_message = self.decoder.result

            if not self.is_crc_aided:
                continue

            # Validate the result using CRC
            hard_decision = self._extract(make_hard_decision(llr_message))
            if self._check_crc(hard_decision):
                return hard_decision

        return self._extract(make_hard_decision(llr_message))
