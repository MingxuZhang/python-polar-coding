from .decoders.i_fast_ssc_decoder import ImprovedFastSSCDecoder
from .g_fast_ssc import GeneralizedFastSSCPolarCode


class ImprovedFastSSCPolarCode(GeneralizedFastSSCPolarCode):
    """Improved Generalized Fast SSC code."""
    decoder_class = ImprovedFastSSCDecoder
