from ..g_fast_ssc import GeneralizedFastSSCPolarCodec
from .decoder import ImprovedFastSSCDecoder


class ImprovedFastSSCPolarCode(GeneralizedFastSSCPolarCodec):
    """Improved Generalized Fast SSC code."""
    decoder_class = ImprovedFastSSCDecoder
