from polar_codes import RCSCANPolarCode

from .sc import VerifySystematicSCCode


class VerifyRCSCANCode(VerifySystematicSCCode):
    messages = 10000
    codec_class = RCSCANPolarCode


class TestSystematicCode_1024_512_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_1024_512_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_1024_512_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 1024,
        'info_length': 512,
        'is_systematic': True,
        'iterations': 4,
    }


class TestSystematicCode_2048_1024_iter_1(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 1,
    }


class TestSystematicCode_2048_1024_iter_2(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 2,
    }


class TestSystematicCode_2048_1024_iter_4(VerifyRCSCANCode):
    code_parameters = {
        'codeword_length': 2048,
        'info_length': 1024,
        'is_systematic': True,
        'iterations': 4,
    }
