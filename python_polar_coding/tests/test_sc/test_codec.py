from unittest import TestCase

from python_polar_coding.polar_codes.sc import SCPolarCodec
from python_polar_coding.tests.base import BasicVerifyPolarCode


class TestSCCode_1024_512(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
    }


class TestSCCode_1024_256(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 256,
    }


class TestSCCode_1024_768(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 768,
    }


class TestSCCode_2048_512(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 512,
    }


class TestSCCode_2048_1024(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
    }


class TestSCCode_2048_1536(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 1536,
    }
