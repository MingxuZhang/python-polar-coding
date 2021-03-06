from unittest import TestCase

from python_polar_coding.polar_codes.sc_list import SCListPolarCodec
from python_polar_coding.tests.base import BasicVerifyPolarCode


class TestSCListPolarCode1024_512_4(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 4,
    }


class TestSCListPolarCode1024_512_8(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 1024,
        'K': 512,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_8(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 8,
    }


class TestSCListPolarCode2048_1024_16(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 16,
    }


class TestSCListPolarCode2048_1024_32(BasicVerifyPolarCode, TestCase):
    polar_code_class = SCListPolarCodec
    code_parameters = {
        'N': 2048,
        'K': 1024,
        'L': 32,
    }
