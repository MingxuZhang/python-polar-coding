from concurrent.futures import ProcessPoolExecutor
from functools import partial

from modelling.utils.db import DB_NAME
from modelling.utils.functions import major_experiment
from modelling.utils.params import code_rates, lengths, snr_range
from polar_codes import FastSSCPolarCode

COLLECTION = 'fast_ssc'
MAX_WORKERS = 4
MESSAGES = 100
STEP_SIZE = 1000

codes = [
    FastSSCPolarCode(
        codeword_length=l,
        info_length=int(l*cr),
        is_systematic=True,
    )
    for l in lengths
    for cr in code_rates
]
fast_ssc_experiment = partial(
    major_experiment,
    snr_range=snr_range,
    db_name=DB_NAME,
    collection=COLLECTION,
    # messages=MESSAGES,
    # step_size=STEP_SIZE,
)


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        results = ex.map(fast_ssc_experiment, codes)
