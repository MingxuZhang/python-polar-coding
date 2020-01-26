from datetime import datetime

from python_polar_coding.channels.simple import SimpleBPSKModulationAWGN
from python_polar_coding.polar_codes.fast_ssc import FastSSCPolarCode
from python_polar_coding.simulation import simulation


def fast_ssc_simulation():
    code = FastSSCPolarCode(N=4096, K=1024, design_snr=2.0)
    channel = SimpleBPSKModulationAWGN(fec_rate=code.K/code.N)
    snr_range = [1.0, 1.5, 2.0, 2.5, 3.0]

    results = list()

    for snr in snr_range:
        start = datetime.now()
        result = simulation(code=code,
                            channel=channel,
                            snr_db=snr,
                            messages=1000)
        end = datetime.now()
        print(f'Experiment took {(end-start).seconds} seconds')
        results.append(result)

    for r in results:
        print(r)


if __name__ == '__main__':
    fast_ssc_simulation()
