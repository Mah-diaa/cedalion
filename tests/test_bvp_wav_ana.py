import numpy as np

from cedalion.sigproc.bvp_wav_ana_v12 import peakseek


def test_peakseek_no_restrictions():
    dummy_time = np.arange(0, 30*np.pi, np.pi/32)
    dummy_ts = np.sin(dummy_time)

    result_idx = np.arange(np.pi/2, 30*np.pi, 2*np.pi)
    result_val = np.sin(result_idx)

    max_idx, max_val = peakseek(dummy_ts)

    assert np.allclose(dummy_time[max_idx], result_idx, 1e-12)
    assert np.allclose(max_val, result_val, 1e-12)
