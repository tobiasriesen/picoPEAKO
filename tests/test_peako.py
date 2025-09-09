import unittest

import numpy as np
import xarray as xr

from picopeako.peako import Peako


def build_example_dataset():
    return xr.Dataset(
        {
            'doppler_spectrum': (('time', 'range', 'spectrum'), np.arange(25).reshape(5, 5, 1)),
            'range_layers': (('range'), np.arange(5)),
            'chirp_start_indices': (('chirp'), [0]),
        }, coords={
            'chirp': [0],
            'time': np.arange(5),
            'range': np.arange(5),
            'spectrum': [0]
        }
    )


class TestPeako(unittest.TestCase):

    def test_average_single_spectrum(self):
        data = build_example_dataset()['doppler_spectrum']
        peako = Peako()

        average = peako._average_single_spectrum(data, 2, 2, params={'t_avg': 2, 'h_avg': 2})

        self.assertEqual(average[0], 12)

    def test_average_multiple_spectra(self):
        data = build_example_dataset()
        peako = Peako()

        average = peako._average_multiple_spectra([data], params={'t_avg': 2, 'h_avg': 2})

        self.assertAlmostEqual(average[0]['doppler_spectrum'][2][2], 12, delta=1e-10)

    def test_smooth_single_spectrum(self):
        data = ((np.arange(32) == 16).astype(float)) * 0.9 + 0.1
        velbins = np.arange(32)
        peako = Peako()

        smoothed = peako._smooth_single_spectrum(data, velbins, params={'span': 2, 'polyorder': 1})

        np.testing.assert_allclose(smoothed[00:15], 0.1, atol=1e-10)
        np.testing.assert_allclose(smoothed[15:18], 10**(-2/3), atol=1e-10)
        np.testing.assert_allclose(smoothed[18:32], 0.1, atol=1e-10)


if __name__ == '__main__':
    unittest.main()
