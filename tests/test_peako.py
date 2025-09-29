import unittest

import numpy as np
import xarray as xr

from picopeako.peako import Peako


def build_example_dataset(data):
    return xr.Dataset(
        {
            'doppler_spectrum': (('time', 'range', 'spectrum'), data),
            'range_layers': (('range'), np.arange(data.shape[1])),
            'chirp_start_indices': (('chirp'), [0]),
            'velocity_vectors': (('chirp', 'spectrum'), np.arange(data.shape[2]).reshape(1, -1))
        }, coords={
            'chirp': [0],
            'time': np.arange(data.shape[0]),
            'range': np.arange(data.shape[1]),
            'spectrum': np.arange(data.shape[2]),
        }
    )


def build_example_ground_truth(data):
    times = [d[0] for d in data]
    ranges = [d[1] for d in data]
    peaks = [d[2] for d in data]
    return xr.Dataset(
        {
            'time': (('spec'), np.array(times, dtype='datetime64[ns]')),
            'range': (('spec'), np.array(ranges, dtype='float32')),
            'positions': (('spec', 'peak'), np.array(peaks, dtype='float32').reshape(len(peaks), 1))
        }
    )


class TestPeako(unittest.TestCase):

    def test_average_single_spectrum(self):
        data = build_example_dataset(np.arange(25).reshape(5, 5, 1))['doppler_spectrum']
        peako = Peako()

        average = peako._average_single_spectrum(data, 2, 2, params={'t_avg': 2, 'h_avg': 2})

        self.assertEqual(average[0], 12)

    def test_average_all_spectra(self):
        data = build_example_dataset(np.arange(25).reshape(5, 5, 1))
        peako = Peako()

        average = peako._average_all_spectra(data, params={'t_avg': 2, 'h_avg': 2})

        self.assertAlmostEqual(average['doppler_spectrum'][2][2], 12, delta=1e-10)

    def test_smooth_single_spectrum(self):
        data = ((np.arange(32) == 16).astype(float)) * 0.9 + 0.1
        peako = Peako()

        smoothed = peako._smooth_single_spectrum(data, params={'span': 2, 'polyorder': 1})

        np.testing.assert_allclose(smoothed[00:15], 0.1, atol=1e-10)
        np.testing.assert_allclose(smoothed[15:18], 10**(-2/3), atol=1e-10)
        np.testing.assert_allclose(smoothed[18:32], 0.1, atol=1e-10)

    def test_detect_single_spectrum(self):
        data = ((np.arange(32) == 16).astype(float)) * 0.9 + 0.1
        max_peaks = 5
        peako = Peako()

        peaks = peako._detect_single_spectrum(data, params={'width': 1, 'prom': 1})

        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(peaks.shape, (max_peaks,))
        self.assertEqual(peaks[0], 16)
        self.assertEqual(peaks[1], 0)
        self.assertEqual(peaks[2], 0)
        self.assertEqual(peaks[3], 0)
        self.assertEqual(peaks[4], 0)

    def test_process(self):
        data = build_example_dataset((np.arange(125) == 62).reshape(5, 5, 5).astype(float))
        peako = Peako(max_peaks=3)

        peaks = peako.process([data], params={
            't_avg': 0, 'h_avg': 0, 'span': 0, 'polyorder': 1, 'width': 0.5, 'prom': 0.0
        })

        self.assertIsInstance(peaks, list)
        self.assertIsInstance(peaks[0], xr.DataArray)
        self.assertEqual(peaks[0].shape, (5, 5, 3))
        self.assertEqual(peaks[0][2, 2, 0], 2)

    def test_train(self):
        data = build_example_dataset((np.arange(125) == 62).reshape(5, 5, 5).astype(float))
        ground_truth = build_example_ground_truth([(978307200 + 2, 2, 2)])
        peako = Peako(max_peaks=1)

        best_params, best_quality = peako.train(
            [data], [ground_truth],
            param_candidate_values={
                't_avg': [0, 1],
                'h_avg': [0, 1],
                'span': [0, 2, 3],
                'polyorder': [1, 2],
                'width': [0.5, 1, 2],
                'prom': [0.0, 0.5, 1]
            }
        )

        self.assertIsInstance(best_params, dict)
        self.assertIsInstance(best_quality, float)
        self.assertIn('t_avg', best_params)
        self.assertIn('h_avg', best_params)
        self.assertIn('span', best_params)
        self.assertIn('polyorder', best_params)
        self.assertIn('width', best_params)
        self.assertIn('prom', best_params)


if __name__ == '__main__':
    unittest.main()
