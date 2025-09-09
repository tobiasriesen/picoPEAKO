# Deactivate flake8 for this file

import os

import numpy as np
import scipy.signal as si
from tqdm import tqdm
import xarray as xr

from picopeako import AreaOverlapMetric, utils


class Peako:
    PARAM_NAMES = ('t_avg', 'h_avg', 'span', 'polyorder', 'width', 'prom')

    def __init__(self, params=None):
        params = params or {}
        self.params = {
            't_avg': 0,
            'h_avg': 0,
            'span': 0.0,
            'width': 0.0,
            'prom': 0.0,
            'polyorder': 2,
        } | params

    def train(self, filenames, param_candidate_values, ground_truth_filenames=None):
        qualities = np.zeros(
            tuple(len(param_candidate_values[param_name]) for param_name in self.PARAM_NAMES)
        )

        if ground_truth_filenames is None:
            ground_truth_filenames = [os.path.join(
                os.path.dirname(filename), f'marked_peaks_{os.path.basename(filename)}'
            ) for filename in filenames]

        for filename, ground_truth_filename in tqdm(
            zip(filenames, ground_truth_filenames), desc="Input files",
            disable=len(filenames) == 1, total=len(filenames)
        ):
            train_dataset = xr.open_dataset(ground_truth_filename)
            spectra_dataset = xr.open_dataset(filename)

            times = train_dataset.time.values
            ranges = train_dataset.range.values

            # Calculate the time indices and range indices for each training sample
            time_indices = [np.argmin(np.abs(
                spectra_dataset.coords['time'].values -
                (time.astype('datetime64[s]').astype(int) - 978307200)
            )) for time in times]
            range_indices = [
                np.argmin(np.abs(spectra_dataset.range_layers.values - rnge)) for rnge in ranges
            ]

            positions = train_dataset.positions.values

            velocity_bins = []
            chirp_start_indices = spectra_dataset.chirp_start_indices.values
            for i, range_index in enumerate(range_indices):
                chirp_index = np.arange(len(chirp_start_indices))[
                    chirp_start_indices <= range_index
                ][-1]
                # TODO: This is inefficient, because it contains many duplicates.
                vel_bins = spectra_dataset.velocity_vectors[chirp_index].values
                velocity_bins.append(vel_bins)

            positions = np.array(positions, dtype=int)

            pbar = tqdm(
                total=np.prod(
                    [len(param_candidate_values[param_name]) for param_name in self.PARAM_NAMES]
                ), desc="Combinations", leave=False
            )

            # Iterate over the parameters of the average function
            for (i, t_avg), (j, h_avg) in utils.enumerate_product(
                param_candidate_values['t_avg'], param_candidate_values['h_avg']
            ):
                # Average the spectra using the parameters
                averaged_spectra = [
                    self._average_single_spectrum(
                        spectra_dataset.doppler_spectrum, t, h,
                        params={'t_avg': t_avg, 'h_avg': h_avg}
                    ) for t, h in zip(time_indices, range_indices)
                ]
                # Iterate over the parameters of the smoothing function
                for (k, span), (l, polyorder) in utils.enumerate_product(
                    param_candidate_values['span'], param_candidate_values['polyorder']
                ):
                    # Smooth the averaged spectra using the parameters
                    smoothed_spectra = [
                        self._smooth_single_spectrum(
                            spectrum, velocity_bins[i],
                            params={'span': span, 'polyorder': polyorder}
                        ) for i, spectrum in enumerate(averaged_spectra)
                    ]

                    # Iterate over the parameters of the peak detection function
                    for (m, width), (n, prom) in utils.enumerate_product(
                        param_candidate_values['width'], param_candidate_values['prom']
                    ):
                        pbar.update(1)

                        # Detect peaks in the smoothed spectra using the parameters
                        detected_peaks = np.array([
                            self._detect_single_spectrum(
                                spectrum, max_peaks=10, params={'prom': prom, 'width': width}
                            ) for spectrum in smoothed_spectra
                        ])

                        # Evaluate the peak detection quality using the quality metric
                        quality = np.sum([AreaOverlapMetric().compute_metric(
                            detected_peaks, positions[i], averaged_spectra[i], velocity_bins[i]
                        ) for i in range(len(detected_peaks))])

                        qualities[i, j, k, l, m, n] += quality

            pbar.close()

        # Save the quality metrics to a file
        quality_filename = os.path.join(os.path.dirname(filename), 'peako_quality_metrics.npz')
        np.savez(quality_filename, qualities=qualities)

        # Find the best parameters based on the quality metrics
        max_quality = np.max(qualities)
        max_indices = np.unravel_index(np.argmax(qualities), qualities.shape)
        max_params = {
            't_avg': param_candidate_values['t_avg'][max_indices[0]],
            'h_avg': param_candidate_values['h_avg'][max_indices[1]],
            'span': param_candidate_values['span'][max_indices[2]],
            'polyorder': param_candidate_values['polyorder'][max_indices[3]],
            'width': param_candidate_values['width'][max_indices[4]],
            'prom': param_candidate_values['prom'][max_indices[5]]
        }

        return max_params, max_quality

    def process(self, spec_data, params=None):
        max_peaks = 5
        params = self.params | (params or {})
        if params['t_avg'] > 0 or params['h_avg'] > 0:
            spec_data = self._average_multiple_spectra(spec_data, params=params)
        processed_spectra = []
        for f in range(len(spec_data)):
            processed = np.zeros(
                spec_data[f]['doppler_spectrum'].shape[:2] + (max_peaks,), dtype=float
            )
            for t in range(spec_data[f]['doppler_spectrum'].shape[0]):
                for h in range(spec_data[f]['doppler_spectrum'].shape[1]):
                    # TODO: Fix velocity bins
                    smoothed = self._smooth_single_spectrum(
                        spec_data[f]['doppler_spectrum'][t][h].values,
                        np.arange(119), params=params
                    )
                    detected = self._detect_single_spectrum(smoothed, max_peaks, params=params)
                    processed[t][h] = detected

            processed_spectra.append(processed)
        return processed_spectra

    def _get_params(self, override_params=None):
        params = self.params | (override_params or {})
        return [params.get(variable) for variable in self.PARAM_NAMES]

    def _average_multiple_spectra(self, spec_data, params=None):
        t_avg, h_avg = self._get_params(override_params=params)[:2]
        average_data = []
        for f in range(len(spec_data)):
            # average spectra over neighbors in time-height
            avg_specs = xr.Dataset({'doppler_spectrum': xr.DataArray(
                np.zeros(spec_data[f].doppler_spectrum.shape), dims=['time', 'range', 'spectrum'],
                coords={
                    'time': spec_data[f].time.values, 'range': spec_data[f].range_layers.values,
                    'spectrum': spec_data[f].spectrum.values
                }
            ), 'chirp': spec_data[f].chirp})

            average_data.append(avg_specs)

            B = np.ones((1 + t_avg * 2, 1 + h_avg * 2)) / ((1 + t_avg * 2) * (1 + h_avg * 2))
            range_offsets = spec_data[f].chirp_start_indices.values
            for d in range(avg_specs['doppler_spectrum'].values.shape[2]):
                one_bin_avg = self._average_single_bin(
                    spec_data[f]['doppler_spectrum'].values, B, d, range_offsets
                )
                avg_specs['doppler_spectrum'][:, :, d] = one_bin_avg

        return average_data

    def _average_single_bin(self, specdata_values: np.array, B: np.array,
                            doppler_bin: int, range_offsets: list):
        C = []
        r_ind = np.hstack((range_offsets, specdata_values.shape[1]))
        for c in range(len(r_ind) - 1):
            A = specdata_values[:, r_ind[c]:r_ind[c + 1], doppler_bin]

            # Convolve with special handling of NaN values:
            # Based on https://stackoverflow.com/questions/38318362/2d-convolution-in-python-with-missing-data # noqa
            # When more than 50% of the values in the convolution window are NaN, the result is set
            # to NaN. Otherwise, only the available values are used for the convolution.
            convolved = si.convolve2d(np.where(np.isnan(A), 0, A), B[::-1, ::-1], mode='same')
            valid_weight = si.convolve2d(~np.isnan(A), B[::-1, ::-1], mode='same')
            convolved = np.where(valid_weight > 0, convolved / valid_weight, np.nan)
            count_missing = si.convolve2d(np.isnan(A), np.full_like(B, 1/B.size), mode='same')
            convolved[count_missing > 0.5] = np.nan

            C.append(si.convolve2d(A, B, mode='same'))

        C = np.hstack(C)
        return C

    def _average_single_spectrum(self, spec_chunk, t, h, params=None):
        t_avg, h_avg = self._get_params(override_params=params)[:2]
        tmin = np.max([0, t - t_avg])
        tmax = np.min([spec_chunk.shape[0], t + t_avg])
        hmin = np.max([0, h - h_avg])
        hmax = np.min([spec_chunk.shape[1], h + h_avg])

        return np.average(
            spec_chunk.isel(time=np.arange(tmin, tmax+1), range=np.arange(hmin, hmax+1)),
            axis=(0, 1)
        )

    # TODO: The velbins is a useless feature. Please remove it.
    def _smooth_single_spectrum(self, averaged_spectrum, velbins, params=None):
        span, polyorder = self._get_params(override_params=params)[2:4]

        # Determine the window length based on the span (m/s) and velocity resolution (m/s)
        window_length = utils.round_to_odd(span / utils.get_vel_resolution(velbins))

        # When window_length is 1, or less than or equal to polyorder, return the averaged spectrum
        # directly, as no smoothing is needed/possible.
        if window_length == 1 or window_length <= polyorder:
            return averaged_spectrum

        # Fill NaN values, convert to logarithmic scale, smooth, and convert back to linear scale
        averaged_spectrum = np.where(
            np.isnan(averaged_spectrum), np.min(averaged_spectrum), averaged_spectrum
        )

        smoothed_spectrum = si.savgol_filter(
            utils.lin2z(averaged_spectrum), window_length, polyorder=polyorder, mode='nearest'
        )
        return utils.z2lin(smoothed_spectrum)

    def _detect_single_spectrum(self, spectrum, max_peaks, params=None):
        width_thresh, prom = self._get_params(override_params=params)[4:6]
        # Convert to logarithmic scale and replace NaNs with a fill value
        fillvalue = -100.0
        spectrum = utils.lin2z(spectrum)
        spectrum[np.isnan(spectrum)] = fillvalue

        # Call scipy.signal.find_peaks to detect peaks in the (logarithmic) spectrum
        # it is important that nan values are not included in the spectrum passed to si
        locs, _ = si.find_peaks(spectrum, prominence=prom, width=width_thresh)
        locs = locs[spectrum[locs] > fillvalue]
        locs = locs[0: max_peaks]

        # Artificially create output dimension of same length as Doppler bins to
        # avoid xarray value error
        out = np.full((max_peaks), 0, dtype=int)
        out[range(len(locs))] = locs

        return out
