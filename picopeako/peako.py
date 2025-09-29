import os
import json

import numpy as np
import scipy.signal as si
from tqdm import tqdm
import xarray as xr

from picopeako import AreaOverlapMetric, utils


class Peako:
    PARAM_NAMES = ('t_avg', 'h_avg', 'span', 'polyorder', 'width', 'prom')

    def __init__(self, configuration_filename=None, params=None, max_peaks=5, show_progress=True):
        """
        :param configuration_filename: filename of a json file containing the parameters
            and max_peaks. If provided, params and max_peaks are ignored.
        :param params: dictionary with parameters for the peak detection algorithm
        :param max_peaks: maximum number of peaks to detect per spectrum
        :param show_progress: if True, show progress bars. Default is True.
        """
        self.show_progress = show_progress
        if configuration_filename is not None:
            with open(configuration_filename, 'r') as f:
                config = json.load(f)
            self.params = config['params']
            self.max_peaks = config['max_peaks']
        else:
            self.params = {
                't_avg': 0,
                'h_avg': 0,
                'span': 0.0,
                'width': 0.0,
                'prom': 0.0,
                'polyorder': 2,
            } | (params or {})
            self.max_peaks = max_peaks

    def save_model(self, configuration_filename):
        """
        Save the model parameters to a json file.
        :param configuration_filename: filename of the json file to save the parameters to
        """
        assert configuration_filename.endswith('.json'), "Filename must end with .json"
        config = {'params': self.params, 'max_peaks': self.max_peaks}
        with open(configuration_filename, 'w') as f:
            json.dump(config, f)

    @classmethod
    def load_model(obj, configuration_filename):
        """
        Load the model parameters from a json file. If called on the class, returns a new instance
        of the class with the loaded parameters. If called on an instance, updates the instance's
        parameters.
        :param configuration_filename: filename of the json file to load the parameters from
        :return: instance of the class with the loaded parameters (if called on the class)
        """
        assert configuration_filename.endswith('.json'), "Filename must end with .json"

        with open(configuration_filename, 'r') as f:
            config = json.load(f)
        if isinstance(obj, type):
            return obj(configuration_filename)
        obj.params = config['params']
        obj.max_peaks = config['max_peaks']
        return obj

    def train_from_files(self, spectra_filenames, ground_truth_filenames, param_candidate_values,
                         get_qualities=False, quality_metric=AreaOverlapMetric(),
                         show_progress=True):
        """
        Train the peak detection algorithm using the provided spectra and ground truth files.

        :param spectra_filenames: list of filenames of the spectra files (netCDF)
        :param ground_truth_filenames: list of filenames of the ground truth files (netCDF)
        :param param_candidate_values: dictionary with lists of candidate values for each parameter
        :param get_qualities: if True, return the qualities for all parameter combinations
        :param quality_metric: instance of a quality metric class (default: AreaOverlapMetric)
        :return: best parameters and best quality (and qualities if get_qualities is True)
        """
        if ground_truth_filenames is None:
            ground_truth_filenames = [os.path.join(
                os.path.dirname(filename), f'marked_peaks_{os.path.basename(filename)}'
            ) for filename in spectra_filenames]

        spectra_datasets = [xr.open_dataset(filename) for filename in spectra_filenames]
        ground_truth_datasets = [xr.open_dataset(filename) for filename in ground_truth_filenames]
        return self.train(
            spectra_datasets, ground_truth_datasets, param_candidate_values=param_candidate_values,
            get_qualities=get_qualities, quality_metric=quality_metric, show_progress=show_progress
        )

    def train(self, spectra_datasets, ground_truth_datasets, param_candidate_values,
              get_qualities=False, quality_metric=AreaOverlapMetric(), show_progress=None):
        """
        Train the peak detection algorithm using the provided spectra and ground truth datasets.
        :param spectra_datasets: list of xarray datasets containing Doppler spectra
        :param ground_truth_datasets: list of xarray datasets containing ground truth peaks
        :param param_candidate_values: dictionary with lists of candidate values for each parameter
        :param get_qualities: if True, return the qualities for all parameter combinations
        :param quality_metric: instance of a quality metric class (default: AreaOverlapMetric)
        :param show_progress: if True, show progress bars
        :return: best parameters and best quality (and qualities if get_qualities is True)
        """
        show_progress = show_progress if show_progress is not None else self.show_progress
        qualities = np.zeros(
            tuple(len(param_candidate_values[param_name]) for param_name in self.PARAM_NAMES)
        )

        for spectra_dataset, ground_truth_dataset in tqdm(
            zip(spectra_datasets, ground_truth_datasets), desc="Input files",
            disable=(len(spectra_datasets) == 1) or not show_progress, total=len(spectra_datasets)
        ):
            times = ground_truth_dataset.time.values
            ranges = ground_truth_dataset.range.values

            # Calculate the time indices and range indices for each training sample
            time_indices = [np.argmin(np.abs(
                spectra_dataset.coords['time'].values -
                (time.astype('datetime64[s]').astype(int) - 978307200)
            )) for time in times]
            range_indices = [
                np.argmin(np.abs(spectra_dataset.range_layers.values - rnge)) for rnge in ranges
            ]

            positions = ground_truth_dataset.positions.values

            velocity_bins = []
            chirp_start_indices = spectra_dataset.chirp_start_indices.values
            for i, range_index in enumerate(range_indices):
                chirp_index = np.arange(len(chirp_start_indices))[
                    chirp_start_indices <= range_index
                ][-1]
                # TODO: This is inefficient, because it contains many duplicates.
                vel_bins = spectra_dataset.velocity_vectors[chirp_index].values
                velocity_bins.append(vel_bins)

            for i, p in enumerate(positions):
                positions[i] = utils.vel_to_ind(p, velocity_bins[i], fill_value=-1)
            positions = np.array(positions, dtype=int)

            pbar = tqdm(
                total=np.prod(
                    [len(param_candidate_values[param_name]) for param_name in self.PARAM_NAMES]
                ), desc="Combinations", leave=False, disable=not show_progress
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
                            spectrum, params={'span': span, 'polyorder': polyorder}
                        ) for spectrum in averaged_spectra
                    ]

                    # Iterate over the parameters of the peak detection function
                    for (m, width), (n, prom) in utils.enumerate_product(
                        param_candidate_values['width'], param_candidate_values['prom']
                    ):
                        pbar.update(1)

                        # Detect peaks in the smoothed spectra using the parameters
                        detected_peaks = np.array([
                            self._detect_single_spectrum(
                                spectrum, params={'prom': prom, 'width': width}
                            ) for spectrum in smoothed_spectra
                        ])

                        # Evaluate the peak detection quality using the quality metric
                        quality = np.sum([quality_metric.compute_metric(
                            detected_peaks[i], positions[i][positions[i] >= 0],
                            averaged_spectra[i], velocity_bins[i]
                        ) for i in range(len(detected_peaks))])

                        qualities[i, j, k, l, m, n] += quality

            pbar.close()

        # Convert the qualities array to an xarray DataArray for easier handling
        qualities = xr.DataArray(
            qualities, dims=self.PARAM_NAMES, coords=param_candidate_values
        ).to_dataset(name='quality')

        # Find the best parameters based on the quality metrics
        max_quality = np.max(qualities.quality.values)
        max_indices = np.unravel_index(np.argmax(qualities.quality.values), qualities.quality.shape)
        max_params = {
            't_avg': param_candidate_values['t_avg'][max_indices[0]],
            'h_avg': param_candidate_values['h_avg'][max_indices[1]],
            'span': param_candidate_values['span'][max_indices[2]],
            'polyorder': param_candidate_values['polyorder'][max_indices[3]],
            'width': param_candidate_values['width'][max_indices[4]],
            'prom': param_candidate_values['prom'][max_indices[5]]
        }

        self.params = max_params

        if get_qualities:
            return max_params, max_quality, qualities
        return max_params, max_quality

    def process_from_files(self, spectra_filenames, params=None, show_progress=True):
        """
        Process the provided spectra files using the peak detection algorithm.
        :param spectra_filenames: list of filenames of the spectra files (netCDF)
        :param params: dictionary with parameters for the peak detection algorithm. If None, the
            parameters stored in the instance are used.
        :param show_progress: if True, show progress bars
        :return: list of numpy arrays with detected peaks for each spectra file
        """
        spectra_datasets = [xr.open_dataset(filename) for filename in spectra_filenames]
        return self.process(spectra_datasets, params=params)

    def process(self, spec_data, params=None, show_progress=None):
        """
        Process the provided spectra datasets using the peak detection algorithm.
        :param spec_data: list of xarray datasets containing Doppler spectra
        :param params: dictionary with parameters for the peak detection algorithm. If None, the
            parameters stored in the instance are used.
        :param show_progress: if True, show progress bars
        :return: list of numpy arrays with detected peaks for each spectra dataset
        """
        show_progress = show_progress if show_progress is not None else self.show_progress
        params = self.params | (params or {})

        processed_spectra = []
        for spec in tqdm(spec_data, desc="Input files",
                         disable=(len(spec_data) == 1) or not show_progress):
            if params['t_avg'] > 0 or params['h_avg'] > 0:
                spec = self._average_all_spectra(spec, params=params, show_progress=show_progress)

            processed = np.zeros(
                spec['doppler_spectrum'].shape[:2] + (self.max_peaks,), dtype=float
            )
            for t in tqdm(range(spec['doppler_spectrum'].shape[0]), desc="Detecting  ", leave=False,
                          disable=spec['doppler_spectrum'].shape[0] == 1 or not show_progress):
                for h in range(spec['doppler_spectrum'].shape[1]):
                    # TODO: Fix velocity bins
                    smoothed = self._smooth_single_spectrum(
                        spec['doppler_spectrum'][t][h].values, params=params
                    )
                    detected = self._detect_single_spectrum(smoothed, params=params)
                    processed[t][h] = detected

            processed = xr.DataArray(
                processed, dims=['time', 'range', 'peaks'],
                coords={
                    'time': spec.time.values, 'range': spec.range_layers.values,
                    'peaks': np.arange(self.max_peaks)
                }
            )
            processed_spectra.append(processed)
        return processed_spectra

    def _get_params(self, override_params=None):
        params = self.params | (override_params or {})
        return [params.get(variable) for variable in self.PARAM_NAMES]

    def _average_all_spectra(self, spec_data, params=None, show_progress=None):
        show_progress = show_progress if show_progress is not None else self.show_progress
        t_avg, h_avg = self._get_params(override_params=params)[:2]

        # average spectra over neighbors in time-height
        avg_specs = xr.Dataset({'doppler_spectrum': xr.DataArray(
            np.zeros(spec_data.doppler_spectrum.shape), dims=['time', 'range_layers', 'spectrum'],
            coords={
                'time': spec_data.time.values, 'range_layers': spec_data.range_layers.values,
                'spectrum': spec_data.spectrum.values
            }
        ), 'chirp': spec_data.chirp})

        B = np.ones((1 + t_avg * 2, 1 + h_avg * 2)) / ((1 + t_avg * 2) * (1 + h_avg * 2))
        range_offsets = spec_data.chirp_start_indices.values
        for d in tqdm(range(avg_specs['doppler_spectrum'].values.shape[2]), desc="Averaging  ",
                      leave=False, disable=not show_progress):
            one_bin_avg = self._average_single_bin(
                spec_data['doppler_spectrum'].values, B, d, range_offsets
            )
            avg_specs['doppler_spectrum'][:, :, d] = one_bin_avg

        return avg_specs

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
            with np.errstate(invalid='ignore'):
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

    def _smooth_single_spectrum(self, averaged_spectrum, params=None):
        span, polyorder = self._get_params(override_params=params)[2:4]

        # When span is 1, or less than or equal to polyorder, return the averaged spectrum
        # directly, as no smoothing is needed/possible.
        span = utils.round_to_odd(span)
        if span == 1 or span <= polyorder:
            return averaged_spectrum

        # Fill NaN values, convert to logarithmic scale, smooth, and convert back to linear scale
        averaged_spectrum = np.where(
            np.isnan(averaged_spectrum), np.min(averaged_spectrum), averaged_spectrum
        )

        smoothed_spectrum = si.savgol_filter(
            utils.lin2z(averaged_spectrum), span, polyorder=polyorder, mode='nearest'
        )
        return utils.z2lin(smoothed_spectrum)

    def _detect_single_spectrum(self, spectrum, params=None):
        width_thresh, prom = self._get_params(override_params=params)[4:6]
        # Convert to logarithmic scale and replace NaNs with a fill value
        fillvalue = -100.0
        spectrum = utils.lin2z(spectrum)
        spectrum[np.isnan(spectrum)] = fillvalue

        # Call scipy.signal.find_peaks to detect peaks in the (logarithmic) spectrum
        # it is important that nan values are not included in the spectrum passed to si
        locs, _ = si.find_peaks(spectrum, prominence=prom, width=width_thresh)
        locs = locs[spectrum[locs] > fillvalue]
        locs = locs[0: self.max_peaks]

        # Artificially create output dimension of same length as Doppler bins to
        # avoid xarray value error
        out = np.full((self.max_peaks), 0, dtype=int)
        out[range(len(locs))] = locs

        return out
