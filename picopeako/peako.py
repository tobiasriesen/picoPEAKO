# Deactivate flake8 for this file

from itertools import product
import math
import os
import warnings

import numpy as np
import scipy.signal as si
from tqdm import tqdm
import xarray as xr

from picopeako import utils


class Peako:
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

    def train(self, filename, param_candidate_values):
        # Load the spectra and the velocity bins for each spectrum that has a ground truth
        ground_truth_filename = os.path.join(
            os.path.dirname(filename), f'marked_peaks_{os.path.basename(filename)}'
        )
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
            # positions[i] = np.argmin(np.abs(vel_bins[:, None] - positions[0][None, :]), axis=0)

        positions = np.array(positions, dtype=int)

        qualities = np.zeros((len(param_candidate_values['t_avg']),
                                len(param_candidate_values['h_avg']),
                                len(param_candidate_values['span']),
                                len(param_candidate_values['polyorder']),
                                len(param_candidate_values['width']),
                                len(param_candidate_values['prom'])))
        max_quality = -math.inf
        max_params = None

        pbar = tqdm(
            total=(len(param_candidate_values['t_avg']) * len(param_candidate_values['h_avg']) *
                   len(param_candidate_values['span']) * len(param_candidate_values['polyorder']) *
                   len(param_candidate_values['width']) * len(param_candidate_values['prom'])),
            desc="Training Peako"
        )

        # Iterate over the parameters of the average function
        for t_avg, h_avg in product(
            param_candidate_values['t_avg'], param_candidate_values['h_avg']
        ):
            # Average the spectra using the parameters
            averaged_spectra = [
                self._average_single_spectrum(
                    spectra_dataset.doppler_spectrum, t, h, params={'t_avg': t_avg, 'h_avg': h_avg}
                ) for t, h in zip(time_indices, range_indices)
            ]
            # Iterate over the parameters of the smoothing function
            for span, polyorder in product(
                param_candidate_values['span'], param_candidate_values['polyorder']
            ):
                # Smooth the averaged spectra using the parameters
                smoothed_spectra = [
                    self._smooth_single_spectrum(
                        spectrum, velocity_bins[i], params={'span': span, 'polyorder': polyorder}
                    ) for i, spectrum in enumerate(averaged_spectra)
                ]

                # Iterate over the parameters of the peak detection function
                for width, prom in product(
                    param_candidate_values['width'], param_candidate_values['prom']
                ):
                    pbar.update(1)

                    # Detect peaks in the smoothed spectra using the parameters
                    detected_peaks = np.array([
                        self._detect_single_spectrum(
                            spectrum, prom, width, max_peaks=10
                        ) for spectrum in smoothed_spectra
                    ])

                    # Evaluate the peak detection quality using the quality metric
                    quality = np.sum([AreaOverlapMetric().compute_metric(
                        detected_peaks, positions[i], averaged_spectra[i], velocity_bins[i]
                    ) for i in range(len(detected_peaks))])

                    max_quality = max(max_quality, quality)
                    if quality == max_quality:
                        max_params = {
                            't_avg': t_avg,
                            'h_avg': h_avg,
                            'span': span,
                            'width': width,
                            'prom': prom,
                            'polyorder': polyorder
                        }

        pbar.close()

        return max_params, max_quality

    def _get_params(self, override_params=None):
        params = self.params | (override_params or {})
        return [params.get(variable) for variable in (
            't_avg', 'h_avg', 'span', 'polyorder', 'width', 'prom'
        )]

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

    def _detect_single_spectrum(self, spectrum, prom, width_thresh, max_peaks):
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
        out = np.full(spectrum.shape[0], 0, dtype=int)
        out[range(len(locs))] = locs

        return out


class PeakDetectionQualityMetric:
    def compute_metric(self, detected_peaks, reference_peaks, spectrum, velocity_bins):
        raise NotImplementedError("This method should be overridden by subclasses")


class AreaOverlapMetric(PeakDetectionQualityMetric):
    FILL_VALUE = -999.0

    def compute_metric(self, detected_peaks, reference_peaks, spectrum, velocity_bins):
        # Convert the spectrum to logarithmic units
        spectrum_db = utils.lin2z(spectrum)

        # Sort the peaks
        reference_peaks.sort()
        reference_peaks = np.unique(reference_peaks[~np.isnan(reference_peaks)])
        # convert velocities to indices
        reference_peaks = np.asarray([utils.argnearest(velocity_bins, val) for val in reference_peaks])
        detected_peaks = np.unique(detected_peaks[(detected_peaks > 0)])
        detected_peaks.sort()

        # Find left and right edges of the peaks
        le_reference_peaks, re_reference_peaks = self.find_edges(spectrum, reference_peaks)
        le_alg_peaks, re_alg_peaks = self.find_edges(spectrum, detected_peaks)
        similarity = 0
        overlap_area = math.inf

        while (len(detected_peaks) > 0) & (len(reference_peaks) > 0) & (overlap_area > 0):
            # compute maximum overlapping area
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                user_ind, alg_ind, overlap_area = self.overlapping_area(
                    [le_reference_peaks, re_reference_peaks], [le_alg_peaks, re_alg_peaks],
                    spectrum_db, np.nanmin(spectrum_db), velocity_bins
                )

            similarity += overlap_area
            if user_ind is not None:
                reference_peaks = np.delete(reference_peaks, user_ind)
                le_reference_peaks = np.delete(le_reference_peaks, user_ind)
                re_reference_peaks = np.delete(re_reference_peaks, user_ind)
            if alg_ind is not None:
                detected_peaks = np.delete(detected_peaks, alg_ind)
                le_alg_peaks = np.delete(le_alg_peaks, alg_ind)
                re_alg_peaks = np.delete(re_alg_peaks, alg_ind)

        # Subtract area of non-overlapping regions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for i in range(len(le_alg_peaks)):
                similarity -= self.area_above_floor(
                    le_alg_peaks[i], re_alg_peaks[i], spectrum_db,
                    np.nanmin(spectrum_db), velocity_bins
                )
            for i in range(len(le_reference_peaks)):
                similarity -= self.area_above_floor(
                    le_reference_peaks[i], re_reference_peaks[i], spectrum_db,
                    np.nanmin(spectrum_db), velocity_bins
                )

        return similarity

    def find_edges(self, spectrum, peak_locations):
        left_edges = []
        right_edges = []

        for p_ind in range(len(peak_locations)):
            # start with the left edge
            p_l = peak_locations[p_ind]

            closest_below_noise_left = np.where(spectrum[0:p_l] == self.FILL_VALUE)
            if len(closest_below_noise_left[0]) == 0:
                closest_below_noise_left = 0
            else:
                # add 1 to get the first bin of the peak which is not fill_value
                closest_below_noise_left = max(closest_below_noise_left[0]) + 1

            if p_ind == 0:
                # if this is the first peak, the left edge is the closest_below_noise_left
                left_edge = closest_below_noise_left
            elif peak_locations[p_ind - 1] > closest_below_noise_left:
                # merged peaks
                left_edge = np.argmin(spectrum[peak_locations[p_ind - 1]: p_l])
                left_edge = left_edge + peak_locations[p_ind - 1]
            else:
                left_edge = closest_below_noise_left

            # Repeat for right edge
            closest_below_noise_right = np.where(spectrum[p_l:-1] == self.FILL_VALUE)
            if len(closest_below_noise_right[0]) == 0:
                # if spectrum does not go below noise (fill value), set it to the last bin
                closest_below_noise_right = len(spectrum) - 1
            else:
                # subtract one to obtain the last index of the peak
                closest_below_noise_right = min(closest_below_noise_right[0]) + p_l - 1

            # if this is the last (rightmost) peak, this first guess is the right edge
            if p_ind == (len(peak_locations) - 1):
                right_edge = closest_below_noise_right

            elif peak_locations[p_ind + 1] < closest_below_noise_right:
                right_edge = np.argmin(spectrum[p_l:peak_locations[p_ind + 1]]) + p_l
            else:
                right_edge = closest_below_noise_right

            left_edges.append(int(left_edge))
            right_edges.append(int(right_edge))

        return left_edges, right_edges

    def area_above_floor(self, left_edge, right_edge, spectrum, noise_floor, velbins):
        spectrum_above_noise = spectrum - noise_floor
        spectrum_above_noise = np.where(spectrum_above_noise < 0, 0, spectrum_above_noise)

        # Riemann sum (approximation of area):
        velocity_resolution = utils.get_vel_resolution(velbins)
        area = np.nansum(spectrum_above_noise[left_edge:right_edge]) * velocity_resolution

        return area

    def overlapping_area(self, edge_list_1, edge_list_2, spectrum, noise_floor, velbins):
        max_area = 0
        peak_ind_1 = None
        peak_ind_2 = None

        for i1 in range(len(edge_list_1[0])):
            for i2 in range(len(edge_list_2[0])):
                this_area = self.compute_overlapping_area(
                    i1, i2, edge_list_1, edge_list_2, spectrum, noise_floor, velbins
                )
                if this_area > max_area:
                    peak_ind_1 = i1
                    peak_ind_2 = i2
                    max_area = this_area
        return peak_ind_1, peak_ind_2, max_area

    def compute_overlapping_area(self, i1, i2, edge_list_1, edge_list_2, spectrum,
                                 noise_floor, velbins):
        left_edge_overlap = max(edge_list_1[0][i1], edge_list_2[0][i2])
        leftest_edge = min(edge_list_1[0][i1], edge_list_2[0][i2])
        right_edge_overlap = min(edge_list_1[1][i1], edge_list_2[1][i2])
        rightest_edge = max(edge_list_1[1][i1], edge_list_2[1][i2])

        # Compute edges of joint area and of region outside joint area
        area = self.area_above_floor(
            left_edge_overlap, right_edge_overlap, spectrum, noise_floor, velbins
        )
        if area > 0:
            area = area - self.area_above_floor(
                leftest_edge, left_edge_overlap, spectrum, noise_floor, velbins
            )
            area = area - self.area_above_floor(
                right_edge_overlap, rightest_edge, spectrum, noise_floor, velbins
            )

        return area
