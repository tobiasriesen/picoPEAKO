import math
import warnings

import numpy as np

from picopeako import PeakDetectionQualityMetric, utils


class AreaOverlapMetric(PeakDetectionQualityMetric):
    FILL_VALUE = -999.0

    def compute_metric(self, detected_peaks, reference_peaks, spectrum, velocity_bins):
        # Convert the spectrum to logarithmic units
        spectrum_db = utils.lin2z(spectrum)

        # Sort the peaks
        reference_peaks.sort()
        reference_peaks = np.unique(reference_peaks[~np.isnan(reference_peaks)])
        # convert velocities to indices
        reference_peaks = np.asarray(
            [utils.argnearest(velocity_bins, val) for val in reference_peaks]
        )
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
