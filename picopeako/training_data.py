import datetime
import os
import random
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import xarray as xr


def lin2z(array):
    """
    convert linear values to dB (for np.array or single number)
    :param array: np.array or single number
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 10 * np.log10(array)
        return out


def format_hms(unixtime):
    """format time stamp in seconds since 01.01.1970 00:00 UTC to HH:MM:SS
    :param unixtime: time stamp (seconds since 01.01.1970 00:00 UTC)
    """
    return datetime.datetime.utcfromtimestamp(unixtime).strftime("%H:%M:%S")


def round_to_odd(f):
    """round to odd number
    :param f: float number to be rounded to odd number
    """
    return round(f) if round(f) % 2 == 1 else round(f) + 1


def argnearest(array, value):
    """larda function to find the index of the nearest value in a sorted array, for example time or range axis

    :param array: sorted array with values, list and dask arrays will be converted to 1D array
    :param value: value for which to find the nearest neighbor
    :return:
        index of the nearest neighbor in array
    """
    if type(array) in [list, xr.DataArray]:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def get_vel_resolution(vel_bins):
    return np.nanmedian(np.diff(vel_bins))


def get_chirp_offsets(specdata):
    """
    utility function to create an array of the range indices of chirp offsets, starting with [0]
    and ending with [n_range_layers]
    :param specdata: Doppler spectra DataSet containing chirp_start_indices and n_range_layers
    :return:
    """
    return np.hstack((specdata.chirp_start_indices.values, specdata.n_range_layers.values))


def get_closest_time(time, time_array):
    """"
    :param time: datetime.datetime
    :param time_array: xr.DataArray containing time stamp
    """
    time_array = time_array.values
    if (time_array < 1e9).all() and (time_array > 3e8).all():
        time_array = (datetime.datetime(2001, 1, 1) - datetime.datetime(1970, 1, 1)).total_seconds() + time_array
    ts = (time - datetime.datetime(1970, 1, 1)).total_seconds()
    return argnearest(time_array, ts)


class TrainingData(object):
    def __init__(self, specfiles_in: list, num_spec=30, max_peaks=5, verbosity=0):
        """
        Initialize TrainingData object; read in the spectra files contained in specfiles_in
        :param specfiles_in: list of strings specifying radar spectra files (netcdf format)
        :param num_spec: (int) number of spectra to mark by the user (default 30)
        :param max_peaks: (int) maximum number of peaks per spectrum (default 5)

        """
        self.specfiles_in = specfiles_in
        self.spec_data = [xr.open_dataset(fin, mask_and_scale=True) for fin in specfiles_in]
        self.num_spec = [num_spec] * len(self.spec_data)
        self.tdim = []
        self.rdim = []
        self.training_data_out = []
        self.peaks_ncfiles = []
        self.plot_count = []
        self.verbosity = verbosity
        self.max_peaks = max_peaks
        self._update_dimensions(num_spec)

    def mark_random_spectra(self, plot_smoothed=False, **kwargs):
        """
        Mark random spectra in TrainingData.spec_data (number of randomly drawn spectra in time-height space defined by
        TrainingData.num_spec) and save x and y locations
        :param kwargs:
               num_spec: update TrainingData.num_spec
               span: span for smoothing. Required if plot_smoothed=True
               yRange: tupel of min and max range index to choose random spectra from
        """

        if 'num_spec' in kwargs:
            self.num_spec[:] = kwargs['num_spec']

        closeby = kwargs['closeby'] if 'closeby' in kwargs else np.repeat(None, len(self.spec_data))
        yRange = kwargs['yRange'] if 'yRange' in kwargs else np.repeat(None, len(self.spec_data))

        for n in range(len(self.spec_data)):
            s = 0
            if closeby[n] is not None:
                tind = get_closest_time(closeby[n][0], self.spec_data[n].time)
                tind = (np.max([1, tind - 10]), np.min([self.tdim[n] - 1, tind + 10]))
                rind = argnearest(self.spec_data[n].range, closeby[n][1])
                rind = (np.max([1, rind - 5]), np.min([self.rdim[n] - 1, rind + 5]))
            elif yRange[n] is not None:
                tind = (1, self.tdim[n] - 1)
                rind = yRange
            else:
                tind = (1, self.tdim[n] - 1)
                rind = (1, self.rdim[n] - 1)
            while s < self.num_spec[n]:
                random_index_t = random.randint(tind[0], tind[1])
                random_index_r = random.randint(rind[0], rind[1])
                if self.verbosity > 1:
                    print(f'r: {random_index_r}, t: {random_index_t}')

                # vals will be NaN if there is no data available for the given spectrum.
                vals, _ = self._input_peak_locations(n, random_index_t, random_index_r, plot_smoothed, **kwargs)

                # Skip if no peaks were found
                if not np.all(np.isnan(vals)):
                    self.training_data_out[n]['time'][s] = self.spec_data[n].time[random_index_t]
                    self.training_data_out[n]['range'][s] = self.spec_data[n].range_layers[random_index_r]
                    self.training_data_out[n]['positions'][s, 0:len(vals)] = vals
                    s += 1
                    self.plot_count[n] = s

    def _update_dimensions(self, num_spec):
        """
        update the list of time and range dimensions stored in TrainingData.tdim and TrainingData.rdim,
        update arrays in which found peaks are stored,
        also update the names of the netcdf files into which found peaks are stored
        """
        self.tdim = []
        self.rdim = []
        self.training_data_out = []

        # loop over netcdf files
        for f in range(len(self.spec_data)):
            self.tdim.append(len(self.spec_data[f]['time']))
            self.rdim.append(len(self.spec_data[f]['range']))
            self.training_data_out.append(self._empty_training_dataset(num_spec))
            ncfile = os.path.join(
                os.path.dirname(self.specfiles_in[f]),
                f'marked_peaks_{os.path.basename(self.specfiles_in[f])}.nc'
            )
            self.peaks_ncfiles.append(ncfile)
            self.plot_count.append(0)

    def _empty_training_dataset(self, num_spec):
        data_dict = {
            'time': (('spec'), np.empty(num_spec, dtype='datetime64[ns]')),
            'range': (('spec'), np.empty(num_spec, dtype='float32')),
            'chirp': (('spec'), np.empty(num_spec, dtype='int32')),
            'positions': (('spec', 'peak'), np.empty((num_spec, self.max_peaks), dtype='int32'))
        }
        ds = xr.Dataset(data_dict)
        return ds

    def _input_peak_locations(self, n_file, t_index, r_index, plot_smoothed, **kwargs):
        """
        :param n_file: the index of the netcdf file from which to mark spectrum by hand
        :param t_index: the time index of the spectrum
        :param r_index: the range index of the spectrum
        :param plot_smoothed: bool, display smoothed spectrum if True
        :return peakVals: The x values (in units of Doppler velocity) of the marked peaks
        :return peakPowers: The y values (in units of dBZ) of the marked peaks
        """

        peakVals = []
        peakPowers = []
        # TODO replace with get_chirp_offsets
        n_rg = self.spec_data[n_file]['chirp_start_indices']
        c_ind = np.digitize(r_index, n_rg)
        # print(f'range index {r_index} is in chirp {c_ind} with ranges in chirps {n_rg[1:]}')

        heightindex_center = r_index
        timeindex_center = t_index
        this_spectrum_center = self.spec_data[n_file]['doppler_spectrum'][int(timeindex_center), int(heightindex_center), :]
        # print(f'time index center: {timeindex_center}, height index center: {heightindex_center}')
        if not np.sum(~np.isnan(this_spectrum_center.values)) < 2:
            velbins = self.spec_data[n_file]['velocity_vectors'][c_ind - 1, :]
            xlim = velbins.values[~np.isnan(this_spectrum_center.values) & ~(this_spectrum_center.values == 0)][[0, -1]]
            xlim += [-1, +1]
            # if this spectrum is not empty, we plot 3x3 panels with shared x and y axes
            fig, ax = plt.subplots(3, 3, figsize=[11, 11], sharex=True, sharey=True)
            fig.suptitle(f'Mark peaks in the center panel spectrum. Fig. {self.plot_count[n_file] + 1} out of '
                         f'{self.num_spec[n_file]}; File {n_file + 1} of {len(self.spec_data)}', size='xx-large',
                         fontweight='semibold')
            for dim1 in range(3):
                for dim2 in range(3):
                    if not (dim1 == 1 and dim2 == 1):  # if this is not the center panel plot
                        comment = ' '
                        heightindex = r_index - 1 + dim1
                        timeindex = t_index - 1 + dim2
                        if heightindex == self.spec_data[n_file]['doppler_spectrum'].shape[1]:
                            heightindex = heightindex - 1
                            comment = comment + ' (range boundary)'
                        if timeindex == self.spec_data[n_file]['doppler_spectrum'].shape[0]:
                            timeindex = timeindex - 1
                            comment = comment + ' (time boundary)'

                        thisSpectrum = self.spec_data[n_file]['doppler_spectrum'][int(timeindex), int(heightindex), :]

                        # print(f'time index: {timeindex}, height index: {heightindex}')
                        if heightindex == -1 or timeindex == -1:
                            thisSpectrum = thisSpectrum.where(thisSpectrum.values == -999)
                            comment = comment + ' (time or range boundary)'

                        ax[dim1, dim2].plot(velbins, lin2z(thisSpectrum.values))
                        ax[dim1, dim2].set_xlim(xlim)
                        ax[dim1, dim2].set_title(f'range:'
                                                 f'{np.round(self.spec_data[n_file]["range_layers"].values[int(heightindex)] / 1000, 2)} km,'
                                                 f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex)])}' + comment,
                                                 fontweight='semibold', fontsize=9, color='b')
                        ax[dim1, dim2].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
                        ax[dim1, dim2].set_ylabel("Reflectivity [dBZ]", fontweight='semibold', fontsize=9)
                        ax[dim1, dim2].grid(True)

            ax[1, 1].plot(velbins, lin2z(this_spectrum_center.values), label='raw')
            if plot_smoothed:
                assert 'span' in kwargs, "span required for mark_random_spectra if plot_smoothed is True"
                window_length = round_to_odd(kwargs['span'] / get_vel_resolution(velbins))
                smoothed_spectrum = lin2z(this_spectrum_center.values)
                if not window_length == 1:
                    smoothed_spectrum[~np.isnan(smoothed_spectrum)] = scipy.signal.savgol_filter(
                        smoothed_spectrum[~np.isnan(smoothed_spectrum)], window_length, polyorder=2, mode='nearest')
                ax[1, 1].plot(velbins, smoothed_spectrum, color='midnightblue', label='smoothed')

            ax[1, 1].set_xlabel("Doppler velocity [m/s]", fontweight='semibold', fontsize=9)
            ax[1, 1].set_ylabel("Reflectivity [dBZ m$^{-1}$s]", fontweight='semibold', fontsize=9)
            ax[1, 1].grid(True)
            ax[1, 1].legend()

            ax[1, 1].set_title(f'range:'
                               f'{np.round(self.spec_data[n_file]["range_layers"].values[int(heightindex_center)] / 1000, 2)} km,'
                               f' time: {format_hms(self.spec_data[n_file]["time"].values[int(timeindex_center)])}' +
                               comment, fontweight='semibold', fontsize=9, color='r')
            x = plt.ginput(self.max_peaks, timeout=0)
            # important in PyCharm:
            # uncheck Settings | Tools | Python Scientific | Show Plots in Toolwindow
            for i in range(len(x)):
                peakVals.append(x[i][0])
                peakPowers.append(x[i][1])
            plt.close()
            return peakVals, peakPowers
        else:
            return np.nan, np.nan

    def save_training_data(self):
        """
        save the marked peaks stored in TrainingData.training_data_out to a netcdf file.
        If the netcdf file does not exist yet, create it in place where spectra netcdf are stored.
        If the netcdf file does exist already, read it in, modify it and overwrite the file.
        """
        for i, dataset in enumerate(self.training_data_out):
            # Write a file if it does not exist yet
            if not os.path.isfile(self.peaks_ncfiles[i]):
                dataset.to_netcdf(self.peaks_ncfiles[i])
                print(f'created new file {self.peaks_ncfiles[i]}')
                return
            # Concatenate the 'spec' dimension if the file already exists
            with xr.open_dataset(self.peaks_ncfiles[i]) as data:
                existing_dataset = data.load()
            dataset = xr.concat([existing_dataset, dataset], dim='spec')
            dataset.to_netcdf(self.peaks_ncfiles[i], mode='w')
            print(f'updated file {self.peaks_ncfiles[i]}')
