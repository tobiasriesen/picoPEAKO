import numpy as np
import xarray as xr


class PeakoModel:
    def __init__(self, data):
        self.data = data

    def compute_mean(self):
        return np.mean(self.data)

    def compute_std(self):
        return np.std(self.data)

    def to_xarray(self):
        return xr.DataArray(self.data)


def process_data(data):
    analyzer = PeakoModel(data)
    mean = analyzer.compute_mean()
    std_dev = analyzer.compute_std()
    xarray_data = analyzer.to_xarray()
    return mean, std_dev, xarray_data

