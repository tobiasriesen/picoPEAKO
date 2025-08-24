class PeakDetectionQualityMetric:
    def compute_metric(self, detected_peaks, reference_peaks, spectrum, velocity_bins):
        raise NotImplementedError("This method should be overridden by subclasses")
