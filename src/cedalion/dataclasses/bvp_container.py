"""Recording class definition for timeseries data."""

from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from cedalion.typing import NDTimeSeries

@dataclass
class BVP_Container:
    """Container for blood volume pulse analysis objects.

    The `bvp_container` class holds timeseries adjunct objects in ordered dictionaries,
    which were created by the bvp_waveform_analysis. It also holds the attributes
    "wav_storage_user" and "wav_storage_details", which contain all non-timeseries
    objects created by the bvp_waveform_analysis.

    Attributes:
        timeseries (OrderedDict[str, NDTimeSeries]): A dictionary of timeseries objects.
            The keys are the names of the timeseries.
        wav_storage_user (OrderedDict[str, Any]): A dictionary of objects most probaly
            used by users.
        wav_storage_details (OrderedDict[str, Any]): A dictionary of objects not that
            frequently used by users.
    """

    timeseries: OrderedDict[str, NDTimeSeries] = field(default_factory=OrderedDict)
    wav_storage_user: OrderedDict[str, Any] = field(default_factory=OrderedDict)
    wav_storage_details: OrderedDict[str, Any] = field(default_factory=OrderedDict)

    def __repr__(self):
        """Return a string representation of the bvp_container object."""

        if self.timeseries:
            text_timeseries = list(self.timeseries.keys())
        if not self.timeseries:
            text_timeseries = '---'

        if self.wav_storage_details:
            text_wav_storage_details = list(self.wav_storage_details[self.timeseries['bvp_ts'].channel.values[0]].keys())  # noqa: E501
        if not self.wav_storage_details:
            text_wav_storage_details = '---'

        if self.wav_storage_user:
            text_wav_storage_user = list(self.wav_storage_user[self.timeseries['bvp_ts'].channel.values[0]].keys())  # noqa: E501
        if not self.wav_storage_user:
            text_wav_storage_user = '---'

        return (
            f"BVP Container\n"
            f"  timeseries: {text_timeseries}\n"
            f"  wav_storage_user: {text_wav_storage_user}\n"
            f"  wav_storage_details: {text_wav_storage_details}"
        )

    def get_timeseries(self, key: Optional[str] = None) -> NDTimeSeries:
        """Get a timeseries object by key.

        Args:
            key (Optional[str]): The key of the timeseries to retrieve. If None, the
                last timeseries is returned.

        Returns:
            NDTimeSeries: The requested timeseries object.
        """
        if not self.timeseries:
            raise ValueError("timeseries dict is empty.")

        if key:
            return self.timeseries[key]
        else:
            last_key = list(self.timeseries.keys())[-1]

            return self.timeseries[last_key]

    # The main objects of interest are timeseries. Make them conveniently
    # accessible. rec[key] is a shortcut for rec.timeseries[key]
    def __getitem__(self, key):
        return self.get_timeseries(key)

    def __setitem__(self, key, value):
        return self.set_timeseries(key, value, overwrite=True)

    def set_timeseries(self, key: str, value: NDTimeSeries, overwrite: bool = False):
        if (overwrite is False) and (key in self.timeseries):
            raise ValueError(f"a timeseries with key '{key}' already exists!")

        self.timeseries[key] = value

    def get_timeseries_type(self, key):
        """Get the type of a timeseries.

        Args:
            key (str): The key of the timeseries.

        Returns:
            str: The type of the timeseries.
        """
        if key not in self.timeseries:
            raise KeyError(f"unknown timeseries '{key}'")

        if key == "bvp_ts" or key.startswith("bvp_"):
            return "blood volume pulse time series"
        elif key == "bvpa_ts" or key.startswith("bvpa_"):
            return "blood volume pulse amplitude time series"
        elif key == "pulse_rate_ts" or key.startswith("pulse_rate_"):
            return "pulse rate time series"
        else:
            raise ValueError(f"could not infer data type of timeseries '{key}'")

    @property
    def source_labels(self):
        """Get the unique source labels from the timeseries.

        Returns:
            list: A list of unique source labels.
        """
        labels = [
            ts.source.values for ts in self.timeseries.values() if "source" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def detector_labels(self):
        """Get the unique detector labels from the timeseries.

        Returns:
            list: A list of unique detector labels.
        """
        labels = [
            ts.detector.values
            for ts in self.timeseries.values()
            if "detector" in ts.coords
        ]
        return list(np.unique(np.hstack(labels)))

    @property
    def wavelengths(self):
        """Get the unique wavelengths from the timeseries.

        Returns:
            list: A list of unique wavelengths.
        """
        wl = [
            ts.wavelength.values
            for ts in self.timeseries.values()
            if "wavelength" in ts.coords
        ]
        return list(np.unique(np.hstack(wl)))
