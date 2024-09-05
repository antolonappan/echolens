"""
This module contains the CMB_Bharat class, which is used to get the Instrument Model parameters such as frequency, beam size, and noise levels.

The data is loaded from a JSON file, which is stored in the same directory as this file.

Example
-------
To get the frequency of the first channel:

.. code-block:: python
    >>> from echolens.instrument import CMB_Bharat
    >>> instrument = CMB_Bharat()
    >>> frequency = instrument.get_frequency(0)
"""

from json import load
from os import path as os_path
from numpy import array, sqrt
from typing import List, Optional, Union

data_file = os_path.join(os_path.dirname(os_path.realpath(__file__)), "IM.json")


class CMB_Bharat:
    """
    A class to handle data related to CMB Bharat frequencies, beam sizes, 
    and noise levels.
    """

    def __init__(self):
        """
        Initializes the CMB_Bharat class by loading data from the specified file.
        """

        with open(data_file, "r") as file:
            data = load(file)
            self.data = data["data"]

    def get_frequency(self, idx: Optional[int] = None) -> Union[array, float]:
        """
        Returns the frequency for a specific channel or all channels.

        :param idx: Index of the specific channel. If None, returns frequencies for all channels.
        :type idx: Optional[int], optional
        :return: The frequency of the specified channel or an array of frequencies for all channels.
        :rtype: Union[array, float]
        :raises IndexError: If the provided index is out of range.
        """
        if idx is None:
            return array([entry["frequency"] for entry in self.data])
        if 0 <= idx < len(self.data):
            return self.data[idx]["frequency"]
        raise IndexError("Index out of range")

    def get_fwhm(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Returns the FWHM (beam size) for a specific channel or all channels.

        :param idx: Index of the specific channel. If None, it checks the frequency.
        :type idx: Optional[int], optional
        :param freq: Frequency to search for. If None, returns FWHM for all channels.
        :type freq: Optional[float], optional
        :return: The FWHM of the specified channel or an array of FWHM values for all channels.
        :rtype: Union[array, float]
        :raises IndexError: If the provided index is out of range.
        :raises ValueError: If the provided frequency is not found.
        """
        if idx is not None:
            if 0 <= idx < len(self.data):
                return self.data[idx]["beam"]
            raise IndexError("Index out of range")
        if freq is not None:
            for entry in self.data:
                if entry["frequency"] == freq:
                    return entry["beam"]
            raise ValueError("Frequency not found")
        return array([entry["beam"] for entry in self.data])

    def get_noise_t(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Returns the noise temperature for a specific channel or all channels.

        :param idx: Index of the specific channel. If None, it checks the frequency.
        :type idx: Optional[int], optional
        :param freq: Frequency to search for. If None, returns noise temperatures for all channels.
        :type freq: Optional[float], optional
        :return: The noise temperature of the specified channel or an array of noise temperatures for all channels.
        :rtype: Union[array, float]
        """
        return self.get_noise_p(idx, freq) / sqrt(2)

    def get_noise_p(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Returns the noise polarization for a specific channel or all channels.

        :param idx: Index of the specific channel. If None, it checks the frequency.
        :type idx: Optional[int], optional
        :param freq: Frequency to search for. If None, returns noise polarization for all channels.
        :type freq: Optional[float], optional
        :return: The noise polarization of the specified channel or an array of noise polarization values for all channels.
        :rtype: Union[array, float]
        :raises IndexError: If the provided index is out of range.
        :raises ValueError: If the provided frequency is not found.
        """
        if idx is not None:
            if 0 <= idx < len(self.data):
                return self.data[idx]["nlev_p"]
            raise IndexError("Index out of range")
        if freq is not None:
            for entry in self.data:
                if entry["frequency"] == freq:
                    return entry["nlev_p"]
            raise ValueError("Frequency not found")
        return array([entry["nlev_p"] for entry in self.data])