from json import load
from os import path as os_path
from numpy import array, sqrt
from typing import List, Optional, Union

data_file = os_path.join(os_path.dirname(os_path.realpath(__file__)), "IM.json")


class CMB_Bharat:
    """
    A class to handle data related to CMB (Cosmic Microwave Background) frequencies, beam sizes, 
    and noise levels for a specific experiment.

    Attributes
    ----------
    data : list
        A list containing information about different CMB frequency channels.

    Methods
    -------
    get_frequency(idx=None):
        Returns the frequency of a particular channel or all channels.
    
    get_fwhm(idx=None, freq=None):
        Returns the beam Full Width at Half Maximum (FWHM) of a channel by index or frequency.
    
    get_noise_t(idx=None, freq=None):
        Returns the noise temperature (T) of a channel by index or frequency.
    
    get_noise_p(idx=None, freq=None):
        Returns the noise polarization (P) of a channel by index or frequency.
    """

    def __init__(self, data_file: str):
        """
        Initializes the CMB_Bharat class by loading data from the specified file.

        Parameters
        ----------
        data_file : str
            The path to the data file (in JSON format) containing CMB frequency channels and properties.
        """
        with open(data_file, "r") as file:
            data = json.load(file)
            self.data = data["data"]

    def get_frequency(self, idx: Optional[int] = None) -> Union[array, float]:
        """
        Returns the frequency for a specific channel or all channels.

        Parameters
        ----------
        idx : Optional[int], optional
            Index of the specific channel (default is None, which returns all frequencies).

        Returns
        -------
        Union[array, float]
            The frequency of the specified channel, or an array of frequencies for all channels.
        
        Raises
        ------
        IndexError
            If the provided index is out of range.
        """
        if idx is None:
            return array([entry["frequency"] for entry in self.data])
        if 0 <= idx < len(self.data):
            return self.data[idx]["frequency"]
        raise IndexError("Index out of range")

    def get_fwhm(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Returns the FWHM (beam size) for a specific channel or all channels.

        Parameters
        ----------
        idx : Optional[int], optional
            Index of the specific channel (default is None).
        freq : Optional[float], optional
            Frequency to search for (default is None).

        Returns
        -------
        Union[array, float]
            The FWHM of the specified channel, or an array of FWHM values for all channels.
        
        Raises
        ------
        IndexError
            If the provided index is out of range.
        ValueError
            If the provided frequency is not found.
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

        Parameters
        ----------
        idx : Optional[int], optional
            Index of the specific channel (default is None).
        freq : Optional[float], optional
            Frequency to search for (default is None).

        Returns
        -------
        Union[array, float]
            The noise temperature of the specified channel, or an array of noise temperatures for all channels.
        """
        return self.get_noise_p(idx, freq) / sqrt(2)

    def get_noise_p(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Returns the noise polarization for a specific channel or all channels.

        Parameters
        ----------
        idx : Optional[int], optional
            Index of the specific channel (default is None).
        freq : Optional[float], optional
            Frequency to search for (default is None).

        Returns
        -------
        Union[array, float]
            The noise polarization of the specified channel, or an array of noise polarization values for all channels.
        
        Raises
        ------
        IndexError
            If the provided index is out of range.
        ValueError
            If the provided frequency is not found.
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
