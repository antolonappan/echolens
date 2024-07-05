from json import load
from os import path as os_path
from numpy import array, sqrt
from typing import List, Optional, Union

data_file = os_path.join(os_path.dirname(os_path.realpath(__file__)), "IM.json")


class CMB_Bharat:
    """
    A class to represent CMB Bharat data and provide methods to access various parameters.

    Attributes:
        data (List[dict]): List of dictionaries containing CMB Bharat data.
    """

    def __init__(self):
        """Initialize CMB_Bharat with data from a JSON file."""
        with open(data_file, "r") as file:
            data = load(file)
            self.data = data["data"]

    def get_frequency(self, idx: Optional[int] = None) -> Union[array, float]:
        """
        Get frequency values from the data.

        Args:
            idx (Optional[int]): Index to get a specific frequency. If None, returns all frequencies.

        Returns:
            Union[array, float]: Frequency value(s).
        
        Raises:
            IndexError: If the provided index is out of range.
        """
        if idx is None:
            return array([entry["frequency"] for entry in self.data])
        if 0 <= idx < len(self.data):
            return self.data[idx]["frequency"]
        raise IndexError("Index out of range")

    def get_fwhm(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Get full width at half maximum (FWHM) beam values from the data.

        Args:
            idx (Optional[int]): Index to get a specific FWHM value. If None, returns all FWHM values.
            freq (Optional[float]): Frequency to get a specific FWHM value.

        Returns:
            Union[array, float]: FWHM beam value(s).
        
        Raises:
            IndexError: If the provided index is out of range.
            ValueError: If the provided frequency is not found.
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
        Get noise temperature values from the data.

        Args:
            idx (Optional[int]): Index to get a specific noise temperature value. If None, returns all noise temperature values.
            freq (Optional[float]): Frequency to get a specific noise temperature value.

        Returns:
            Union[array, float]: Noise temperature value(s).
        """
        return self.get_noise_p(idx, freq) / sqrt(2)

    def get_noise_p(self, idx: Optional[int] = None, freq: Optional[float] = None) -> Union[array, float]:
        """
        Get noise polarization values from the data.

        Args:
            idx (Optional[int]): Index to get a specific noise polarization value. If None, returns all noise polarization values.
            freq (Optional[float]): Frequency to get a specific noise polarization value.

        Returns:
            Union[array, float]: Noise polarization value(s).
        
        Raises:
            IndexError: If the provided index is out of range.
            ValueError: If the provided frequency is not found.
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
