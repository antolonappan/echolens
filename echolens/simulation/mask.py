"""
This module provides a class to handle the masks used in the simulations.
"""
import healpy as hp
import os
import numpy as np
from typing import Optional

mask_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "masks.fits")

class Mask:
    """
    This class is used to handle the masks used in the simulations.

    :param nside: The resolution of the HEALPix map.
    :type nside: int

    """
    def __init__(self,nside : int) -> None:
        self.nside = nside
        self.which = {60:0,70:1,80:2,90:3}

    def get_mask(self,fsky : float,save : str = None) -> np.ndarray:
        """
        The method get_mask is used to get the mask for a specific fsky.

        :param fsky: The fraction of the sky to be masked.
        :type fsky: float
        :param save: The path to save the mask.
        :type save: str, optional
        :return: The mask for the specified fsky.
        :rtype: np.ndarray
        :raises ValueError: If the fsky is not valid.
        """
        fsky = int(fsky)
        if fsky not in [60,70,80,90]:
            raise ValueError('Invalid fsky. Choose between 60,70,80,90.')
        mask = hp.read_map(mask_file, self.which[fsky])
        if save:
            hp.write_map(save,mask)
        else:
            return hp.ud_grade(mask,self.nside)
    
    @staticmethod
    def apodize_mask(mask : np.ndarray, 
                     scale : float, 
                     method : Optional[str] = 'hybrid',
                     mult_factor : Optional[float] = 0.1,
                     min_factor : Optional[float] = 0.1) -> np.ndarray:
        """
        The method apodize_mask is used to apodize the mask.

        :param mask: The mask to be apodized.
        :type mask: np.ndarray
        :param scale: The scale of the apodization.
        :type scale: float
        :param method: The method of apodization. Choose between 'gaussian' and 'hybrid'.
        :type method: str, optional
        :param mult_factor: The multiplication factor for the hybrid method.
        :type mult_factor: float, optional
        :param min_factor: The minimum factor for the hybrid method.
        :type min_factor: float, optional
        :return: The apodized mask.
        :rtype: np.ndarray
        :raises ValueError: If the method is not valid.
        """

        sigma_rad = np.radians(scale)
        ap_mask = hp.smoothing(mask, sigma=sigma_rad)
        if method == 'gaussian': 
            return ap_mask
        elif method == 'hybrid': 
            ap_mask = 1 - np.minimum(1., np.maximum(0., mult_factor * (1 - ap_mask) - min_factor))
            ap_mask = hp.smoothing(ap_mask, sigma=sigma_rad / 2,)
        else:
            raise ValueError('Invalid apodization method. Choose between "gaussian" and "hybrid".')
        return ap_mask

