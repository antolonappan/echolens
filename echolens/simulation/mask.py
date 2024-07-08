import healpy as hp
import os
import numpy as np

mask_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "masks.fits")

class Mask:
    def __init__(self, nside=1024):
        self.nside = nside
        self.which = {60:0,70:1,80:2,90:3}

    def get_mask(self,fsky):
        mask = hp.read_map(mask_file, self.which[fsky])
        return hp.ud_grade(mask,self.nside)
    
    @staticmethod
    def apodize_mask(mask, scale, method='hybrid', mult_factor=3, min_factor=0.1):
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

