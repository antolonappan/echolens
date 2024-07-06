from .cmb import CMBspectra
from .noise import NoiseSpectra
from .cmb import CMBlensed
from .noise import GaussianNoiseMap
from .fg import Foregrounds
from echolens import CMB_Bharat
from tqdm import tqdm
import healpy as hp
import numpy as np


class CMBbharatSky:

    def __init__(self,folder,nside,fg_model):
        self.cmb = CMBlensed(nside)
        self.noise = GaussianNoiseMap(nside)
        self.fg = Foregrounds(folder,nside,fg_model)
        self.im = CMB_Bharat()

    
    def observed_map(self,idx):
        cmb = self.cmb.get_lensed_map(idx)
        noise = self.noise.get_noiseTQU()
        fg = self.fg.get_fg_TQU()
        freqs = self.im.get_frequency()
        maps = []
        for v in tqdm(range(len(freqs)),desc='Total Sky',unit='Freq'):
            mapi = hp.smoothing(cmb + fg[v],fwhm=np.radians(self.im.get_fwhm(v)/60)) + noise[v]
            maps.append(mapi)
        return np.array(maps)
    
    def observed_alms(self,idx,deconvolve=False):
        maps = self.observed_map(idx)
        alms = []
        for mapi in tqdm(maps,desc='ALM',unit='Freq'):
            alm = hp.map2alm(mapi)
            if deconvolve:
                bl = hp.gauss_beam(np.radians(self.im.get_fwhm(idx)/60), lmax=self.cmb.lmax, pol=True).T
                hp.almxfl(alm[0],1/bl[0],inplace=True)
                hp.almxfl(alm[1],1/bl[1],inplace=True)
                hp.almxfl(alm[2],1/bl[2],inplace=True)
            alms.append(alm)
        return np.array(alms)
            
