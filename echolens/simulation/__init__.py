from .cmb import CMBspectra
from .noise import NoiseSpectra
from .cmb import CMBlensed
from .noise import GaussianNoiseMap
from .fg import Foregrounds, HILC
from echolens import CMB_Bharat
from tqdm import tqdm
import healpy as hp
import numpy as np


class CMBbharatSky:

    def __init__(self,folder,nside,fg_model,inc_fg=True):
        self.cmb = CMBlensed(nside)
        self.noise = GaussianNoiseMap(nside)
        self.fg = Foregrounds(folder,nside,fg_model)
        self.im = CMB_Bharat()
        self.inc_fg = inc_fg
        self.hilc = HILC()

    
    def observed_alms(self,idx):
        cmb = self.cmb.get_lensed_alms(idx)
        fg = self.fg.get_fg_alms()
        freqs = self.im.get_frequency()
        alms = []
        for v in tqdm(range(len(freqs)),desc='Total Sky',unit='Freq'):
            if self.inc_fg:
                alms.append(cmb + fg[v] + self.noise.noise_alm_idx(v))
            else:
                alms.append(cmb + self.noise.noise_alm_idx(v))
        return np.array(alms)
    
    def observed_cmb_alms(self,idx):
        alms = self.observed_alms(idx)
        lbins = np.arange(1000) * 10
        return self.hilc.harmonic_ilc_alm(alms,lbins=lbins)
    
            
