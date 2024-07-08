from .cmb import CMBspectra
from .noise import NoiseSpectra
from .cmb import CMBlensed
from .cmb import CMBlensedISW
from .noise import GaussianNoiseMap
from .fg import Foregrounds, HILC
from .mask import Mask
from echolens import CMB_Bharat
from tqdm import tqdm
import healpy as hp
import numpy as np
import os


class CMBbharatSky:

    def __init__(self,libdir,nside,fg_model,inc_fg=True,inc_isw=False,cache=True):
        self.fgdir = os.path.join(libdir,'foregrounds')
        fg_str = ''.join(fg_model)
        fg_inc_str = 'inc_fg' if inc_fg else 'no_fg'
        self.ilcdir = os.path.join(libdir,f'ilc_{fg_inc_str}' + (fg_str if inc_fg else ''))
        os.makedirs(self.ilcdir,exist_ok=True)
        os.makedirs(self.fgdir,exist_ok=True)
        if inc_isw:
            self.cmb = CMBlensedISW(libdir,nside,cache)
        else:
            self.cmb = CMBlensed(libdir,nside,cache)
        self.noise = GaussianNoiseMap(nside)
        self.fg = Foregrounds(self.fgdir,nside,fg_model)
        self.im = CMB_Bharat()
        self.inc_fg = inc_fg
        self.hilc = HILC()
        self.cache = cache

    
    def observed_cmb_alms(self,idx):
        fname_cmb = os.path.join(self.ilcdir,f'ilc_cmb_{idx:03}.fits')
        fname_noise = os.path.join(self.ilcdir,f'ilc_noise_{idx:03}.fits')
        if os.path.isfile(fname_cmb) and os.path.isfile(fname_noise):
            return hp.read_alm(fname_cmb,hdu=(1,2,3)),hp.read_alm(fname_noise,hdu=(1,2,3))
        else:
            cmb = self.cmb.get_lensed_alms(idx)
            if self.inc_fg:
                fg = self.fg.get_fg_alms()
            noises = self.noise.noise_alms()
            freqs = self.im.get_frequency()
            if self.inc_fg:
                alms = cmb + fg + noises
            else:
                alms = cmb + noises

            lbins = np.arange(1000) * 10
            results, ilc_weights = self.hilc.harmonic_ilc_alm(alms,lbins=lbins)
            ilc_noise = self.hilc.apply_harmonic_W(ilc_weights, noises)
            if self.cache:
                hp.write_alm(fname_cmb,results[0])
                hp.write_alm(fname_noise,ilc_noise[0])

            return results[0],ilc_noise[0]
        
    def noise_alms(self,idx):
        fname_noise = os.path.join(self.ilcdir,f'ilc_noise_{idx}.fits')
        return hp.read_alm(fname_noise,hdu=(1,2,3))
    
    def noise_map(self,idx):
        noise = self.noise_alms(idx)
        return hp.alm2map(noise,nside=self.nside)


    
            
