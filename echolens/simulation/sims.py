"""
This module contains the class CMBbharatSky which is used to simulate the CMB observation sky with foregrounds and noise.
"""

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
from echolens import mpi
import toml
from typing import List, Optional, Union


class CMBbharatSky:
    """
    This class is used to simulate the CMB observation sky with foregrounds and noise.

    :param libdir: The directory where the simulation data is stored.
    :type libdir: str
    :param nside: The resolution of the HEALPix map.
    :type nside: int
    :param fg_model: The foreground model to be used in the simulation.
    :type fg_model: list
    :param inc_fg: If True, the simulation includes foregrounds.
    :type inc_fg: bool
    :param inc_isw: If True, the simulation includes the ISW effect.
    :type inc_isw: bool
    :param cache: If True, the simulation data is cached.
    :type cache: bool

    """

    def __init__(self, 
                 libdir : str,
                 nside : int,
                 fg_model : List[str],
                 ilc_bins : Optional[int] = 50,
                 inc_fg : Optional[bool] = False,
                 inc_isw : Optional[bool] = False,
                 cache : Optional[bool] = False) -> None:
        """
        Initializes the CMB_Bharat class by loading data from the specified file.

        :param libdir: The directory where the simulation data is stored.
        :type libdir: str
        :param nside: The resolution of the HEALPix map.
        :type nside: int
        :param fg_model: The foreground model to be used in the simulation.
        :type fg_model: list
        :param ilc_bins: The bin width for ILC covariance.
        :type ilc_bins: int
        :param inc_fg: If True, the simulation includes foregrounds.
        :type inc_fg: bool
        :param inc_isw: If True, the simulation includes the ISW effect.
        :type inc_isw: bool
        :param cache: If True, the simulation data is cached.
        :type cache: bool
        """
        self.fgdir = os.path.join(libdir,'foregrounds')
        self.__fg_str__ = ''.join(fg_model)
        self.__fg_inc_str__ = 'inc_fg' if inc_fg else 'no_fg'
        self.ilcdir = os.path.join(libdir,f'ilc_{self.__fg_inc_str__}' + (self.__fg_str__ if inc_fg else ''))
        if mpi.rank == 0:
            os.makedirs(self.ilcdir,exist_ok=True)
            os.makedirs(self.fgdir,exist_ok=True)
        mpi.barrier()
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
        self.nside = nside
        self.inc_isw = inc_isw
        self.ilc_bins = ilc_bins
    
    @classmethod
    def from_file(cls,config : str) -> 'CMBbharatSky':
        """
        This class method is used to create an instance of the CMBbharatSky class from a configuration file.

        :param config: The path to the configuration file.
        :type config: str

        """

        config = toml.load(config)['simulation']
        return cls(config['libdir'],
                config['nside'],
                config['fg_model'],
                config['inc_fg'],
                config['inc_isw'],
                config['cache'])

    
    def observed_cmb_alms(self,idx : int) -> tuple:
        """
        This method is used to get the observed CMB alms. It returns the Harmonic ILC results, both CMB and Noise.

        :param idx: The index of the simulation.
        :type idx: int

        :return: The CMB and Noise alms.
        :rtype: tuple
        """
        fname_cmb = os.path.join(self.ilcdir,f'ilc_cmb_b{self.ilc_bins}_{idx:03}.fits')
        fname_noise = os.path.join(self.ilcdir,f'ilc_noise_b{self.ilc_bins}_{idx:03}.fits')
        if os.path.isfile(fname_cmb) and os.path.isfile(fname_noise):
            return hp.read_alm(fname_cmb,hdu=(1,2,3)),hp.read_alm(fname_noise,hdu=(1,2,3))
        else:
            cmb = self.cmb.get_lensed_alms(idx)
            if self.inc_fg:
                fg = self.fg.get_fg_alms()
            noises = self.noise.noise_alms()
            if self.inc_fg:
                alms = cmb + fg + noises
            else:
                alms = cmb + noises

            lbins = np.arange(1000) * self.ilc_bins
            results, ilc_weights = self.hilc.harmonic_ilc_alm(alms,lbins=lbins)
            ilc_noise = self.hilc.apply_harmonic_W(ilc_weights, noises)
            if self.cache:
                hp.write_alm(fname_cmb,results[0])
                hp.write_alm(fname_noise,ilc_noise[0])

            return results[0],ilc_noise[0]
        
    def noise_alms(self,idx : int) -> list:
        """
        The attribute noise_alms is used to read the noise alms from the file.

        :param idx: The index of the simulation.
        :type idx: int

        :return: The noise alms.
        :rtype: list
        """
        fname_noise = os.path.join(self.ilcdir,f'ilc_noise_b{self.ilc_bins}_{idx:03}.fits')
        return hp.read_alm(fname_noise,hdu=(1,2,3))
    
    def noise_map(self,idx : int) -> list:
        """
        The attribute noise_map is used to read the noise map from the file.

        :param idx: The index of the simulation.
        :type idx: int

        :return: The noise map.
        :rtype: list
        """
        noise = self.noise_alms(idx)
        return hp.alm2map(noise,nside=self.nside)

    
    def hashdict(self):
        return {'sim_lib': "CMBbharatSky",
                'fg_model': self.fg.fg_model,
                'inc_fg': self.inc_fg,
                'inc_isw': self.inc_isw,}

    
    def get_sim_tlm(self,idx):
        fname_cmb = os.path.join(self.ilcdir,f'ilc_cmb_b{self.ilc_bins}_{idx:03}.fits')
        return hp.read_alm(fname_cmb,hdu=1)
        

    def get_sim_elm(self,idx):
        fname_cmb = os.path.join(self.ilcdir,f'ilc_cmb_b{self.ilc_bins}_{idx:03}.fits')
        return hp.read_alm(fname_cmb,hdu=2)

    def get_sim_blm(self,idx):
        fname_cmb = os.path.join(self.ilcdir,f'ilc_cmb_b{self.ilc_bins}_{idx:03}.fits')
        return hp.read_alm(fname_cmb,hdu=3)
    
    def get_sim_tmap(self,idx):
        tmap = self.get_sim_tlm(idx)
        tmap = hp.alm2map(tmap,self.nside)
        return tmap

    def get_sim_pmap(self,idx):
        elm = self.get_sim_elm(idx)
        blm = self.get_sim_blm(idx)
        Q,U = hp.alm2map_spin([elm,blm], self.nside, 2,hp.Alm.getlmax(elm.size))
        del elm,blm
        return Q ,U
    
    def make_sims(self,n : int) -> None:
        """
        The attribute make_sims is used to make the simulations.

        :param n: The number of simulations.
        :type n: int
        """
        if mpi.size < 2:
            for i in tqdm(range(n)):
                self.observed_cmb_alms(i)
        else:
            jobs = np.arange(n)
            for i in jobs[mpi.rank::mpi.size]:
                self.observed_cmb_alms(i)
        mpi.barrier()


        


    
            
