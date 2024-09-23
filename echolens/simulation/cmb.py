"""
This module provides the class to generate CMB lensed and unlensed maps and power spectra.
"""
import camb
import numpy as np
import os
import pickle as pl
import healpy as hp
import lenspyx
from echolens.utils import synalm_c2
from echolens import mpi
from typing import Optional, Union

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "echo.ini")
spectra = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")


class CMBspectra:
    """
    This class is used to get the CMB power spectra.
    """


    def __init__(self) -> None:
        """
        Initializes the CMBspectra class by loading the CMB power spectra.
        """
        if os.path.isfile(spectra):
            self.powers = pl.load(open(spectra, 'rb'))
        else:
            self.powers = self.compute_powers()


    def compute_powers(self) -> dict:
        """
        computes the CMB power spectra using the CAMB package.
        """
        params = camb.read_ini(ini_file)
        results = camb.get_results(params)
        powers = {}
        powers['cls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
        powers['dls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=False)
        pl.dump(powers, open(spectra, 'wb'))
        return powers
    
    def get_power(self, dl : bool = True) -> dict:
        """
        Returns the CMB power spectra.
        
        :param dl: If True, returns the lensed power spectra. If False, returns the unlensed power spectra.
        :type dl: bool
        :return: The CMB power spectra.
        :rtype: dict
        """

        return self.powers['dls'] if dl else self.powers['cls']
    

    def get_lensed_spectra(self, dl : bool = True, dtype : str = 'd') -> dict:
        """
        Returns the lensed CMB power spectra.

        :param dl: If True, returns the lensed power spectra. If False, returns the unlensed power spectra.
        :type dl: bool
        :param dtype: The type of the power spectra. If 'd', returns the power spectra as a dictionary. If 'a', returns the power spectra as an array.
        :type dtype: str
        :return: The lensed CMB power spectra.
        :rtype: dict or array
        """
        powers = self.get_power(dl)['lensed_scalar']
        if dtype == 'd':
            pow = {}
            pow['tt'] = powers[:, 0]
            pow['ee'] = powers[:, 1]
            pow['bb'] = powers[:, 2]
            pow['te'] = powers[:, 3]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_unlensed_spectra(self, dl : bool = True, dtype : str = 'd') -> dict:
        """
        Returns the unlensed CMB power spectra.

        :param dl: If True, returns the lensed power spectra. If False, returns the unlensed power spectra.
        :type dl: bool
        :param dtype: The type of the power spectra. If 'd', returns the power spectra as a dictionary. If 'a', returns the power spectra as an array.
        :type dtype: str
        :return: The unlensed CMB power spectra.
        :rtype: dict or array
        """

        powers = self.get_power(dl)['unlensed_scalar']
        if dtype == 'd':
            pow = {}
            pow['tt'] = powers[:, 0]
            pow['ee'] = powers[:, 1]
            pow['bb'] = powers[:, 2]
            pow['te'] = powers[:, 3]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
    
    def get_lens_potential(self, dl : bool = True, dtype : str = 'd') -> dict:
        """
        Returns the lens potential power spectra.

        :param dl: If True, returns the lensed power spectra. If False, returns the unlensed power spectra.
        :type dl: bool
        :param dtype: The type of the power spectra. If 'd', returns the power spectra as a dictionary. If 'a', returns the power spectra as an array.
        :type dtype: str
        :return: The lens potential power spectra.
        :rtype: dict or array
        """

        powers = self.get_power(dl)['lens_potential']
        if dtype == 'd':
            pow = {}
            pow['pp'] = powers[:, 0]
            pow['pt'] = powers[:, 1]
            pow['pe'] = powers[:, 2]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
        

class CMBlensed:
    """
    This class is used to generate lensed CMB maps.

    :param libdir: The directory to store the simulation data.
    :type libdir: str
    :param nside: The resolution of the maps.
    :type nside: int
    :param cache: If True, the simulation data is cached.
    :type cache: bool

    """

    def __init__(self,libdir : str,nside : Optional[int] = 1024,cache : Optional[bool] = True) -> None:
        self.cmbdir = os.path.join(libdir,'cmb')
        self.massdir = os.path.join(libdir,'mass')
        self.hilcdir = os.path.join(libdir,'hilc')
        if mpi.rank == 0:
            os.makedirs(self.cmbdir,exist_ok=True)
            os.makedirs(self.massdir,exist_ok=True)
            os.makedirs(self.hilcdir,exist_ok=True)
        mpi.barrier()
    
        self.spectra = CMBspectra()
        self.cl_unl = self.spectra.get_unlensed_spectra(dl=False,dtype='d')
        self.cl_pot = self.spectra.get_lens_potential(dl=False,dtype='d')
        self.cmbseeds = None
        self.phiseeds = None
        self.nside = nside
        self.lmax = (3 * nside) - 1
        self.set_seeds()
        self.lmax_len = self.lmax
        self.dlmax = 1024 
        self.epsilon = 1e-6 
        self.lmax_unl, self.mmax_unl = self.lmax_len + self.dlmax, self.lmax_len + self.dlmax
        assert self.lmax_unl <= len(self.cl_unl['tt'])-1, f"Spectra has not enough multipoles, increase lmax more than {self.lmax_unl}"
        self.cache = cache
    
    def set_seeds(self) -> None:
        """
        Sets the seeds for the simulation.
        """
        nos = 1000
        self.cmbseeds = np.arange(11111,11111+nos)
        self.phiseeds = np.arange(22222,22222+nos)

    
    def get_unlensed_alms(self,idx : int) -> np.ndarray:
        """
        Returns the unlensed alm's.

        :param idx: The index of the seed.
        :type idx: int
        :return: The unlensed alm's.
        :rtype: np.ndarray
        """
        Cls = [ self.cl_unl['tt'],
                self.cl_unl['ee'],
                self.cl_unl['bb']*0,
                self.cl_unl['te']]
        np.random.seed(self.cmbseeds[idx])
        alms = hp.synalm(Cls, lmax=self.lmax_unl, new=True,)
        return alms
    
    def get_phi_alms(self,idx : int) -> np.ndarray:
        """
        Returns the lens potential alm's.

        :param idx: The index of the seed.
        :type idx: int
        :return: The lens potential alm's.
        :rtype: np.ndarray
        """
        fname = os.path.join(self.massdir,f'phi_{idx:03}.fits')
        if not os.path.isfile(fname):
            np.random.seed(self.phiseeds[idx])
            alms = hp.synalm(self.cl_pot['pp'], lmax=self.lmax_unl)
            if self.cache:
                hp.write_alm(fname,alms)
            return alms
        else:
            return hp.read_alm(fname)
    
    def get_deflection(self,idx : int) -> np.ndarray:
        """
        Returns the deflection field.

        :param idx: The index of the seed.
        :type idx: int
        :return: The deflection field.
        :rtype: np.ndarray
        """

        alms = self.get_phi_alms(idx)
        return hp.almxfl(alms, np.sqrt(np.arange(self.lmax_unl + 1, dtype=float) * np.arange(1, self.lmax_unl + 2)), None, False)
    
    def get_lensed_TQU(self,idx : int) -> np.ndarray:
        """
        Returns the lensed TQU maps.

        :param idx: The index of the seed.
        :type idx: int
        :return: The lensed TQU maps.
        :rtype: np.ndarray
        """
        alms = self.get_unlensed_alms(idx)
        defl = self.get_deflection(idx)
        geom_info = ('healpix', {'nside':self.nside})
        Tlen, Qlen, Ulen = lenspyx.alm2lenmap([alms[0], alms[1]], defl, geometry=geom_info, verbose=1, epsilon=self.epsilon)
        del (alms, defl)
        return np.array([Tlen, Qlen, Ulen])
    
    def get_lensed_alms(self,idx : int) -> np.ndarray:
        """
        Returns the lensed alm's.

        :param idx: The index of the seed.
        :type idx: int
        :return: The lensed alm's.
        :rtype: np.ndarray
        """
        fname = os.path.join(self.cmbdir,f'lensed_{idx:03}.fits')
        if not os.path.isfile(fname):
            alms = hp.map2alm(self.get_lensed_TQU(idx),lmax=self.lmax)
            if self.cache:
                hp.write_alm(fname,alms)
            return alms
        else:
            return hp.read_alm(fname, hdu=(1,2,3))
    

class CMBlensedISW(CMBlensed):
    """
    This class is used to generate lensed CMB maps with ISW effect.

    :param libdir: The directory to store the simulation data.
    :type libdir: str
    :param nside: The resolution of the maps.
    :type nside: int
    :param cache: If True, the simulation data is cached.
    :type cache: bool
    """

    def __init__(self, libdir : str, nside : Optional[int] = 1024, cache : Optional[bool] = True) -> None:
        super().__init__(libdir, nside, cache)

    def get_unlensed_alms(self,idx):
        Cls = [ self.cl_unl['tt'],
                self.cl_unl['ee'],
                self.cl_unl['bb'],
                self.cl_unl['te']]
        np.random.seed(self.cmbseeds[idx])
        alms = hp.synalm(Cls, lmax=self.lmax_unl, new=True,)
        plm = synalm_c2(self.cl_unl['tt'], alms[0], 
                        self.cl_unl['ee'], alms[1], 
                        self.cl_pot['pp'], 
                        self.cl_pot['pt'], 
                        self.cl_pot['pe'])
        return alms, plm
    
    def get_deflection(self,alms):
        return hp.almxfl(alms, np.sqrt(np.arange(self.lmax_unl + 1, dtype=float) * np.arange(1, self.lmax_unl + 2)), None, False)
   
    def get_lensed_TQU(self,idx):
        alms,plm = self.get_unlensed_alms(idx)
        defl = self.get_deflection(plm)
        geom_info = ('healpix', {'nside':self.nside})
        Tlen, Qlen, Ulen = lenspyx.alm2lenmap([alms[0], alms[1]], defl, geometry=geom_info, verbose=1, epsilon=self.epsilon)
        del (alms, defl)
        return np.array([Tlen, Qlen, Ulen])      



        

            
    



