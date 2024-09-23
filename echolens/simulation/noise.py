"""
This module contains classes to simulate noise in the CMB maps.
"""
from echolens import CMB_Bharat
import numpy as np
import healpy as hp
from echolens.utils import arc2cl, cl2arc, ilcnoise
from typing import Optional
from echolens import utils


class NoiseSpectra:
    """
    This class is used to get the noise spectra for the CMB Bharat instrument model.

    :param lmax: The maximum multipole moment.
    :type lmax: int

    """

    def __init__(self,lmax : Optional[int] = 3071) -> None:
        self.lmax = lmax
        self.im = CMB_Bharat()
        
    def get_beam(self,idx : int) -> np.ndarray:
        """
        Returns the beam size for a specific channel.
        
        :param idx: Index of the specific channel.
        :type idx: int
        :return: The beam size for the specified channel.
        :rtype: np.ndarray
        """
        return hp.gauss_beam(np.radians(self.im.get_fwhm(idx=idx)/60), lmax=self.lmax)
    
    def noise_T_idx(self,idx : int,deconvolve : Optional[bool] = True) -> np.ndarray:
        """
        Returns the noise power spectrum for the temperature for a specific channel.

        :param idx: Index of the specific channel.
        :type idx: int
        :param deconvolve: If True, the beam is deconvolved from the noise power spectrum.
        :type deconvolve: Optional[bool], optional
        :return: The noise power spectrum for the temperature for the specified channel.
        :rtype: np.ndarray
        """
        if deconvolve:
            return (arc2cl(self.im.get_noise_t(idx=idx)) * np.ones(self.lmax+1)) * utils.cli(self.get_beam(idx)**2,overflow=True)
        else:
            return (arc2cl(self.im.get_noise_t(idx=idx)) * np.ones(self.lmax+1)) 
    
    def noise_P_idx(self,idx : int, deconvolve : Optional[bool] = True) -> np.ndarray:
        """
        Returns the noise power spectrum for the polarization for a specific channel.

        :param idx: Index of the specific channel.
        :type idx: int
        :param deconvolve: If True, the beam is deconvolved from the noise power spectrum.
        :type deconvolve: Optional[bool], optional
        :return: The noise power spectrum for the polarization for the specified channel.
        :rtype: np.ndarray
        """

        if deconvolve:
            return (arc2cl(self.im.get_noise_p(idx=idx)) * np.ones(self.lmax+1)) * utils.cli(self.get_beam(idx)**2,overflow=True)
        else:
            return (arc2cl(self.im.get_noise_p(idx=idx)) * np.ones(self.lmax+1))
    
    def noise_T(self) -> np.ndarray:
        """
        Returns the noise power spectrum for the temperature for all channels.

        :return: The noise power spectrum for the temperature for all channels.
        :rtype: np.ndarray
        """
        freqs = len(self.im.get_frequency())
        noise = np.zeros((freqs,self.lmax+1))
        for i in range(freqs):
            noise[i] = self.noise_T_idx(i)
        return noise
    
    def noise_P(self) -> np.ndarray:
        """
        Returns the noise power spectrum for the polarization for all channels.

        :return: The noise power spectrum for the polarization for all channels.
        :rtype: np.ndarray
        """
        freqs = len(self.im.get_frequency())
        noise = np.zeros((freqs,self.lmax+1))
        for i in range(freqs):
            noise[i] = self.noise_P_idx(i)
        return noise
    
    def noise_ilc(self) -> np.ndarray:
        """
        Returns the inverse noise power spectrum for the temperature and polarization.
        
        :return: The inverse noise power spectrum for the temperature and polarization.
        :rtype: np.ndarray
        """
        noise_t = self.noise_T()
        ilc_t = 1/np.sum(1/noise_t,axis=0)
        noise_p = self.noise_P()
        ilc_p = 1/np.sum(1/noise_p,axis=0)
        return np.array([ilc_t,ilc_p])
    
    def eqv_noise(self,unit='muk-arcmin') -> np.ndarray:
        """
        Returns the equivalent noise power spectrum for the temperature and polarization.

        :return: The equivalent noise power spectrum for the temperature and polarization.
        :rtype: np.ndarray
        """
    
        t = self.im.get_noise_t()
        T =  cl2arc(1/sum(1/arc2cl(t)))
        p = self.im.get_noise_p()
        P =  cl2arc(1/sum(1/arc2cl(p)))
        N = np.array([T,P])
        if unit == 'muk-arcmin':
            return N
        elif unit == 'muk':
            return np.radians(N/60)**2
        else:
            raise ValueError('unit must be muk-arcmin or muk')
    
    def eqv_beam(self) -> np.ndarray:
        """
        Returns the equivalent beam size for the temperature and polarization.

        :return: The equivalent beam size for the temperature and polarization.
        :rtype: np.ndarray
        """

        Nt,Np = self.noise_ilc()
        nnt = ilcnoise(self.im.get_noise_t())
        nnp = ilcnoise(self.im.get_noise_p())
        bl2t = np.radians(nnt/60)**2 / Nt
        bl2p = np.radians(nnp/60)**2 / Np
        return np.sqrt(np.array([bl2t,bl2p]))


class GaussianNoiseMap:
    """
    This class is used to generate Gaussian noise maps for the CMB Bharat instrument model.

    :param nside: The resolution parameter.
    :type nside: int
    :param decon: If True, the beam is deconvolved from the noise maps.
    :type decon: Optional[bool], optional

    """

    def __init__(self,nside : int,decon : Optional[bool] = True) -> None:
        self.nside = nside
        self.spectra = NoiseSpectra(lmax=3*nside-1)
        self.decon = decon

    
    def noise_alm_idx(self,idx : int) -> np.ndarray:
        """
        Returns the noise alm for a specific channel.

        :param idx: Index of the specific channel.
        :type idx: int
        :return: The noise alm for the specified channel.
        :rtype: np.ndarray
        """
        nlt = self.spectra.noise_T_idx(idx,deconvolve=self.decon)
        nlp = self.spectra.noise_P_idx(idx,deconvolve=self.decon)
        return np.array([hp.synalm(nlt,lmax=self.spectra.lmax),
                         hp.synalm(nlp,lmax=self.spectra.lmax),
                         hp.synalm(nlp,lmax=self.spectra.lmax)])
    
    def noise_alms(self) -> np.ndarray:
        """
        Returns the noise alms for all channels.

        :return: The noise alms for all channels.
        :rtype: np.ndarray
        """
        freqs = len(self.spectra.im.get_frequency())
        noise = []
        for i in range(freqs):
            noise.append(self.noise_alm_idx(i))
        return np.array(noise)

    
    def noiseTQU(self,idx : Optional[int] = None) -> np.ndarray:
        """
        Returns the noise maps for the temperature and polarization.
        
        :param idx: Index of the specific channel. If None, returns noise maps for all channels.
        :type idx: Optional[int], optional
        :return: The noise maps for the temperature and polarization.
        :rtype: np.ndarray
        """
        nlep = self.im.get_noise_p()
        depth_p =np.array(nlep)
        depth_i = depth_p/np.sqrt(2)
        pix_amin2 = 4. * np.pi / float(hp.nside2npix(self.nside)) * (180. * 60. / np.pi) ** 2
        sigma_pix_I = np.sqrt(depth_i ** 2 / pix_amin2)
        sigma_pix_P = np.sqrt(depth_p ** 2 / pix_amin2)
        npix = hp.nside2npix(self.nside)
        noise = np.random.randn(len(depth_i), 3, npix)
        noise[:, 0, :] *= sigma_pix_I[:, None]
        noise[:, 1, :] *= sigma_pix_P[:, None]
        noise[:, 2, :] *= sigma_pix_P[:, None]
        if idx is not None:
            return noise[idx]
        else:
            return noise
    
