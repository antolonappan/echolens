"""
This module is used to generate foregrounds using PySM3 and perform harmonic ILC on the foregrounds.
"""
import pysm3
import pysm3.units as u
import healpy as hp
from echolens import CMB_Bharat
import os
from tqdm import tqdm
import numpy as np
from numba import jit, njit
from echolens import mpi
from typing import Optional, List

class Foregrounds:
    """
    This class is used to generate foregrounds using PySM3

    :param folder: The directory where the foreground data is stored.
    :type folder: str
    :param nside: The resolution of the HEALPix map.
    :type nside: int
    :param fg_model: The foreground model to be used in the simulation.
    :type fg_model: list

    """

    def __init__(self, folder : str, nside : int, fg_model : List[str]) -> None:
        self.folder = os.path.join(folder,''.join(fg_model))
        if mpi.rank == 0:
            os.makedirs(self.folder,exist_ok=True)
        mpi.barrier()
        self.nside = nside
        self.fg_model = fg_model
        self.freq = CMB_Bharat().get_frequency()
        self.sky = None
        self.lmax = 3*nside - 1

    
    def set_sky(self) -> None:
        """
        This method is used to set the PySM3 Sky object.
        """
        self.sky = pysm3.Sky(nside=self.nside, preset_strings=self.fg_model)
    
    def get_fg_alm_v(self,freq : float) -> np.ndarray:
        """
        This method is used to get the foregrounds in alm format for a specific frequency.

        :param freq: The frequency of the foregrounds.
        :type freq: float
        :return: The foregrounds in alm format.
        :rtype: np.ndarray
        """
        fname = os.path.join(self.folder,f'fg_{freq}.fits')
        if not os.path.exists(fname):
            if self.sky is None:
                self.set_sky()
                print('PySM3 Sky is computed.')
            tqu = self.sky.get_emission(freq * u.GHz)
            tqu = tqu.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz)).value
            alm = hp.map2alm(tqu,lmax=self.lmax)
            del tqu
            hp.write_alm(fname,alm)
            return alm
        else:
            return hp.read_alm(fname, hdu=(1,2,3))
        
    
    def get_fg_alms(self) -> np.ndarray:
        """
        This method is used to get the foregrounds in alm format for all frequencies.

        :return: The foregrounds in alm format.
        :rtype: np.ndarray
        """
        fg = []
        for freq in tqdm(self.freq,desc='computing foregrounds',unit='Freq'):
            fg.append(self.get_fg_alm_v(freq))
        return np.array(fg)



class HILC:
    """
    This class is used to perform Harmonic ILC on the foregrounds.

    """

    def __init__(self):
        pass

    def harmonic_ilc_alm(self, alms : np.ndarray, lbins : Optional[List[int]] = None) -> tuple:
        """
        This method is used to perform Harmonic ILC on the foregrounds.

        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray
        :param lbins: The list of ell bins.
        :type lbins: Optional[List[int]], optional
        :return: The Harmonic ILC results and the ILC filter.
        :rtype: tuple
        """

        A = np.ones((len(alms), 1))
        cov = self.empirical_harmonic_covariance(alms)
        if lbins is not None:
            for lmin, lmax in zip(lbins[:-1], lbins[1:]):
                # Average the covariances in the bin
                lmax = min(lmax, cov.shape[-1])
                dof = 2 * np.arange(lmin, lmax) + 1
                cov[..., lmin:lmax] = (
                    (dof / dof.sum() * cov[..., lmin:lmax]).sum(-1)
                    )[..., np.newaxis]
        cov = self.regularized_inverse(cov.swapaxes(-1, -3))
        ilc_filter = np.linalg.inv(A.T @ cov @ A) @ A.T @ cov
        del cov, dof
        result = self.apply_harmonic_W(ilc_filter, alms)
        return result,ilc_filter

    def empirical_harmonic_covariance(self,alms : np.ndarray) -> np.ndarray:
        """
        The method empirical_harmonic_covariance is used to compute the empirical harmonic covariance.

        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray
        :return: The empirical harmonic covariance.
        :rtype: np.ndarray
        """
        alms = np.array(alms, copy=False, order='C')
        alms = alms.view(np.float64).reshape(alms.shape+(2,))
        if alms.ndim > 3:  # Shape has to be ([Stokes], freq, lm, ri)
            alms = alms.transpose(1, 0, 2, 3)
        lmax = hp.Alm.getlmax(alms.shape[-2])

        res = (alms[..., np.newaxis, :, :lmax+1, 0]
            * alms[..., :, np.newaxis, :lmax+1, 0])  # (Stokes, freq, freq, ell)


        consumed = lmax + 1
        for i in range(1, lmax+1):
            n_m = lmax + 1 - i
            alms_m = alms[..., consumed:consumed+n_m, :]
            res[..., i:] += 2 * np.einsum('...fli,...nli->...fnl', alms_m, alms_m)
            consumed += n_m

        res /= 2 * np.arange(lmax + 1) + 1
        return res

    def regularized_inverse(self,cov : np.ndarray) -> np.ndarray:
        """
        The method regularized_inverse is used to compute the regularized inverse of the covariance.

        :param cov: The covariance.
        :type cov: np.ndarray
        :return: The regularized inverse of the covariance.
        :rtype: np.ndarray
        """

        inv_std = np.einsum('...ii->...i', cov)
        inv_std = 1 / np.sqrt(inv_std)
        np.nan_to_num(inv_std, False, 0, 0, 0)
        np.nan_to_num(cov, False, 0, 0, 0)

        inv_cov = np.linalg.pinv(cov
                                * inv_std[..., np.newaxis]
                                * inv_std[..., np.newaxis, :])
        return inv_cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]
    

    def apply_harmonic_W(self,W : np.ndarray, alms: np.ndarray)  ->  np.ndarray:
        """
        The method apply_harmonic_W is used to apply the harmonic weights.

        :param W: The harmonic weights.
        :type W: np.ndarray
        :param alms: The foregrounds in alm format.
        :type alms: np.ndarray

        :return: The alms after applying the harmonic weights.
        :rtype: np.ndarray
        """
    
        lmax = hp.Alm.getlmax(alms.shape[-1])
        res = np.full((W.shape[-2],) + alms.shape[1:], np.nan, dtype=alms.dtype)
        start = 0
        for i in range(0, lmax+1):
            n_m = lmax + 1 - i
            res[..., start:start+n_m] = np.einsum('...lcf,f...l->c...l',
                                                W[..., i:, :, :],
                                                alms[..., start:start+n_m])
            start += n_m
        return res
