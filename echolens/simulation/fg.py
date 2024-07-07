import pysm3
import pysm3.units as u
import healpy as hp
from echolens import CMB_Bharat
import os
from tqdm import tqdm
import numpy as np

class Foregrounds:

    def __init__(self,folder,nside,fg_model):
        self.folder = os.path.join(folder,''.join(fg_model))
        os.makedirs(self.folder,exist_ok=True)
        self.nside = nside
        self.fg_model = fg_model
        self.freq = CMB_Bharat().get_frequency()
        self.sky = None
        self.lmax = 3*nside - 1

    
    def set_sky(self):
        self.sky = pysm3.Sky(nside=self.nside, preset_strings=self.fg_model)
    
    def get_fg_alm_v(self,freq):
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
        
    
    def get_fg_alms(self):
        fg = []
        for freq in tqdm(self.freq,desc='computing foregrounds',unit='Freq'):
            fg.append(self.get_fg_alm_v(freq))
        return np.array(fg)



class HILC:

    def __init__(self):
        pass

    def harmonic_ilc_alm(self,alms,lbins=None):
        A = np.ones((len(alms), 1))
        cov = self._empirical_harmonic_covariance(alms)
        if lbins is not None:
            for lmin, lmax in zip(lbins[:-1], lbins[1:]):
                # Average the covariances in the bin
                lmax = min(lmax, cov.shape[-1])
                dof = 2 * np.arange(lmin, lmax) + 1
                cov[..., lmin:lmax] = (
                    (dof / dof.sum() * cov[..., lmin:lmax]).sum(-1)
                    )[..., np.newaxis]
        cov = self._regularized_inverse(cov.swapaxes(-1, -3))
        ilc_filter = np.linalg.inv(A.T @ cov @ A) @ A.T @ cov
        del cov, dof
        result = self._apply_harmonic_W(ilc_filter, alms)
        return result


    def _empirical_harmonic_covariance(self,alms):
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


    def _regularized_inverse(self,cov):

        inv_std = np.einsum('...ii->...i', cov)
        inv_std = 1 / np.sqrt(inv_std)
        np.nan_to_num(inv_std, False, 0, 0, 0)
        np.nan_to_num(cov, False, 0, 0, 0)

        inv_cov = np.linalg.pinv(cov
                                * inv_std[..., np.newaxis]
                                * inv_std[..., np.newaxis, :])
        return inv_cov * inv_std[..., np.newaxis] * inv_std[..., np.newaxis, :]
    
    def _apply_harmonic_W(self,W,  # (..., ell, comp, freq)
                      alms):  # (freq, ..., lm)
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
        