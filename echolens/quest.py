import os
import healpy as hp
import numpy as np
import copy

import plancklens
from plancklens.filt import filt_simple, filt_util
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import planck2018_sims, phas, maps, utils as maps_utils
from plancklens.filt import filt_cinv, filt_util


from echolens import simulation
from echolens import Mask


class CMBbharatQE:

    def __init__(self,libdir,
                 nside,
                 fg_model,
                 lmin_cmb,
                 lmax_cmb,
                 lmax_recon,
                 fsky=0.8,
                 inc_fg=True,
                 inc_isw=False,
                 cache=True,
                 
                 ):
        
        self.qedir = os.path.join(libdir,'qe')
        os.makedirs(self.qedir,exist_ok=True)
        self.sims = simulation.CMBbharatSky(libdir,nside,fg_model,inc_fg,inc_isw,cache)
        self.lmax = 3*nside-1
        theory_bl = simulation.NoiseSpectra(lmax=self.lmax).eqv_beam()
        transf =theory_bl * hp.pixwin(nside)[:self.lmax_ivf + 1]

        cl_len = simulation.CMBspectra(lmax=self.lmax).get_lensed_spectra()

        cl_weight = copy.deepcopy(cl_len)
        cl_weight['bb'] *= 0.

        self.maskpath = None
        self.set_mask()

        libdir_cinvt = os.path.join(self.qedir, 'cinv_t')
        libdir_cinvp = os.path.join(self.qedir, 'cinv_p')
        libdir_ivfs  = os.path.join(self.qedir, 'ivfs')

        ninv_t = [something + self.maskpath] # TODO
        cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_cmb,nside, cl_len, transf, ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])

        ninv_p = [something + self.maskpath] # TODO
        cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_cmb, nside, cl_len, transf, ninv_p)

        ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, self.sims, cinv_t, cinv_p, cl_len)
        ftl = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb) # rescaling or cuts. Here just a lmin cut
        fel = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb)
        fbl = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb)
        self.ivfs   = filt_util.library_ftl(ivfs_raw, lmax_cmb, ftl, fel, fbl)


    def set_mask(self):
        fsky = int(self.fsky*100)
        maskpath = os.path.join(self.qedir,f'mask_fsky{fsky}.fits')
        Mask(nside=self.nside).get_mask(self.fsky,save=maskpath)
        self.maskpath = maskpath







