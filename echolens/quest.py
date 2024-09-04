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
from echolens import mpi


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
        """Class to handle the quadratic estimator for CMB lensing reconstruction.

        Args:
            libdir (str): Path to the directory where the output files will be saved.
            nside (int): Resolution parameter of the HEALPix map.
            fg_model (str): The foreground model to be used.
            lmin_cmb (int): Minimum multipole to be considered in the CMB power spectrum.
            lmax_cmb (int): Maximum multipole to be considered in the CMB power spectrum.
            lmax_recon (int): Maximum multipole to be considered in the reconstruction.
            fsky (float, optional): Fraction of the sky to be considered. Defaults to 0.8.
            inc_fg (bool, optional): Include foregrounds in the simulation. Defaults to True.
            inc_isw (bool, optional): Include the ISW effect in the simulation. Defaults to False.
            cache (bool, optional): Cache the results. Defaults to True.
        """
           
        
        self.qedir = os.path.join(libdir,'qe')
        if mpi.rank == 0:
            os.makedirs(self.qedir,exist_ok=True)
        mpi.barrier()

        self.nside = nside
        self.fsky = fsky
        self.lmin_cmb = lmin_cmb
        self.lmax_cmb = lmax_cmb
        self.lmax_recon = lmax_recon
        self.lmax = 3*nside - 1

        self.sims = simulation.CMBbharatSky(libdir,nside,fg_model,inc_fg,inc_isw,cache)

        theory_bl = simulation.NoiseSpectra(lmax=self.lmax).eqv_beam()
        transf =theory_bl * hp.pixwin(nside)[:self.lmax_cmb + 1]

        cl_len = simulation.CMBspectra().get_lensed_spectra()

        cl_weight = copy.deepcopy(cl_len)
        cl_weight['bb'] *= 0.

        self.maskpath = None
        self.set_mask()

        libdir_cinvt = os.path.join(self.qedir, 'cinv_t')
        ninv_t = [self.sims.inv_noise_map_fname(50,'t')] + [self.maskpath] 
        cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_cmb,nside, cl_len, transf, ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])



        libdir_cinvp = os.path.join(self.qedir, 'cinv_p')
        ninv_p = [self.sims.inv_noise_map_fname(50,'p')] + [self.maskpath]
        cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_cmb, nside, cl_len, transf, ninv_p)


        libdir_ivfs  = os.path.join(self.qedir, 'ivfs')
        ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, self.sims, cinv_t, cinv_p, cl_len)

        ftl = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb)
        fel = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb)
        fbl = np.ones(lmax_cmb + 1, dtype=float) * (np.arange(lmax_cmb + 1) >= lmin_cmb)
        self.ivfs   = filt_util.library_ftl(ivfs_raw, lmax_cmb, ftl, fel, fbl)

        qe_dir = os.path.join(self.qedir, 'qlms')
        self.qlms = qest.library_sepTP(qe_dir, self.ivfs, self.ivfs,   cl_len['te'], nside, lmax_qlm=self.lmax_recon)

        nhl_dir = os.path.join(self.qedir, 'nhl')
        self.nhl = nhl.nhl_lib_simple(nhl_dir, self.ivfs, cl_weight, self.lmax_recon)

        qresp_dir = os.path.join(self.qedir, 'qresp')
        self.qresp = qresp.resp_lib_simple(qresp_dir, lmax_recon, cl_weight, cl_len,
                                 {'t': self.ivfs.get_ftl(), 'e':self.ivfs.get_fel(), 'b':self.ivfs.get_fbl()}, self.lmax_recon)


    def set_mask(self):
        fsky = int(self.fsky*100)
        maskpath = os.path.join(self.qedir,f'mask_fsky{fsky}.fits')
        Mask(nside=self.nside).get_mask(self.fsky,save=maskpath)
        self.maskpath = maskpath







