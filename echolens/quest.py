import os
import healpy as hp
import numpy as np
import copy
import matplotlib.pyplot as plt

from plancklens import qest, qresp
from plancklens import nhl
from plancklens.filt import filt_simple,filt_cinv,filt_util


from echolens import simulation
from echolens.simulation import Mask
from echolens import mpi
from echolens import utils


class CMBbharatQE:

    def __init__(self,libdir,
                 nside,
                 fg_model,
                 lmin_cmb,
                 lmax_cmb,
                 lmax_recon,
                 fsky=0.8,
                 inc_fg=True,
                 ilc_bin=50,
                 inc_isw=False,
                 cache=True,
                 mask_apo=0.0,
                 filt_method='simple',
                 ):
        """
        Class to handle the quadratic estimator for CMB lensing reconstruction.

        :param libdir: Path to the directory where the data will be stored.
        :type libdir: str
        :param nside: Healpix resolution parameter.
        :type nside: int
        :param fg_model: Model for foregrounds.
        :type fg_model: list[str]
        :param lmin_cmb: Minimum multipole for CMB reconstruction.
        :type lmin_cmb: int
        :param lmax_cmb: Maximum multipole for CMB reconstruction.
        :type lmax_cmb: int
        :param lmax_recon: Maximum multipole for lensing reconstruction.
        :type lmax_recon: int
        :param fsky: Fraction of sky used for the analysis.
        :type fsky: float
        :param inc_fg: Include foregrounds in the simulations.
        :type inc_fg: bool
        :param ilc_bin: Bin width for ILC covariance.
        :type ilc_bin: int
        :param inc_isw: Include ISW in the simulations.
        :type inc_isw: bool
        :param cache: Cache the simulations.
        :type cache: bool
        :param mask_apo: Apodization scale for the mask.
        :type mask_apo: float
        :param filt_method: Filtering method.
        :type filt_method: str
        """
        
        self.filt_method = filt_method
        assert self.filt_method in ['simple','cinv'], "Invalid filtering method, choose between simple and cinv"
        
        self.qedir = os.path.join(libdir,f'qe_{self.filt_method}')
        if mpi.rank == 0:
            os.makedirs(self.qedir,exist_ok=True)
        mpi.barrier()

        self.nside = nside
        self.fsky = fsky
        self.lmin_cmb = lmin_cmb
        self.lmax_cmb = lmax_cmb
        self.lmax_recon = lmax_recon
        self.lmax = 3*nside - 1
        

        self.sims = simulation.CMBbharatSky(libdir,nside,fg_model,ilc_bin,inc_fg,inc_isw,cache)

        transf = np.ones(self.lmax_cmb + 1, dtype=float) 
        cl_len = simulation.CMBspectra().get_lensed_spectra(dl=False)
        self.cl_pp = simulation.CMBspectra().get_lens_potential(dl=False)['pp']

        cl_weight = copy.deepcopy(cl_len)
        cl_weight['bb'] *= 0.

        self.maskpath = None
        self.set_mask(mask_apo)

        noiseT, noiseP = self.sims.noise.spectra.eqv_noise(unit='muk-arcmin')

        if self.filt_method == 'simple':
            ftl = utils.cli(cl_len['tt'][:lmax_cmb + 1] + (noiseT / 60. / 180. * np.pi / transf) ** 2)
            fel = utils.cli(cl_len['ee'][:lmax_cmb + 1] + (noiseP / 60. / 180. * np.pi / transf) ** 2)
            fbl = utils.cli(cl_len['bb'][:lmax_cmb + 1] + (noiseP / 60. / 180. * np.pi / transf) ** 2)
            ftl[:lmin_cmb] *= 0.
            fel[:lmin_cmb] *= 0.
            fbl[:lmin_cmb] *= 0.

            ivfs_dir = os.path.join(self.qedir, 'ivfs')
            self.ivfs = filt_simple.library_apo_sepTP(ivfs_dir, self.sims, self.maskpath[0], cl_len, transf,ftl, fel, fbl)
        
        elif self.filt_method == 'cinv':
            libdir_cinvt = os.path.join(self.qedir, 'cinv_t')
            libdir_cinvp = os.path.join(self.qedir, 'cinv_p')
            libdir_ivfs  = os.path.join(self.qedir, 'ivfs')
            pixelarea = self.get_areapixel()
            ninv_t = [np.array([pixelarea / noiseT ** 2])] + self.maskpath
            cinv_t = filt_cinv.cinv_t(libdir_cinvt, self.lmax_cmb,nside, cl_len, transf, ninv_t,
                                    marge_monopole=True, marge_dipole=True, marge_maps=[])

            ninv_p = [[np.array([pixelarea / noiseP ** 2])] + self.maskpath]
            cinv_p = filt_cinv.cinv_p(libdir_cinvp, self.lmax_cmb, nside, cl_len, transf, ninv_p)

            ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, self.sims, cinv_t, cinv_p, cl_len)
            ftl = np.ones(self.lmax_cmb + 1, dtype=float) * (np.arange(self.lmax_cmb + 1) >= self.lmin_cmb) # rescaling or cuts. Here just a lmin cut
            fel = np.ones(self.lmax_cmb + 1, dtype=float) * (np.arange(self.lmax_cmb + 1) >= self.lmin_cmb)
            fbl = np.ones(self.lmax_cmb + 1, dtype=float) * (np.arange(self.lmax_cmb + 1) >= self.lmin_cmb)
            self.ivfs   = filt_util.library_ftl(ivfs_raw, self.lmax_cmb, ftl, fel, fbl)




        qe_dir = os.path.join(self.qedir, 'qlms')
        self.qlms = qest.library_sepTP(qe_dir, self.ivfs, self.ivfs,   cl_len['te'], nside, lmax_qlm=self.lmax_recon)

        nhl_dir = os.path.join(self.qedir, 'nhl')
        self.nhl = nhl.nhl_lib_simple(nhl_dir, self.ivfs, cl_weight, self.lmax_recon)

        qresp_dir = os.path.join(self.qedir, 'qresp')
        self.qresp = qresp.resp_lib_simple(qresp_dir, lmax_recon, cl_weight, cl_len,
                                 {'t': self.ivfs.get_ftl(), 'e':self.ivfs.get_fel(), 'b':self.ivfs.get_fbl()}, self.lmax_recon)


    def set_mask(self,apo):
        fsky = int(self.fsky*100)
        maskpath = os.path.join(self.qedir,f"mask_fsky{fsky}{'_apo'+str(apo) if apo > 0 else ''}.fits")
        Mask(nside=self.nside).get_mask(self.fsky,save=maskpath,apodize=True if apo > 0 else False,apo_scale=apo)
        self.maskpath = [maskpath]

    def get_areapixel(self):
        pixel_area_square_degrees = hp.nside2pixarea(self.nside,degrees=True)
        pixel_area_square_arcmin = pixel_area_square_degrees * 3600
        return pixel_area_square_arcmin
    
    def get_normalization(self,idx):
        qresp = self.qresp.get_response('p_p','p')
        return utils.cli(qresp)
    
    def get_unnormalized_qe(self,idx,key='p_p'):
        return self.qlms.get_sim_qlm(key,idx)
    
    def get_normalized_qe(self,idx,key='p_p'):
        qe_alm = self.get_unnormalized_qe(idx,key)
        normalization = self.get_normalization(idx)
        return hp.almxfl(qe_alm,normalization)
    
    def get_input_phi(self,idx):
        phi_alm = self.sims.cmb.get_phi_alms(idx)
        return phi_alm
    
    def cross_spectra(self,idx):
        phi_alm = self.get_input_phi(idx)
        qe_alm = self.get_normalized_qe(idx)
        phi_alm, qe_alm = utils.regularize_alms([phi_alm, qe_alm])
        return hp.alm2cl(phi_alm,qe_alm)
    
    def test_plot(self,idx):
        qlm = self.get_unnormalized_qe(idx)
        qresp = self.qresp.get_response('p_p','p')
        qnorm = utils.cli(qresp)
        n0 = self.nhl.get_sim_nhl(idx,'p_p','p_p')
        ell = np.arange(0,2048)
        w = lambda ell : ell ** 2 * (ell + 1.) ** 2 * 0.5 / np.pi * 1e7
        qcl = hp.alm2cl(qlm)[ell] * w(ell) * qnorm[ell]**2 / self.qlms.fsky12
        ncl = n0[ell] * w(ell) * qnorm[ell]**2
        input = self.get_input_phi(idx)
        input_cl = hp.alm2cl(input)[ell] * w(ell)
        inpXrec = self.cross_spectra(idx)
        plt.figure(figsize=(6,4))
        plt.loglog(ell,self.cl_pp[ell]*w(ell),label='Theory')
        plt.loglog(ell,qcl-ncl,label='QE-N0')
        plt.loglog(ell,ncl,label='N0')
        plt.loglog(ell,input_cl,label='Input')
        plt.loglog(ell,inpXrec[ell]*w(ell),label='Input x Recon')
        plt.legend()
        plt.xlim(2,2048)


    
    







