import camb
import numpy as np
import os
import pickle as pl
import healpy as hp
import lenspyx

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "echo.ini")
spectra = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")


class CMBspectra:

    def __init__(self):
        if os.path.isfile(spectra):
            self.powers = pl.load(open(spectra, 'rb'))
        else:
            self.powers = self.compute_powers()


    def compute_powers(self):
        params = camb.read_ini(ini_file)
        results = camb.get_results(params)
        powers = {}
        powers['cls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=True)
        powers['dls'] = results.get_cmb_power_spectra(params, CMB_unit='muK', raw_cl=False)
        pl.dump(powers, open(spectra, 'wb'))
        return powers
    
    def get_power(self, dl=True):
        return self.powers['dls'] if dl else self.powers['cls']
    

    def get_lensed_spectra(self, dl=True,dtype='d'):
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
    
    def get_unlensed_spectra(self, dl=True,dtype='d'):
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
    
    def get_lens_potential(self, dl=True,dtype='d'):
        powers = self.get_power(dl)['lens_potential']
        if dtype == 'd':
            pow = {}
            pow['pp'] = powers[:, 0]
            return pow
        elif dtype == 'a':
            return powers
        else:
            raise ValueError("dtype should be 'd' or 'a'")
        

class CMBlensed:

    def __init__(self,nside=1024):
        self.spectra = CMBspectra()
        self.cl_unl = self.spectra.get_unlensed_spectra(dl=False,dtype='d')
        self.cl_pp = self.spectra.get_lens_potential(dl=False,dtype='d')
        self.cmbseeds = None
        self.phiseeds = None
        self.nside = nside
        self.lmax = 3*nside-1
        self.set_seeds()
        self.lmax_len = self.lmax
        self.dlmax = 1024 
        self.epsilon = 1e-6 
        self.lmax_unl, self.mmax_unl = self.lmax_len + self.dlmax, self.lmax_len + self.dlmax
        assert self.lmax_unl <= len(self.cl_unl['tt'])-1, f"Spectra has not enough multipoles, increase lmax more than {self.lmax_unl}"
    
    def set_seeds(self):
        nos = 1000
        self.cmbseeds = np.arange(1111,1111+nos)
        self.phiseeds = np.arange(2222,2222+nos)

    
    def get_unlensed_alms(self,idx):
        Cls = [ self.cl_unl['tt'],
                self.cl_unl['ee'],
                self.cl_unl['bb']*0,
                self.cl_unl['te']]
        np.random.seed(self.cmbseeds[idx])
        alms = hp.synalm(Cls, lmax=self.lmax_unl, new=True,)
        return alms
    
    def get_phi_alms(self,idx):
        np.random.seed(self.phiseeds[idx])
        alms = hp.synalm(self.cl_pp['pp'], lmax=self.lmax_unl)
        return alms
    
    def get_deflection(self,idx):
        alms = self.get_phi_alms(idx)
        return hp.almxfl(alms, np.sqrt(np.arange(self.lmax_unl + 1, dtype=float) * np.arange(1, self.lmax_unl + 2)), None, False)
    
    def get_lensed_TQU(self,idx):
        alms = self.get_unlensed_alms(idx)
        defl = self.get_deflection(idx)
        geom_info = ('healpix', {'nside':self.nside})
        Tlen, Qlen, Ulen = lenspyx.alm2lenmap([alms[0], alms[1]], defl, geometry=geom_info, verbose=1, epsilon=self.epsilon)
        del (alms, defl)
        return np.array([Tlen, Qlen, Ulen])
    
    def get_lensed_alms(self,idx):
        maps = self.get_lensed_TQU(idx)
        return hp.map2alm(maps,lmax=self.lmax)




        

            
    



