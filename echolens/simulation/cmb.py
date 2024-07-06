import camb
import numpy as np
import os
import pickle as pl

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
        self.cl_len = self.spectra.get_unlensed_spectra(dl=False,dtype='d')
        self.cl_pp = self.spectra.get_lens_potential(dl=False,dtype='d')
    
    def set_seeds(self):
        nos = 1000
        


        

            
    



