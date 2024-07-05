import camb
import numpy as np
import os
import pickle as pl

ini_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "echo.ini")
spectra = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spectra.pkl")


class CMBspetra:

    def __ini__(self):
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
    
    def get_power(self, dl=True, dtype=dict):
        powers = self.powers['dls'] if dl else self.powers['cls']
        
    



