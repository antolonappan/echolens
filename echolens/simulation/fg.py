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

    
    def set_sky(self):
        self.sky = pysm3.Sky(nside=self.nside, preset_strings=self.fg_model)
    
    def get_fg_TQU_v(self,freq):
        fname = os.path.join(self.folder,f'fg_{freq}.fits')
        if not os.path.exists(fname):
            if self.sky is None:
                self.set_sky()
                print('PySM3 Sky is computed.')
            tqu = self.sky.get_emission(freq * u.GHz)
            tqu = tqu.to(u.uK_CMB, equivalencies=u.cmb_equivalencies(freq*u.GHz)).value
            hp.write_map(fname, tqu)
            return tqu
        else:
            return hp.read_map(fname,field=[0,1,2])
        
    
    def get_fg_TQU(self):
        fg = []
        for freq in tqdm(self.freq,desc='computing foregrounds',unit='Freq'):
            fg.append(self.get_fg_TQU_v(freq))
        return np.array(fg)




        