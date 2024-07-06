from echolens import CMB_Bharat
import numpy as np
import healpy as hp

def arc2cl(arc):
    return np.radians(arc/60)**2
def cl2arc(cl):
    return np.rad2deg(np.sqrt(cl))*60


class NoiseSpectra:

    def __init__(self,lmax=4096) -> None:
        self.lmax = lmax
        self.im = CMB_Bharat()
        
    def get_beam(self,idx):
        return hp.gauss_beam(np.radians(self.im.get_fwhm(idx=idx)/60), lmax=self.lmax)
    
    def noise_T_idx(self,idx):
        return (arc2cl(self.im.get_noise_t(idx=idx)) * np.ones(self.lmax+1)) / self.get_beam(idx)**2
    
    def noise_P_idx(self,idx):
        return (arc2cl(self.im.get_noise_p(idx=idx)) * np.ones(self.lmax+1)) / self.get_beam(idx)**2
    
    def noise_T(self):
        freqs = len(self.im.get_frequency())
        noise = np.zeros((freqs,self.lmax+1))
        for i in range(freqs):
            noise[i] = self.noise_T_idx(i)
        return noise
    
    def noise_P(self):
        freqs = len(self.im.get_frequency())
        noise = np.zeros((freqs,self.lmax+1))
        for i in range(freqs):
            noise[i] = self.noise_P_idx(i)
        return noise
    
    def noise_ilc(self):
        noise_t = self.noise_T()
        ilc_t = 1/np.sum(1/noise_t,axis=0)
        noise_p = self.noise_P()
        ilc_p = 1/np.sum(1/noise_p,axis=0)
        return np.array([ilc_t,ilc_p])
    
    def eqv_noise(self):
        t = self.im.get_noise_t()
        T =  cl2arc(1/sum(1/arc2cl(t)))
        p = self.im.get_noise_p()
        P =  cl2arc(1/sum(1/arc2cl(p)))
        return np.array([T,P])

class GaussianNoiseMap:

    def __init__(self,nside=1024) -> None:
        self.nside = nside
        self.im = CMB_Bharat()

    
    def noiseTQU(self,idx=None):
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
    
