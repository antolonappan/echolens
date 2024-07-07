import numpy as np
import healpy as hp

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def constrained_synalm(cl_x, alm_x, cl_y, cl_xy):
    lmax = hp.Alm.getlmax(len(alm_x))
    assert len(cl_x) >= lmax + 1, 'cl_x is less than the lmax of alm_x'
    assert len(cl_y) >= lmax + 1, 'cl_y is less than the lmax of alm_x'
    assert len(cl_xy) >= lmax + 1, 'cl_xy is less than the lmax of alm_x'
    corr = cl_xy * cli(cl_x)
    elm = hp.synalm(cl_y - corr*cl_xy, lmax=lmax)
    return elm + hp.almxfl(alm_x ,corr)