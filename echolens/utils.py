import numpy as np
import healpy as hp

def cli(cl):
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def synalm_c1(cl_x, alm_x, cl_y, cl_xy):
    lmax = hp.Alm.getlmax(len(alm_x))
    assert len(cl_x) >= lmax + 1, 'cl_x is less than the lmax of alm_x'
    assert len(cl_y) >= lmax + 1, 'cl_y is less than the lmax of alm_x'
    assert len(cl_xy) >= lmax + 1, 'cl_xy is less than the lmax of alm_x'
    corr = cl_xy * cli(cl_x)
    elm = hp.synalm(cl_y - corr*cl_xy, lmax=lmax)
    return elm + hp.almxfl(alm_x ,corr)

def synalm_c2(cl_x, alm_x, cl_y, alm_y, cl_z, cl_xz, cl_yz):
    lmax = hp.Alm.getlmax(len(alm_x))
    assert len(cl_x) >= lmax + 1, 'cl_x is less than the lmax of alm_x'
    assert len(cl_y) >= lmax + 1, 'cl_y is less than the lmax of alm_x'
    assert len(cl_z) >= lmax + 1, 'cl_z is less than the lmax of alm_x'
    assert len(cl_xz) >= lmax + 1, 'cl_xz is less than the lmax of alm_x'
    assert len(cl_yz) >= lmax + 1, 'cl_yz is less than the lmax of alm_x'
    corr_xz = cl_xz * cli(cl_x)
    corr_yz = cl_yz * cli(cl_y)
    elm = hp.synalm(cl_z - corr_xz*cl_xz - corr_yz*cl_yz, lmax=lmax)
    return elm + hp.almxfl(alm_x ,corr_xz) + hp.almxfl(alm_y ,corr_yz)


def slice_alms(alms, lmax_new):
    if len(alms) > 3:
        lalm = 1
        lmax = hp.Alm.getlmax(len(alms))
    elif len(alms) <= 3:
        lalm = len(alms)
        lmax = hp.Alm.getlmax(len(alms[0]))
    else:
        raise ValueError('something is wrong with the input alms')
    
    if lmax_new > lmax:
        raise ValueError('lmax_new must be smaller or equal to lmax')
    elif lmax_new == lmax:
        return alms
    else:
        if lalm == 1:
            alms_new = np.zeros(hp.Alm.getsize(lmax_new), dtype=alms.dtype)
        else:
            alms_new = np.zeros((lalm, hp.Alm.getsize(lmax_new)), dtype=alms.dtype)
        indices_full = hp.Alm.getidx(lmax,*hp.Alm.getlm(lmax_new))
        indices_new = hp.Alm.getidx(lmax_new,*hp.Alm.getlm(lmax_new))
        if lalm == 1:
            alms_new[indices_new] = alms[indices_full]
        else:
            alms_new[:,indices_new] = alms[:,indices_full]
        return alms_new
    
def cut_alms(alms,lmax_new):
    assert type(alms) == list, 'alms must be a list of alms'
    lmax_all = []
    for i in range(len(alms)):
        lmax_all.append(hp.Alm.getlmax(len(alms[i])))
    lmax = np.min(lmax_all)
    assert lmax >= lmax_new, 'lmax_new must be smaller or equal to the minimum lmax of the input alms'
    alms_new = []
    for i in range(len(alms)):
        alms_new.append(slice_alms(alms[i],lmax_new))  
    return alms_new
