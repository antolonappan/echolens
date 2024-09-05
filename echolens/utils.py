"""
This module contains utility functions used in the echolens package.
"""
import numpy as np
import healpy as hp

def cli(cl : np.ndarray) -> np.ndarray:
    """
    The function cli is used to compute the inverse of the input power spectrum.

    :param cl: The input power spectrum.
    :type cl: np.ndarray
    :return: The inverse of the input power spectrum.
    :rtype: np.ndarray
    """
    ret = np.zeros_like(cl)
    ret[np.where(cl > 0)] = 1. / cl[np.where(cl > 0)]
    return ret

def synalm_c1(cl_x : np.ndarray, alm_x : np.ndarray, cl_y : np.ndarray, cl_xy : np.ndarray) -> np.ndarray:
    """
    The function synalm_c1 is used to generate a realization of a field given the power spectrum of the field and the cross-power spectrum with another field.

    :param cl_x: The power spectrum of the field.
    :type cl_x: np.ndarray
    :param alm_x: The alm of the field.
    :type alm_x: np.ndarray
    :param cl_y: The power spectrum of the other field.
    :type cl_y: np.ndarray
    :param cl_xy: The cross-power spectrum between the two fields.
    :type cl_xy: np.ndarray
    :return: The realization of the field.
    :rtype: np.ndarray
    """

    lmax = hp.Alm.getlmax(len(alm_x))
    assert len(cl_x) >= lmax + 1, 'cl_x is less than the lmax of alm_x'
    assert len(cl_y) >= lmax + 1, 'cl_y is less than the lmax of alm_x'
    assert len(cl_xy) >= lmax + 1, 'cl_xy is less than the lmax of alm_x'
    corr = cl_xy * cli(cl_x)
    elm = hp.synalm(cl_y - corr*cl_xy, lmax=lmax)
    return elm + hp.almxfl(alm_x ,corr)

def synalm_c2(cl_x : np.ndarray, alm_x : np.ndarray, cl_y : np.ndarray, alm_y : np.ndarray, cl_xy : np.ndarray) -> np.ndarray:
    """
    The function synalm_c2 is used to generate a realization of a field given the power spectrum of the field and the cross-power spectrum with another field.

    :param cl_x: The power spectrum of the field.
    :type cl_x: np.ndarray
    :param alm_x: The alm of the field.
    :type alm_x: np.ndarray
    :param cl_y: The power spectrum of the other field.
    :type cl_y: np.ndarray
    :param alm_y: The alm of the other field.
    :type alm_y: np.ndarray
    :param cl_xy: The cross-power spectrum between the two fields.
    :type cl_xy: np.ndarray
    :return: The realization of the field.
    :rtype: np.ndarray
    """

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


def slice_alms(alms : np.ndarray, lmax_new : int) -> np.ndarray:
    """
    The function slice_alms is used to slice the input alms to a new lmax.

    :param alms: The input alms.
    :type alms: np.ndarray
    :param lmax_new: The new lmax.
    :type lmax_new: int
    :return: The sliced alms.
    :rtype: np.ndarray
    """
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
    
def cut_alms(alms : list, lmax_new : int) -> list:
    """
    The function cut_alms is used to slice the input alms to a new lmax.

    :param alms: The input alms.
    :type alms: list
    :param lmax_new: The new lmax.
    :type lmax_new: int
    :return: The sliced alms.
    :rtype: list
    """
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


def arc2cl(arc : np.ndarray) -> np.ndarray:
    """
    The function arc2cl is used to convert arcmin to Cl.

    :param arc: The input arcmin.
    :type arc: np.ndarray
    :return: The output Cl.
    :rtype: np.ndarray
    """
    return np.radians(arc/60)**2


def cl2arc(cl : np.ndarray) -> np.ndarray:
    """
    The function cl2arc is used to convert Cl to arcmin.

    :param cl: The input Cl.
    :type cl: np.ndarray
    :return: The output arcmin.
    :rtype: np.ndarray
    """

    return np.rad2deg(np.sqrt(cl))*60


def ilcnoise(arr : np.ndarray) -> np.ndarray:
    """
    The function ilcnoise is used to compute the noise of an ILC map.

    :param arr: The input noise.
    :type arr: np.ndarray
    :return: The output noise.
    :rtype: np.ndarray
    """
    
    return cl2arc(1/sum(1/arc2cl(arr)))
