"""
import numpy as np
import Corrfunc.theory.DD

# pos has shape of (N_gal, 3) # Mpc/h
N_gal = 10000
box = 200 # Mpc/h
pos = np.vstack((np.random.rand(N_gal)*box, np.random.rand(N_gal)*box, np.random.rand(N_gal)*box)).T
vel = np.vstack((np.random.rand(N_gal)*box, np.random.rand(N_gal)*box, np.random.rand(N_gal)*box))#.T
w = np.random.rand(N_gal)
edges = np.linspace(0, 30, 21)
auto = 1 # this is an auto-correlation
nthread = 8

#res = Corrfunc.theory.DD(auto, nthread, edges, *pos.T, weights1=w, boxsize=box, periodic=True, weight_type='pair_product')
#res = Corrfunc.theory.DD(auto, nthread, edges, *pos.T, weights1=vel, boxsize=box, periodic=True, weight_type='pairwise_vel2')
res = Corrfunc.theory.DD(auto, nthread, edges, *pos.T, weights1=vel.T, boxsize=box, periodic=True, weight_type='pairwise_vel2', isa='fallback')
counts = res['npairs']
counts[0] -= len(pos) # subtract self-count from first bin
"""

import time

import Corrfunc
import numpy as np
import matplotlib.pyplot as plt
from numba_2pcf.cf import numba_pairwise_vel#, numba_2pcf

np.random.seed(3000)

bins = np.linspace(0, 30, 21)
N_gals = [1000, 10000, 100000, 1000000]
box = 200 # Mpc/h
periodic = False
verbose = False
autocorr = 1
nthread = 4

def pairwise_vel_asymm(pos, deltaT, bins, nthread, periodic=False, box=None, tau=None, pos2=None, v1d2=None, isa='avx512f'):#'fallback'):#'avx'):
    """
    If isa="avx" or "avx512f" (fastest) is not implemented, would "fallback" to the DOUBLE implementation
    """
    
    # determine auto or cross correlation
    if pos2 is not None:
        autocorr = 0
    else:
        autocorr = 1
    if tau is None and autocorr == 0:
        tau = np.ones(pos.shape[0]) 
    if v1d2 is None and autocorr == 0:
        v1d2 = np.ones(pos2.shape[0])
        
    # combine position and velocity into weights
    rv_num1 = np.vstack((pos.T, deltaT*tau)).T
    #rv_den1 = np.vstack((pos.T, tau)).T # not used
    if autocorr:
        rv_num2 = rv_num1; pos2 = pos; #rv_den2 = rv_den1
    else:
        rv_num2 = np.vstack((pos2.T, v1d2)).T
        #rv_den2 = np.vstack((pos2.T, v1d2)).T # not used
    pos2= pos2.astype(np.float32)
    pos = pos.astype(np.float32)
    rv_num1= rv_num1.astype(np.float32)
    rv_num2= rv_num2.astype(np.float32)
        
    # pairwise_vel_los_asymm requires 5 weights (3 positions, fourth is deltaT, fifth is tau proxy, whereas pairwise_vel_los_asymm_norm requires only 4 weights, but we still pass 5
    
    # compute numerator and denominator
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv_num1.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv_num2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_asymm')
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=pos.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=pos2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_norm')
    #res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv_den1.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv_den2.T, verbose=False, isa=isa, weight_type='pairwise_vel_los_asymm_norm')
    pairwise = -res['weightavg']/(res_norm['weightavg'])
    pairwise[res_norm['weightavg'] == 0.] = 0.
    
    # return pairwise estimator
    return pairwise


def pairwise_vel_symm(pos, v1d, bins, nthread, periodic=False, box=None, pos2=None, v1d2=None, isa='avx512f'):
    # determine auto or cross correlation
    if v1d2 is not None:
        autocorr = 0
    else:
        autocorr = 1

    # combine position and velocity into weights
    #v1d *= 0.
    rv = np.vstack((pos.T, v1d)).T
    #rv = np.vstack((v1d, pos.T)).T
    #print(rv[:10])
    #quit()
    print(rv.shape)
    rv_norm = rv[:, :3]
    if autocorr:
        rv2 = rv; pos2 = pos
    else:
        rv2 = np.vstack((pos2.T, v1d2)).T

    """
    # ensure that particles are within [0, box)
    if autocorr:
        min_pos = np.min(pos, axis=0)
        pos -= min_pos
        box = np.max(pos)
    else:
        N = pos.shape[0]
        N2 = pos2.shape[0]
        pos_both = np.vstack((pos, pos2))
        min_pos = np.min(pos_both, axis=0)
        pos_both -= min_pos
        box = np.max(pos_both)
        pos = pos_both[:N]
        pos2 = pos_both[N:]
        assert pos.shape[0] == N
        assert pos2.shape[0] == N2
    """
    box = None
    #isa = 'fallback'
    #isa = 'avx'
    
    # compute numerator and denominator
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=rv2.T, verbose=False, weight_type='pairwise_vel_los', isa=isa)
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=pos.T, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=pos2.T, verbose=False, weight_type='pairwise_vel_los_norm', isa=isa)
    """
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=v1d, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=v1d, verbose=False, weight_type='pair_product', isa=isa)
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=v1d, periodic=periodic, boxsize=box, X2=pos2[:, 0], Y2=pos2[:, 1], Z2=pos2[:, 2], weights2=v1d, verbose=False, weight_type='pair_product', isa=isa)
    """
    
    pair = -res['weightavg']/(res_norm['weightavg'])
    pair[res_norm['weightavg'] == 0.] = 0.
    
    # return pairwise estimator
    return pair

        
for i in range(len(N_gals)):
    N_gal = N_gals[i]
    
    pos = np.vstack((np.random.rand(N_gal)*box, np.random.rand(N_gal)*box, np.random.rand(N_gal)*box)).T - box/2.
    v1d = np.random.rand(N_gal)*1000. - 500.

    # if not random
    want_true = 0 #0#1
    if want_true:
        data = np.load("test.npz")
        pos = data['P']
        v1d = data['V']
        bins = data['rbins']
    print(pos.shape[0])

    # have checked that the answers agree for feeding the same data
    if autocorr == 0:
        pairwise_vel = pairwise_vel_asymm
        pos2 = pos
    else:
        pairwise_vel = pairwise_vel_symm
        pos2 = None
    
    t = time.time()
    PV = pairwise_vel(pos, v1d, bins, nthread, periodic=False, box=None, pos2=None, v1d2=None, isa='avx512f')
    print("time avx512 = ", time.time()-t)
    
    
    t = time.time()
    PV_corr = pairwise_vel(pos, v1d, bins, nthread, periodic=False, box=None, pos2=pos2, v1d2=None, isa='avx')
    """
    rv = np.vstack((pos.T, v1d)).T
    print(rv.shape)
    res = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=rv.T, periodic=periodic, boxsize=box, X2=pos[:, 0], Y2=pos[:, 1], Z2=pos[:, 2], weights2=rv.T, verbose=verbose, isa='fallback', weight_type='pairwise_vel_los')
    #weights1=None, periodic=True, boxsize=None, X2=None, Y2=None, Z2=None, weights2=None
    res_norm = Corrfunc.theory.DD(autocorr, nthread, bins, *pos.T, weights1=pos.T, periodic=periodic, boxsize=box, X2=pos[:, 0], Y2=pos[:, 1], Z2=pos[:, 2], weights2=pos.T, verbose=verbose, isa='fallback', weight_type='pairwise_vel_los_norm')
    print("cij^2 = ", res_norm['weightavg'])
    PV_corr = -res['weightavg']/res_norm['weightavg']
    """
    print("time avx = ", time.time()-t)
    #print("pairwise = ", PV_corr)


    """
    t = time.time()
    PV = pairwise_vel(pos, v1d, bins, nthread, periodic=False, box=None, pos2=None, v1d2=None, isa='fallback')
    print("time fallback = ", time.time()-t)
    """

    """
    t = time.time()
    PV = numba_pairwise_vel(pos, v1d, box=box, Rmax=np.max(bins), nbin=len(bins)-1, corrfunc=False, nthread=nthread, periodic=periodic)['pairwise']
    print("time numba = ", time.time()-t)
    #print("numba pairwise = ", PV)
    print("NOTE THAT THIS ALTERS YOUR POS ARRAY")
    """
    
    frac = (PV_corr - PV)*100./PV
    frac[PV == 0.] = 0.
    print("frac [%] = ", frac)
    print("--------------------")

    want_plot = False
    if want_plot:
        binc = (bins[1:]+bins[:-1])*.5
        plt.plot(binc, PV)
        plt.plot(binc, PV_corr)
        plt.show()

    if want_true:
        break

# time the functions twice faster
# implement avx
# to do cross-correlation
# implement in multi pairwise auto
# implement the cross-correlation one

#res = Corrfunc.theory.DD(1, 4, bins, *pos.T, weights1=vel.T, periodic=True, boxsize=box, verbose=True, weight_type='pairwise_vel', isa='avx')#, isa='fallback')

# 4/3 cij^2
