import numpy as np
try:
    import cupy as xp
    xp.array(0)
except:
    import numpy as xp
from numpy.linalg import inv
from .. import units
from .. import kit
from ..athena_data import asnumpy


def add_data(ad,add_bcc=True):
    for var in ['bcc1','bcc2','bcc3']:
        if ((var not in ad.data_raw.keys()) and add_bcc):
                ad.add_data_func(var, lambda sf : sf.data('zeros'))
    ad.add_data_func('velR', lambda sf : (sf.data('velx')*sf.data('x')+sf.data('vely')*sf.data('y'))/sf.data('R'))
    ad.add_data_func('velphi', lambda sf : (-sf.data('velx')*sf.data('y')+sf.data('vely')*sf.data('x'))/sf.data('R'))
    ad.add_data_func('bccR', lambda sf : (sf.data('bccx')*sf.data('x')+sf.data('bccy')*sf.data('y'))/sf.data('R'))
    ad.add_data_func('bccphi', lambda sf : (-sf.data('bccx')*sf.data('y')+sf.data('bccy')*sf.data('x'))/sf.data('R'))
    ad.add_data_func('Omega', lambda sf : xp.sqrt(1/sf.data('R')**3))
    ad.add_data_func('dens*velR', lambda sf : sf.data('dens')*sf.data('velR'))
    ad.add_data_func('dens*velphi', lambda sf : sf.data('dens')*sf.data('velphi'))
    ad.add_data_func('dens*velphi^2', lambda sf : sf.data('dens')*sf.data('velphi')**2)
    ad.add_data_func('dens*velz', lambda sf : sf.data('dens')*sf.data('velz'))
    ad.add_data_func('T_Rphi_h', lambda sf : sf.data('dens')*sf.data('velR')*sf.data('velphi'))
    ad.add_data_func('T_zphi_h', lambda sf : sf.data('dens')*sf.data('velz')*sf.data('velphi'))
    ad.add_data_func('T_Rphi_m', lambda sf : sf.data('bccR')*sf.data('bccphi'))
    ad.add_data_func('T_zphi_m', lambda sf : sf.data('bccz')*sf.data('bccphi'))
    ad.add_data_func('T_Rphi', lambda sf : sf.data('T_Rphi_h')+sf.data('T_Rphi_m'))
    ad.add_data_func('T_zphi', lambda sf : sf.data('T_zphi_h')+sf.data('T_zphi_m'))
    for var in ['bccr','bccR','bccphi','bccz']:
        ad.add_data_func(f'|{var}|', lambda sf,var=var : xp.abs(sf.data(var)))
