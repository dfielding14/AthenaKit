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
from .. import macros, mpi

def add_tools(ad):
    ad.rmin = asnumpy(np.min(ad.data('r').min()))
    if (macros.mpi_enabled): ad.rmin = mpi.min(ad.rmin)
    ad.rmin = float(ad.rmin)
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))  

    return

def add_data(ad,add_bcc=True):
    for var in ['bcc1','bcc2','bcc3']:
        if ((var not in ad.data_raw.keys()) and add_bcc):
                ad.add_data_func(var, lambda data : data('zeros'))
    ad.add_data_func('z/R', lambda data : data('z')/data('R'))
    ad.add_data_func('zhat', lambda data : data('z')/xp.abs(data('z')))
    ad.add_data_func('velR', lambda data : (data('velx')*data('x')+data('vely')*data('y'))/data('R'))
    ad.add_data_func('velphi', lambda data : (-data('velx')*data('y')+data('vely')*data('x'))/data('R'))
    ad.add_data_func('bccR', lambda data : (data('bccx')*data('x')+data('bccy')*data('y'))/data('R'))
    ad.add_data_func('bccphi', lambda data : (-data('bccx')*data('y')+data('bccy')*data('x'))/data('R'))
    ad.add_data_func('Omega', lambda data : xp.sqrt(1/data('R')**3))
    ad.add_data_func('dens*velz*zhat', lambda data : data('dens')*data('velz')*data('zhat'))
    ad.add_data_func('pmag*velz*zhat', lambda data : data('pmag')*data('velz')*data('zhat'))
    ad.add_data_func('btot^2*velz*zhat', lambda data : data('btot^2')*data('velz')*data('zhat'))
    ad.add_data_func('velphi*bccR', lambda data : data('velphi')*data('bccR'))
    ad.add_data_func('velR*bccphi', lambda data : data('velR')*data('bccphi'))
    ad.add_data_func('velz*bccR', lambda data : data('velz')*data('bccR'))
    ad.add_data_func('velR*bccz', lambda data : data('velR')*data('bccz'))
    ad.add_data_func('velphi*bccz', lambda data : data('velphi')*data('bccz'))
    ad.add_data_func('velz*bccphi', lambda data : data('velz')*data('bccphi'))
    ad.add_data_func('T_Rphi_h', lambda data : data('dens')*data('velR')*data('velphi'))
    ad.add_data_func('T_zphi_h', lambda data : data('dens')*data('velz')*data('velphi'))
    ad.add_data_func('T_Rphi_m', lambda data : data('bccR')*data('bccphi'))
    ad.add_data_func('T_zphi_m', lambda data : data('bccz')*data('bccphi'))
    ad.add_data_func('T_Rphi', lambda data : data('T_Rphi_h')+data('T_Rphi_m'))
    ad.add_data_func('T_zphi', lambda data : data('T_zphi_h')+data('T_zphi_m'))
    for var in ['bccr','bccR','bccphi','bccz']:
        ad.add_data_func(f'|{var}|', lambda data,var=var : xp.abs(data(var)))
    ad.add_data_func('dvRdx', lambda data : data.ad.gradient(data('velR'), 'x', **data.kwargs))
    ad.add_data_func('dvRdy', lambda data : data.ad.gradient(data('velR'), 'y', **data.kwargs))
    ad.add_data_func('dvRdz', lambda data : data.ad.gradient(data('velR'), 'z', **data.kwargs))
    ad.add_data_func('B dot nabla vR', lambda data : data('bccx')*data('dvRdx')+
                     data('bccy')*data('dvRdy')+data('bccz')*data('dvRdz'))
    for key,inte in zip(['mdot','momdot','eidot','ekdot','edot'],
                        ['dens','momr',  'eint', 'ekin', 'etot']):
        for direc in ['','in','out']:
            var = key+direc
            ad.add_data_func(var, lambda data, inte=inte, direc=direc : 4.0*xp.pi*data('r')**2*data(inte)*data('velr'+direc))

    return

