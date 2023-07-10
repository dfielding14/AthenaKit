try:
    import cupy as xp
except ImportError:
    import numpy as xp
from numpy.linalg import inv
from ... import athenakit as ak

def add_tools(ad):
    mu = ad.header('units','mu',float)
    bhmass_msun = ad.header('units','bhmass_msun',float)
    bhmass_cgs = bhmass_msun * ak.msun_cgs
    length_cgs_ = ak.grav_constant_cgs*bhmass_cgs/(ak.speed_of_light_cgs)**2
    time_cgs_ = length_cgs_/ak.speed_of_light_cgs
    density_scale = mu*ak.atomic_mass_unit_cgs
    mass_cgs_ = density_scale*(length_cgs_**3)
    ad.unit=ak.Units(lunit=length_cgs_,munit=mass_cgs_,tunit=time_cgs_,mu=mu)

    ad.accel = lambda x: -ak.Acceleration(x,m=ad.header('problem','m_bh',float),
                            mc=ad.header('problem','m_star',float),
                            rc=ad.header('problem','r_star',float),
                            ms=ad.header('problem','m_dm',float),
                            rs=ad.header('problem','r_dm',float),
                            g=ad.unit.grav_constant)
    return

def add_tran(ad):
    jx=ad.average('jx',where=ad.data('temp')<ad.header('problem','t_cold',float),weights='mass')
    jy=ad.average('jy',where=ad.data('temp')<ad.header('problem','t_cold',float),weights='mass')
    jz=ad.average('jz',where=ad.data('temp')<ad.header('problem','t_cold',float),weights='mass')
    def normal(vec):
        return vec/xp.sqrt(xp.sum(vec**2))
    disk_z = normal(xp.array([jx,jy,jz]))
    y_12=xp.asarray([0,1])
    disk_y=normal(xp.array([(-y_12[0]*disk_z[1]-y_12[1]*disk_z[2])/disk_z[0],y_12[0],y_12[1]]))
    disk_x=normal(xp.cross(disk_y,disk_z))
    ad.tran=inv(xp.stack((disk_x,disk_y,disk_z)).T)

def add_data(ad,add_bcc=True):
    if ('bcc1' not in ad.data_raw.keys()) and add_bcc:
        ad.add_data_func('bcc1', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc2', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc3', lambda sf : sf.data('zeros'))
    
    ad.add_data_func('tran_x', lambda sf : sf.tran[0,0]*sf.data('x')+sf.tran[0,1]*sf.data('y')+sf.tran[0,2]*sf.data('z'))
    ad.add_data_func('tran_y', lambda sf : sf.tran[1,0]*sf.data('x')+sf.tran[1,1]*sf.data('y')+sf.tran[1,2]*sf.data('z'))
    ad.add_data_func('tran_z', lambda sf : sf.tran[2,0]*sf.data('x')+sf.tran[2,1]*sf.data('y')+sf.tran[2,2]*sf.data('z'))
    
    ad.add_data_func('tran_velx', lambda sf : sf.tran[0,0]*sf.data('velx')+sf.tran[0,1]*sf.data('vely')+sf.tran[0,2]*sf.data('velz'))
    ad.add_data_func('tran_vely', lambda sf : sf.tran[1,0]*sf.data('velx')+sf.tran[1,1]*sf.data('vely')+sf.tran[1,2]*sf.data('velz'))
    ad.add_data_func('tran_velz', lambda sf : sf.tran[2,0]*sf.data('velx')+sf.tran[2,1]*sf.data('vely')+sf.tran[2,2]*sf.data('velz'))

    ad.add_data_func('tran_bccx', lambda sf : sf.tran[0,0]*sf.data('bccx')+sf.tran[0,1]*sf.data('bccy')+sf.tran[0,2]*sf.data('bccz'))
    ad.add_data_func('tran_bccy', lambda sf : sf.tran[1,0]*sf.data('bccx')+sf.tran[1,1]*sf.data('bccy')+sf.tran[1,2]*sf.data('bccz'))
    ad.add_data_func('tran_bccz', lambda sf : sf.tran[2,0]*sf.data('bccx')+sf.tran[2,1]*sf.data('bccy')+sf.tran[2,2]*sf.data('bccz'))

    ad.add_data_func('tran_R', lambda sf : xp.sqrt(sf.data('tran_x')**2+sf.data('tran_y')**2))
    ad.add_data_func('tran_velR', lambda sf : (sf.data('tran_x')*sf.data('tran_velx')+sf.data('tran_y')*sf.data('tran_vely'))/sf.data('tran_R'))
    ad.add_data_func('tran_velphi', lambda sf : (sf.data('tran_x')*sf.data('tran_vely')-sf.data('tran_y')*sf.data('tran_velx'))/sf.data('tran_R'))
    ad.add_data_func('tran_bccR', lambda sf : (sf.data('tran_x')*sf.data('tran_bccx')+sf.data('tran_y')*sf.data('tran_bccy'))/sf.data('tran_R'))
    ad.add_data_func('tran_bccphi', lambda sf : (sf.data('tran_x')*sf.data('tran_bccy')-sf.data('tran_y')*sf.data('tran_bccx'))/sf.data('tran_R'))

    ad.add_data_func('tran_stress_zphi_reynolds', lambda sf : sf.data('dens')*sf.data('tran_velz')*sf.data('tran_velphi'))
    ad.add_data_func('tran_stress_zphi_maxwell', lambda sf : -sf.data('tran_bccz')*sf.data('tran_bccphi'))
    ad.add_data_func('tran_stress_zphi', lambda sf : sf.data('tran_stress_zphi_reynolds')+sf.data('tran_stress_zphi_maxwell'))

    ad.add_data_func('tran_stress_Rphi_reynolds', lambda sf : sf.data('dens')*sf.data('tran_velR')*sf.data('tran_velphi'))
    ad.add_data_func('tran_stress_Rphi_maxwell', lambda sf : -sf.data('tran_bccR')*sf.data('tran_bccphi'))
    ad.add_data_func('tran_stress_Rphi', lambda sf : sf.data('tran_stress_Rphi_reynolds')+sf.data('tran_stress_Rphi_maxwell'))

    ad.add_data_func('tran_Omega', lambda sf : xp.sqrt(sf.accel(sf.data('tran_R'))/sf.data('tran_R')))
    ad.add_data_func('tran_radial_flow', lambda sf : 0.5*sf.data('dens')*sf.data('tran_velR')*sf.data('tran_Omega'))
    ad.add_data_func('tran_z/R', lambda sf : sf.data('tran_z')/sf.data('tran_R'))

    ad.add_data_func('tran_stress_zphi_reynolds/R', lambda sf : sf.data('tran_stress_zphi_reynolds')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_zphi_maxwell/R', lambda sf : sf.data('tran_stress_zphi_maxwell')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_zphi/R', lambda sf : sf.data('tran_stress_zphi')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi_reynolds/R', lambda sf : sf.data('tran_stress_Rphi_reynolds')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi_maxwell/R', lambda sf : sf.data('tran_stress_Rphi_maxwell')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi/R', lambda sf : sf.data('tran_stress_Rphi')/sf.data('tran_R'))

