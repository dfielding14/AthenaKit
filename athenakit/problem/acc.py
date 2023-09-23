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

bhmass_msun = 6.5e9
mu = 0.618
bhmass_cgs = bhmass_msun * units.msun_cgs
length_cgs_ = units.grav_constant_cgs*bhmass_cgs/(units.speed_of_light_cgs)**2
time_cgs_ = length_cgs_/units.speed_of_light_cgs
density_scale = mu*units.atomic_mass_unit_cgs
mass_cgs_ = density_scale*(length_cgs_**3)
grunit=units.Units(lunit=length_cgs_,munit=mass_cgs_,tunit=time_cgs_,mu=mu)

##############################################################
# Solar metallicity
# H, He, C, N, O, Ne, Na, Mg, Al, Si, S, Ar, Ca, Fe, Ni
atom_number=np.array([1,2,6,7,8,10,11,12,13,14,16,18,20,26,28]) # atomic number
atom_mass=np.array([1.00784,4.0026,12,14,16,20,23,24,27,28,32,40,40,56,58]) # Atomic mass
# From Schure et al. 2009 A&A 508, 751
logni=np.array([0,-1.01,-3.44,-3.95,-3.07,-3.91,-5.67,-4.42,-5.53,-4.45,-4.79,-5.44,-5.64,-4.33,-5.75])
n_i=10**logni # ion number density
n_e=n_i*atom_number # electron number density
n_t=n_i+n_e  # total number density
m_t=np.sum(n_i*atom_mass) # total mass density
n_t_h_ratio=n_t.sum()
n_e_h_ratio=n_e.sum()
n_i_h_ratio=n_i.sum()
n_t_e_ratio=n_t_h_ratio/n_e_h_ratio
n_t_i_ratio=n_t_h_ratio/n_i_h_ratio
##############################################################

def add_tools(ad):
    ad.rin = ad.header('problem','r_in',float)
    ad.rmin = float(asnumpy(np.min(ad.data('r').min())))
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))
            
    mu = ad.header('units','mu',float)
    bhmass_msun = ad.header('units','bhmass_msun',float)
    bhmass_cgs = bhmass_msun * units.msun_cgs
    length_cgs_ = units.grav_constant_cgs*bhmass_cgs/(units.speed_of_light_cgs)**2
    time_cgs_ = length_cgs_/units.speed_of_light_cgs
    density_scale = mu*units.atomic_mass_unit_cgs
    mass_cgs_ = density_scale*(length_cgs_**3)
    ad.unit=units.Units(lunit=length_cgs_,munit=mass_cgs_,tunit=time_cgs_,mu=mu)
    ad.unit.mdot_msun_yr = ad.unit.mass_cgs/ad.unit.time_cgs/units.msun_cgs*units.yr_cgs

    m_bh=ad.header('problem','m_bh',float)
    m_star=ad.header('problem','m_star',float)
    r_star=ad.header('problem','r_star',float)
    m_dm=ad.header('problem','m_dm',float)
    r_dm=ad.header('problem','r_dm',float)
    rad_entry=ad.header('problem','rad_entry',float)
    dens_entry=ad.header('problem','dens_entry',float)
    gamma=5.0/3.0
    k0_entry=ad.header('problem','k0_entry',float)
    xi_entry=ad.header('problem','xi_entry',float)
    
    def accel(r,m=m_bh,mc=m_star,rc=r_star,ms=m_dm,rs=r_dm,g=ad.unit.grav_constant):
        return kit.Acceleration(r,m,mc,rc,ms,rs,g)

    ad.accel = accel
    
    # profile solver
    def NFWMass(r,ms,rs):
        return ms*(np.log(1+r/rs)-r/(rs+r))
    def TotMass(r,m=m_bh,mc=m_star,rc=r_star,ms=m_dm,rs=r_dm):
        return m+NFWMass(r,mc,rc)+NFWMass(r,ms,rs)
    def Acceleration(r,m,mc,rc,ms,rs,g):
        return -g*(TotMass(r,m,mc,rc,ms,rs))/r**2
    def DrhoDr(x,rho):
        r = x*rad_entry
        accel = rad_entry*Acceleration(r,m_bh,m_star,r_star,m_dm,r_dm,ad.unit.grav_constant)
        #print(accel)
        #print(rho,gamma,xi)
        grad = (2*rho**(2-gamma)*accel/k0_entry-rho*xi_entry*x**(xi_entry-1))/((1+x**xi_entry)*gamma)
        return grad
    def RK4(func,x,y,h):
        k1=func(x,y)
        x+=0.5*h
        k2=func(x,y+0.5*k1*h)
        k3=func(x,y+0.5*k2*h)
        x+=0.5*h
        k4=func(x,y+k3*h)
        y+=1/6*(k1+2*k2+2*k3+k4)*h
        return y
    def SolveDens(N1=1024,N2=1024,logh=0.002):
        N=N1+N2
        dens_arr = np.zeros(N)
        dens = dens_entry
        dens_arr[N1]=dens
        for i in range(N1):
            x = 10**(-i*logh)
            h = 10**(-(i+1)*logh)-x
            dens = RK4(DrhoDr,x,dens,h)
            dens_arr[N1-i-1]=dens
        dens = dens_entry
        for i in range(N2-1):
            x = 10**(i*logh)
            h = 10**((i+1)*logh)-x
            dens = RK4(DrhoDr,x,dens,h)
            dens_arr[N1+i+1]=dens
        xs=np.logspace(-logh*N1,logh*(N2-1),N,endpoint=True)
        pres_arr = 0.5*k0_entry*(1.0+pow(xs,xi_entry))*pow(dens_arr,gamma)
        rss=dict()
        rss['r']=xs*rad_entry
        rss['dens']=dens_arr
        rss['pres']=pres_arr
        rss['temp']=rss['pres']/rss['dens']
        rss['entropy']=rss['pres']/rss['dens']**(gamma)
        return rss

    # solve
    ran=SolveDens(N1=2000,N2=1000,logh=0.004)
    ran['mass']=TotMass(ran['r'],m_bh,m_star,r_star,m_dm,r_dm)
    ran['g']=Acceleration(ran['r'],m_bh,m_star,r_star,m_dm,r_dm,ad.unit.grav_constant)
    ran['t_ff']=np.pi/4.0*np.sqrt(2.*ran['r']/-ran['g'])
    ran['v_ff']=np.sqrt(2.*ran['r']*-ran['g'])
    ran['v_kep']=np.sqrt(ran['r']*-ran['g'])
    ran['Omega']=ran['v_kep']/ran['r']
    ran['am_kep']=np.sqrt(ran['r']**3*-ran['g'])
    ran['potential']=ran['r']*-ran['g']
    ran['r_B']=ad.unit.grav_constant*m_bh/(gamma*ran['temp'])
    ran['Mdot_B']=np.pi*(ad.unit.grav_constant*m_bh)**2*ran['dens']/(gamma*ran['temp'])**1.5
    ad.rad_initial=ran
    return

def add_tran(ad):
    where=ad.data('temp')<ad.header('problem','t_cold',float)
    if (where.any()):
        amx=ad.average('amx',where=where,weights='mass')
        amy=ad.average('amy',where=where,weights='mass')
        amz=ad.average('amz',where=where,weights='mass')
    else:
        amx, amy, amz = 0.0, 0.0, 1.0
    def normal(vec):
        return vec/xp.sqrt(xp.sum(vec**2))
    disk_z = normal(xp.array([amx,amy,amz]))
    y_01=xp.asarray([0,1])
    disk_y=normal(xp.array([y_01[0],y_01[1],(-y_01[0]*disk_z[0]-y_01[1]*disk_z[1])/disk_z[2]]))
    disk_x=normal(xp.cross(disk_y,disk_z))
    ad.tran=inv(xp.stack((disk_x,disk_y,disk_z)).T)
    return

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

    ad.add_data_func('tran_stress_zphi_hydro', lambda sf : sf.data('dens')*sf.data('tran_velz')*sf.data('tran_velphi'))
    ad.add_data_func('tran_stress_zphi_maxwell', lambda sf : -sf.data('tran_bccz')*sf.data('tran_bccphi'))
    ad.add_data_func('tran_stress_zphi', lambda sf : sf.data('tran_stress_zphi_hydro')+sf.data('tran_stress_zphi_maxwell'))

    ad.add_data_func('tran_stress_Rphi_hydro', lambda sf : sf.data('dens')*sf.data('tran_velR')*sf.data('tran_velphi'))
    ad.add_data_func('tran_stress_Rphi_maxwell', lambda sf : -sf.data('tran_bccR')*sf.data('tran_bccphi'))
    ad.add_data_func('tran_stress_Rphi', lambda sf : sf.data('tran_stress_Rphi_hydro')+sf.data('tran_stress_Rphi_maxwell'))

    ad.add_data_func('tran_Omega', lambda sf : xp.sqrt(sf.accel(sf.data('tran_R'))/sf.data('tran_R')))
    ad.add_data_func('tran_dens_velR', lambda sf : sf.data('dens')*sf.data('tran_velR'))
    ad.add_data_func('tran_dens_velz', lambda sf : sf.data('dens')*sf.data('tran_velz'))
    ad.add_data_func('tran_radial_flow', lambda sf : 0.5*sf.data('dens')*sf.data('tran_velR')*sf.data('tran_Omega'))
    ad.add_data_func('tran_z/R', lambda sf : sf.data('tran_z')/sf.data('tran_R'))

    ad.add_data_func('tran_stress_zphi_hydro/R', lambda sf : sf.data('tran_stress_zphi_hydro')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_zphi_maxwell/R', lambda sf : sf.data('tran_stress_zphi_maxwell')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_zphi/R', lambda sf : sf.data('tran_stress_zphi')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi_hydro/R', lambda sf : sf.data('tran_stress_Rphi_hydro')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi_maxwell/R', lambda sf : sf.data('tran_stress_Rphi_maxwell')/sf.data('tran_R'))
    ad.add_data_func('tran_stress_Rphi/R', lambda sf : sf.data('tran_stress_Rphi')/sf.data('tran_R'))

    ad.add_data_func('t_initial', lambda sf : xp.interp(sf.data('r'),xp.asarray(sf.rad_initial['r']),xp.asarray(sf.rad_initial['temp'])))
    ad.add_data_func('t_hot', lambda sf : sf.header('problem','tf_hot',float)*sf.data('t_initial'))

    for var in ['mdot','mdotin','mdotout','momdot','momdotin','momdotout','ekdot','ekdotin','ekdotout']:
        ad.add_data_func(var, lambda sf, var=var : 4.0*xp.pi*sf.data('r')**2*sf.data(var.replace('dot','flxr')))
        ad.add_data_func(var+'_cold', lambda sf, var=var : sf.data(var)*(sf.data('temp')<sf.header('problem','t_cold',float)))
        ad.add_data_func(var+'_warm', lambda sf, var=var : sf.data(var)*(sf.data('temp')>=sf.header('problem','t_cold',float))*(sf.data('temp')<sf.data('t_hot')))
        ad.add_data_func(var+'_hot', lambda sf, var=var : sf.data(var)*(sf.data('temp')>=sf.data('t_hot')))

    return
