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

class InitialCondition:
    def __init__(self,m_bh,m_star,r_star,m_dm,r_dm,r_entropy,k_entropy,xi_entropy,x_0,dens_0,
                 gamma=5.0/3.0,unit=grunit) -> None:
        self.m_bh=m_bh
        self.m_star=m_star
        self.r_star=r_star
        self.m_dm=m_dm
        self.r_dm=r_dm
        self.r_entropy=r_entropy
        self.k_entropy=k_entropy
        self.xi_entropy = xi_entropy
        self.x_0=x_0
        self.dens_0=dens_0
        self.gamma=gamma
        self.unit=unit
        self.rs=dict()
        pass
    # profile solver
    def NFWMass(self,r,ms,rs):
        return ms*(np.log(1+r/rs)-r/(rs+r))
    def NFWDens(self,r,ms,rs):
        return ms/(4*np.pi*rs**2*r*(1+r/rs)**2)
    def StellarMass(self,r):
        return self.NFWMass(r,self.m_star,self.r_star)
    def StellarDens(self,r):
        return self.NFWDens(r,self.m_star,self.r_star)
    def DMmass(self,r):
        return self.NFWMass(r,self.m_dm,self.r_dm)
    def DMdens(self,r):
        return self.NFWDens(r,self.m_dm,self.r_dm)
    def TotMass(self,r):
        return self.m_bh+self.StellarMass(r)+self.DMmass(r)
    def Acceleration(self,r):
        return -self.unit.grav_constant*(self.TotMass(r))/r**2
    def DrhoDr(self,x,rho):
        r = x*self.r_entropy
        accel = self.r_entropy*self.Acceleration(r)
        #print(accel)
        #print(rho,gamma,xi)
        gamma = self.gamma
        xi = self.xi_entropy
        k_0 = self.k_entropy
        grad = (2*rho**(2-gamma)*accel/k_0-rho*xi*x**(xi-1))/((1+x**xi)*gamma)
        return grad
    def RK4(self,func,x,y,h):
        k1=func(x,y)
        x+=0.5*h
        k2=func(x,y+0.5*k1*h)
        k3=func(x,y+0.5*k2*h)
        x+=0.5*h
        k4=func(x,y+k3*h)
        y+=1/6*(k1+2*k2+2*k3+k4)*h
        return y
    def solve(self,x0=None,dens0=None,N1=2048,N2=1024,logh=0.002):
        if x0 is None:
            x0 = self.x_0
        if dens0 is None:
            dens0 = self.dens_0
        gamma = self.gamma
        unit = self.unit
        Ntotal=N1+N2
        dens_arr = np.zeros(Ntotal)
        dens = dens0
        dens_arr[N1]=dens
        for i in range(N1):
            x = x0*10**(-i*logh)
            h = x0*10**(-(i+1)*logh)-x
            dens = self.RK4(self.DrhoDr,x,dens,h)
            dens_arr[N1-i-1]=dens
        dens = dens0
        for i in range(N2-1):
            x = x0*10**(i*logh)
            h = x0*10**((i+1)*logh)-x
            dens = self.RK4(self.DrhoDr,x,dens,h)
            dens_arr[N1+i+1]=dens
        xs=x0*np.logspace(-logh*N1,logh*(N2-1),Ntotal,endpoint=True)
        pres_arr = 0.5*self.k_entropy*(1.0+pow(xs,self.xi_entropy))*pow(dens_arr,self.gamma)
        ran=dict()
        ran['r']=xs*self.r_entropy
        ran['dens']=dens_arr
        ran['pres']=pres_arr
        ran['temp']=ran['pres']/ran['dens']
        ran['eint']=ran['pres']/(gamma-1)
        ran['entropy']=ran['pres']/ran['dens']**(5/3)
        ran['mass']=self.TotMass(ran['r'])
        ran['g']=self.Acceleration(ran['r'])
        ran['t_ff']=np.pi/4.0*np.sqrt(2.*ran['r']/-ran['g'])
        ran['v_ff']=np.sqrt(2.*ran['r']*-ran['g'])
        ran['v_kep']=np.sqrt(ran['r']*-ran['g'])
        ran['t_orbit']=2*np.pi*(np.sqrt(ran['r']/-ran['g']))
        ran['Omega']=ran['v_kep']/ran['r']
        ran['am_kep']=np.sqrt(ran['r']**3*-ran['g'])
        ran['potential']=ran['r']*-ran['g']
        ran['r_B']=self.unit.grav_constant*self.m_bh/(self.gamma*ran['temp'])
        ran['Mdot_B']=np.pi*(self.unit.grav_constant*self.m_bh)**2*ran['dens']/(self.gamma*ran['temp'])**1.5
        ran['tempK']=ran['temp']*unit.temperature_cgs
        #ran['tcool']=1/(gamma-1)*ran['pres']*unit.pressure_cgs/(ran['dens']**2*CoolFnShure(ran['tempK']))/unit.time_cgs
        ran['cooling_rate']=ran['dens']**2*kit.CoolFnShure(ran['tempK'])/unit.cooling_cgs/n_t_h_ratio**2
        ran['tcool']=ran['eint']/(ran['dens']**2*kit.CoolFnShure(ran['tempK'])/unit.cooling_cgs/n_t_h_ratio**2)
        ran['theat']=ran['pres']/(2e-4*ran['dens']*ran['r']**-1.5+0.95*0.0716455*ran['dens']*(ran['r']+2.0)**-1.5)
        #ran['tcool_c']=1/(gamma-1)*1e5/unit.temperature_cgs/(ran['dens']*CoolFnShure(1e5))/unit.time_cgs
        ran['tcool_c']=1/(gamma-1)*1e4/unit.temperature_cgs/(1e2*ran['dens']/ran['dens']*kit.CoolFnShure(1e4)/unit.cooling_cgs)
        ran['theat_c']=1/(gamma-1)*1e5/unit.temperature_cgs/(2e-4*ran['r']**-1.5+0.95*0.0716455*(ran['r']+2.0)**-1.5)
        Mdot_B=ran['Mdot_B'][0]
        ran['v_inflow']=Mdot_B/(4*np.pi*ran['dens']*ran['r']**2)
        ran['t_inflow']=ran['r']/ran['v_inflow']
        self.rs=ran
        return ran
    
    def __call__(self, key):
        if not self.rs:
            self.solve()
        return self.rs[key]

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

    ad.ic = InitialCondition(m_bh,m_star,r_star,m_dm,r_dm,rad_entry,k0_entry,xi_entry,1.0,dens_entry,gamma=gamma,unit=ad.unit)
    ad.ic.solve(N1=2000,N2=1000,logh=0.004)
    ad.rad_initial = ad.ic.rs
    ad.accel = ad.ic.Acceleration
    return

def add_tran(ad):
    where=ad.data('temp')<ad.header('problem','t_cold',float)
    amx, amy, amz = 0.0, 0.0, 1.0
    if (where.any()):
        amx=ad.average('amx',where=where,weights='mass')
        amy=ad.average('amy',where=where,weights='mass')
        amz=ad.average('amz',where=where,weights='mass')
    if (amx==0.0 and amy==0.0 and amz==0.0):
        amx, amy, amz = 0.0, 0.0, 1.0
    def normal(vec):
        return vec/np.sqrt(np.sum(vec**2))
    disk_z = normal(np.array([amx,amy,amz]))
    y_01=np.asarray([0,1])
    disk_y=normal(np.array([y_01[0],y_01[1],(-y_01[0]*disk_z[0]-y_01[1]*disk_z[1])/disk_z[2]]))
    disk_x=normal(np.cross(disk_y,disk_z))
    ad.tran=inv(np.stack((disk_x,disk_y,disk_z)).T)
    return

def add_data(ad,add_bcc=True):
    for var in ['bcc1','bcc2','bcc3']:
        if ((var not in ad.data_raw.keys()) and add_bcc):
                ad.add_data_func(var, lambda d : d('zeros'))
    
    ad.add_data_func('tran_x', lambda d : d.ad.tran[0,0]*d('x')+d.ad.tran[0,1]*d('y')+d.ad.tran[0,2]*d('z'))
    ad.add_data_func('tran_y', lambda d : d.ad.tran[1,0]*d('x')+d.ad.tran[1,1]*d('y')+d.ad.tran[1,2]*d('z'))
    ad.add_data_func('tran_z', lambda d : d.ad.tran[2,0]*d('x')+d.ad.tran[2,1]*d('y')+d.ad.tran[2,2]*d('z'))
    
    ad.add_data_func('tran_velx', lambda d : d.ad.tran[0,0]*d('velx')+d.ad.tran[0,1]*d('vely')+d.ad.tran[0,2]*d('velz'))
    ad.add_data_func('tran_vely', lambda d : d.ad.tran[1,0]*d('velx')+d.ad.tran[1,1]*d('vely')+d.ad.tran[1,2]*d('velz'))
    ad.add_data_func('tran_velz', lambda d : d.ad.tran[2,0]*d('velx')+d.ad.tran[2,1]*d('vely')+d.ad.tran[2,2]*d('velz'))

    ad.add_data_func('tran_bccx', lambda d : d.ad.tran[0,0]*d('bccx')+d.ad.tran[0,1]*d('bccy')+d.ad.tran[0,2]*d('bccz'))
    ad.add_data_func('tran_bccy', lambda d : d.ad.tran[1,0]*d('bccx')+d.ad.tran[1,1]*d('bccy')+d.ad.tran[1,2]*d('bccz'))
    ad.add_data_func('tran_bccz', lambda d : d.ad.tran[2,0]*d('bccx')+d.ad.tran[2,1]*d('bccy')+d.ad.tran[2,2]*d('bccz'))

    ad.add_data_func('tran_R', lambda d : xp.sqrt(d('tran_x')**2+d('tran_y')**2))
    ad.add_data_func('tran_velR', lambda d : (d('tran_x')*d('tran_velx')+d('tran_y')*d('tran_vely'))/d('tran_R'))
    ad.add_data_func('tran_velphi', lambda d : (d('tran_x')*d('tran_vely')-d('tran_y')*d('tran_velx'))/d('tran_R'))
    ad.add_data_func('tran_bccR', lambda d : (d('tran_x')*d('tran_bccx')+d('tran_y')*d('tran_bccy'))/d('tran_R'))
    ad.add_data_func('tran_bccphi', lambda d : (d('tran_x')*d('tran_bccy')-d('tran_y')*d('tran_bccx'))/d('tran_R'))

    ad.add_data_func('tran_stress_zphi_hydro', lambda d : d('dens')*d('tran_velz')*d('tran_velphi'))
    ad.add_data_func('tran_stress_zphi_maxwell', lambda d : -d('tran_bccz')*d('tran_bccphi'))
    ad.add_data_func('tran_stress_zphi', lambda d : d('tran_stress_zphi_hydro')+d('tran_stress_zphi_maxwell'))

    ad.add_data_func('tran_stress_Rphi_hydro', lambda d : d('dens')*d('tran_velR')*d('tran_velphi'))
    ad.add_data_func('tran_stress_Rphi_maxwell', lambda d : -d('tran_bccR')*d('tran_bccphi'))
    ad.add_data_func('tran_stress_Rphi', lambda d : d('tran_stress_Rphi_hydro')+d('tran_stress_Rphi_maxwell'))

    ad.add_data_func('tran_Omega', lambda d : xp.sqrt(d.ad.accel(d('tran_R'))/d('tran_R')))
    ad.add_data_func('tran_dens_velR', lambda d : d('dens')*d('tran_velR'))
    ad.add_data_func('tran_dens_velz', lambda d : d('dens')*d('tran_velz'))
    ad.add_data_func('tran_radial_flow', lambda d : 0.5*d('dens')*d('tran_velR')*d('tran_Omega'))
    ad.add_data_func('tran_z/R', lambda d : d('tran_z')/d('tran_R'))

    ad.add_data_func('tran_stress_zphi_hydro/R', lambda d : d('tran_stress_zphi_hydro')/d('tran_R'))
    ad.add_data_func('tran_stress_zphi_maxwell/R', lambda d : d('tran_stress_zphi_maxwell')/d('tran_R'))
    ad.add_data_func('tran_stress_zphi/R', lambda d : d('tran_stress_zphi')/d('tran_R'))
    ad.add_data_func('tran_stress_Rphi_hydro/R', lambda d : d('tran_stress_Rphi_hydro')/d('tran_R'))
    ad.add_data_func('tran_stress_Rphi_maxwell/R', lambda d : d('tran_stress_Rphi_maxwell')/d('tran_R'))
    ad.add_data_func('tran_stress_Rphi/R', lambda d : d('tran_stress_Rphi')/d('tran_R'))

    ad.add_data_func('dens_initial', lambda d : xp.interp(d('r'),xp.asarray(d.ad.rad_initial['r']),xp.asarray(d.ad.rad_initial['dens'])))
    ad.add_data_func('temp_initial', lambda d : xp.interp(d('r'),xp.asarray(d.ad.rad_initial['r']),xp.asarray(d.ad.rad_initial['temp'])))
    ad.add_data_func('vel_kep', lambda d : xp.interp(d('r'),xp.asarray(d.ad.rad_initial['r']),xp.asarray(d.ad.rad_initial['v_kep'])))
    ad.add_data_func('t_hot', lambda d : d.ad.header('problem','tf_hot',float)*d('temp_initial'))

    '''
    for var in ['mdot','mdotin','mdotout','momdot','momdotin','momdotout','eidot','eidotin','eidotout','ekdot','ekdotin','ekdotout']:
        ad.add_data_func(var, lambda d, var=var : 4.0*xp.pi*d('r')**2*d(var.replace('dot','flxr')))
        ad.add_data_func(var+'_cold', lambda d, var=var : d(var)*(d('temp')<d.ad.header('problem','t_cold',float)))
        ad.add_data_func(var+'_warm', lambda d, var=var : d(var)*(d('temp')>=d.ad.header('problem','t_cold',float))*(d('temp')<d('t_hot')))
        ad.add_data_func(var+'_hot', lambda d, var=var : d(var)*(d('temp')>=d('t_hot')))
    '''
    for key,inte in zip(['mdot','momdot','eidot','ekdot','edot'],
                        ['dens','momr',  'eint', 'ekin', 'etot']):
        for direc in ['','in','out']:
            var = key+direc
            ad.add_data_func(var, lambda d, inte=inte, direc=direc : 4.0*xp.pi*d('r')**2*d(inte)*d('velr'+direc))
            ad.add_data_func(var+'_cold', lambda d, var=var : d(var)*(d('temp')<d.ad.header('problem','t_cold',float)))
            ad.add_data_func(var+'_warm', lambda d, var=var : d(var)*(d('temp')>=d.ad.header('problem','t_cold',float))*(d('temp')<d('t_hot')))
            ad.add_data_func(var+'_hot', lambda d, var=var : d(var)*(d('temp')>=d('t_hot')))

    return
