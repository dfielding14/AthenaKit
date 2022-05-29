import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# units
cm_cgs = 1.0;                           # cm
pc_cgs = 3.0856775809623245e+18;        # cm
kpc_cgs = 3.0856775809623245e+21;       # cm
g_cgs = 1.0;                            # g
msun_cgs = 1.98841586e+33;              # g
atomic_mass_unit_cgs = 1.660538921e-24; # g
s_cgs = 1.0;                            # s
yr_cgs = 3.15576e+7;                    # s
myr_cgs = 3.15576e+13;                  # s
cm_s_cgs = 1.0;                         # cm/s
km_s_cgs = 1.0e5;                       # cm/s
g_cm3_cgs = 1.0;                        # g/cm^3
erg_cgs = 1.0;                          # erg
dyne_cm2_cgs = 1.0;                     # dyne/cm^2  
kelvin_cgs = 1.0;                       # k
k_boltzmann_cgs = 1.3806488e-16;        # erg/k
grav_constant_cgs = 6.67408e-8;         # cm^3/(g*s^2)
speed_of_light_cgs = 2.99792458e10      # cm/s
class Units:
    def __init__(self,lunit=pc_cgs,munit=atomic_mass_unit_cgs*pc_cgs**3,tunit=myr_cgs,mu=1.0):
        self.length_cgs=lunit
        self.mass_cgs=munit
        self.time_cgs=tunit
        self.mu=mu
    @property
    def velocity_cgs(self):
        return self.length_cgs/self.time_cgs
    @property
    def density_cgs(self):
        return self.mass_cgs/self.length_cgs**3
    @property
    def energy_cgs(self):
        return self.mass_cgs*self.velocity_cgs**2
    @property
    def pressure_cgs(self):
        return self.energy_cgs/self.length_cgs**3
    @property
    def temperature_cgs(self):
        return self.velocity_cgs**2*self.mu*atomic_mass_unit_cgs/k_boltzmann_cgs
    @property
    def grav_constant(self):
        return grav_constant_cgs*self.density_cgs*self.time_cgs**2
    @property
    def speed_of_light(self):
        return speed_of_light_cgs/self.velocity_cgs
    @property
    def number_density_cgs(self):
        return self.density_cgs/self.mu/atomic_mass_unit_cgs
    @property
    def cooling_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs**2
    @property
    def heating_cgs(self):
        return self.pressure_cgs/self.time_cgs/self.number_density_cgs
    @property
    def conductivity_cgs(self):
        return self.pressure_cgs*self.velocity_cgs*self.length_cgs/self.temperature_cgs

mu=0.618
unit=Units(lunit=kpc_cgs,munit=mu*atomic_mass_unit_cgs*kpc_cgs**3,mu=mu)

@np.vectorize
def CoolFnShure(T):
    if(np.isnan(T)):
        return 0.0
    # original data from Shure et al. paper, covers 4.12 < logt < 8.16
    lhd = [
      -22.5977, -21.9689, -21.5972, -21.4615, -21.4789, -21.5497, -21.6211, -21.6595,
      -21.6426, -21.5688, -21.4771, -21.3755, -21.2693, -21.1644, -21.0658, -20.9778,
      -20.8986, -20.8281, -20.7700, -20.7223, -20.6888, -20.6739, -20.6815, -20.7051,
      -20.7229, -20.7208, -20.7058, -20.6896, -20.6797, -20.6749, -20.6709, -20.6748,
      -20.7089, -20.8031, -20.9647, -21.1482, -21.2932, -21.3767, -21.4129, -21.4291,
      -21.4538, -21.5055, -21.5740, -21.6300, -21.6615, -21.6766, -21.6886, -21.7073,
      -21.7304, -21.7491, -21.7607, -21.7701, -21.7877, -21.8243, -21.8875, -21.9738,
      -22.0671, -22.1537, -22.2265, -22.2821, -22.3213, -22.3462, -22.3587, -22.3622,
      -22.3590, -22.3512, -22.3420, -22.3342, -22.3312, -22.3346, -22.3445, -22.3595,
      -22.3780, -22.4007, -22.4289, -22.4625, -22.4995, -22.5353, -22.5659, -22.5895,
      -22.6059, -22.6161, -22.6208, -22.6213, -22.6184, -22.6126, -22.6045, -22.5945,
      -22.5831, -22.5707, -22.5573, -22.5434, -22.5287, -22.5140, -22.4992, -22.4844,
      -22.4695, -22.4543, -22.4392, -22.4237, -22.4087, -22.3928]
    #  for temperatures less than 10^4 K, use Koyama & Inutsuka
    logt=np.log10(T)
    if (logt <= 4.2):
        temp = pow(10.0, logt)
        return (2.0e-19*np.exp(-1.184e5/(temp + 1.0e3)) + 2.8e-28*np.sqrt(temp)*np.exp(-92.0/temp))
    
    #for temperatures above 10^8.15 use CGOLS fit
    if (logt > 8.15): return pow(10.0, (0.45*logt - 26.065))
    
    # in between values of 4.2 < log(T) < 8.15
    # linear interpolation of tabulated SPEX cooling rate
    
    ipps = int(25.0*logt) - 103
    ipps = ipps if (ipps < 100) else 100
    ipps = ipps if (ipps > 0  ) else 0
    
    x0    = 4.12 + 0.04*ipps
    dx    = logt - x0
    tcool = (lhd[ipps+1]*dx - lhd[ipps]*(dx - 0.04))*25.0
    return pow(10.0,tcool)

# profile solver
def NFWMass(r,ms,rs):
    return ms*(np.log(1+r/rs)-r/(rs+r))
def TotMass(r,m,mc,rc,ms,rs):
    return m+NFWMass(r,mc,rc)+NFWMass(r,ms,rs)
def Acceleration(r,m,mc,rc,ms,rs,g):
    return -g*(TotMass(r,m,mc,rc,ms,rs))/r**2
def DrhoDr(x,rho):
    r = x*rad_entry
    accel = rad_entry*Acceleration(r,m_bh,m_star,r_star,m_dm,r_dm,unit.grav_constant)
    #print(accel)
    #print(rho,gamma,xi)
    grad = (2*rho**(2-gamma)*accel/k_0-rho*xi*x**(xi-1))/((1+x**xi)*gamma)
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
def SolveDens(N=2048,logh=0.002):
    N2=int(N/2)
    dens_arr = np.zeros(N)
    dens = dens_entry
    dens_arr[N2]=dens
    for i in range(N2):
        x = 10**(-i*logh)
        h = 10**(-(i+1)*logh)-x
        dens = RK4(DrhoDr,x,dens,h)
        dens_arr[N2-i-1]=dens
    dens = dens_entry
    for i in range(N2-1):
        x = 10**(i*logh)
        h = 10**((i+1)*logh)-x
        dens = RK4(DrhoDr,x,dens,h)
        dens_arr[N2+i+1]=dens
    xs=np.logspace(-logh*N2,logh*(N2-1),N,endpoint=True)
    pres_arr = 0.5*k_0*(1.0+pow(xs,xi))*pow(dens_arr,gamma)
    rss=dict()
    rss['r']=xs*rad_entry
    rss['dens']=dens_arr
    rss['pres']=pres_arr
    rss['temp']=rss['pres']/rss['dens']
    rss['entropy']=rss['pres']/rss['dens']**(5/3)
    return rss

# solve
gamma       = 5/3     # gamma
potential   = 1       # turn on potential
m_bh        = 4.3e2   # BH mass: 6.5e9 Msun # Mass unit: 1.516e7 Msun
r_in        = 0.0     # inner radius: 100 pc
m_star      = 2e4     # 3e11 Msun # Mass unit: 1.516e7 Msun
r_star      = 2.0     # 2kpc
m_dm        = 3.0e6   # DM Mass: 4.5e13 Msun # Mass unit: 1.516e7 Msun
r_dm        = 80.0    # 2kpc
sink_d      = 1e-2    # density of sink cells
sink_t      = 1e-1    # eint/dens=1/gm1*T: temperature of sink cells
rad_entry   = 2.0
dens_entry  = 1e-1
#k_0         = 1.0
#xi          = 1.75
k_0         = 1.1
xi          = 1.1


ran=SolveDens(N=4096)
ran['mass']=TotMass(ran['r'],m_bh,m_star,r_star,m_dm,r_dm)
ran['g']=Acceleration(ran['r'],m_bh,m_star,r_star,m_dm,r_dm,unit.grav_constant)
ran['t_ff']=np.sqrt(2.*ran['r']/-ran['g'])
ran['v_ff']=np.sqrt(2.*ran['r']*-ran['g'])
ran['v_kep']=np.sqrt(ran['r']*-ran['g'])
ran['am_kep']=np.sqrt(ran['r']**3*-ran['g'])
ran['potential']=ran['r']*-ran['g']
ran['r_B']=unit.grav_constant*m_bh/(gamma*ran['temp'])
ran['Mdot_B']=np.pi*(unit.grav_constant*m_bh)**2*ran['dens']/(gamma*ran['temp'])**1.5

##########################################################################################
## cooling time as a function of temperature and density
##########################################################################################
def rho_T_t_cool(cooling_rho=np.logspace(-4,4,400),cooling_temp=np.logspace(0,8,400)):
    cooling_rho,cooling_temp=np.meshgrid(cooling_rho,cooling_temp)
    cooling_tcool=k_boltzmann_cgs*cooling_temp/cooling_rho/CoolFnShure(cooling_temp)/(gamma-1)/myr_cgs
    return cooling_rho,cooling_temp,cooling_tcool

##########################################################################################
## SNR
##########################################################################################

# analytical function
class SNR:
    def __init__(self,E,n,M=3):
        #E [erg]
        #n cm^-3
        #M Msun
        self.E=E
        self.n=n
        self.t_free=4.64e-4*E**(-1/2)*n**(-1/3)
        self.r_free=2.75*n**(-1/3)
        self.t_sf=0.044*E**0.22*n**-0.55
        self.r_sf=22.6*E**0.29*n**-0.42
        self.mom_sf=2.17e5*E**0.93*n**-0.13
        return
    
    @np.vectorize
    def _r(self,t):
        if(t<=self.t_sf):
            return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
        else:
            return self.r_sf*(t/self.t_sf)**(2/7)
        #else:
        #    return 30*(t/0.1)**(2/7)
    def r(self,t):
        return self._r(self,t)
##########################################################################################
## plots
##########################################################################################
def ave(a,n):
    end = -(a.size%n) if(-(a.size%n)) else None
    return np.average((a.ravel()[:end]).reshape(-1,n),axis=1)
##########################################################################################
## plots
##########################################################################################
def colors(n,x2=0.88):
    colors = ['k','darkviolet',  'b', 'royalblue', 'c', 'g', 'springgreen', 'gold', 'y',  
          'salmon','pink', 'r','darkred', 'm', 'violet',]
    cmap = plt.cm.nipy_spectral  # define the colormap
    #cmap = plt.cm.terrain  # define the colormap
    #cmap = plt.cm.viridis  # define the colormap
    colors=[tuple(np.array(cmap(i/max(1,n-1)*x2))/1.01) for i in range(n)]
    return colors
def figure(nrows=1,ncols=1,figsize=(6.4,4.8),dpi=120,sharex=True,squeeze=False,\
    constrained_layout=False,top=0.94, wspace=0.02, hspace=0.0):
    fig, axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,sharex=sharex,\
                         constrained_layout=constrained_layout,squeeze=squeeze)
    fig.subplots_adjust(top=top, wspace=wspace, hspace=hspace)
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    for ax in axes.flat:
        ax.grid(linestyle='--')
        ax.tick_params(top=True,right=True,which='both',direction="in")
    if(ncols>1):
        for ax in axes[:,-1]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
    return fig,ax
def subplots(nrows=2,ncols=2,figsize=(7.2,5.0),dpi=120,sharex=True,squeeze=False,\
    constrained_layout=False,top=0.94, wspace=0.02, hspace=0.0,raw=False):
    fig, axes = plt.subplots(nrows,ncols,figsize=figsize,dpi=dpi,sharex=sharex,\
                         constrained_layout=constrained_layout,squeeze=squeeze)
    fig.subplots_adjust(top=top, wspace=wspace, hspace=hspace)
    if (raw): return fig,axes
    #fig.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.2)
    for ax in axes.flat:
        ax.grid(linestyle='--')
        ax.tick_params(top=True,right=True,which='both',direction="in")
    if(ncols>1):
        for ax in axes[:,-1]:
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
    return fig,axes
